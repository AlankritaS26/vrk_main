"""
backend/companion.py — QR Companion Feature
============================================

Endpoints mounted on the main FastAPI app via `app.include_router(companion_router)`:

  POST /companion/token            — issue a short-lived companion token for the active session
  GET  /companion/validate/{token} — validate token, return session metadata + summary
  GET  /companion/brochure/{token} — stream the best-matched brochure PDF
  GET  /companion/{token}          — serve the mobile landing page HTML

Token strategy: opaque 32-byte hex secret stored in Redis as:
    Key:   companion:{token}
    Value: JSON-encoded session snapshot
    TTL:   COMPANION_TOKEN_TTL_MINUTES (default 20 min, configurable)

Client-side QR rendering (WelcomeScreen.js) calls POST /companion/token once on
session start and renders the returned URL using the qrcode npm package. This
endpoint also stores the URL in Redis so the mobile page can retrieve it.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("RNSIT_Kiosk.Companion")

# ── Configuration ────────────────────────────────────────────────────────────
# ── Configuration ────────────────────────────────────────────────────────────
COMPANION_TOKEN_TTL_MINUTES: int = int(os.getenv("COMPANION_TOKEN_TTL_MINUTES", "20"))


def _get_lan_ip() -> str:
    """Return the machine's local LAN IP (e.g. 192.168.x.x) so mobile devices on Wi-Fi can connect."""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_companion_base_url() -> str:
    explicit = os.getenv("COMPANION_BASE_URL", "").strip().rstrip("/")
    if explicit and not ("127.0.0.1" in explicit or "localhost" in explicit):
        return explicit
    lan_ip = _get_lan_ip()
    return f"http://{lan_ip}:8001"


COMPANION_BASE_URL: str = get_companion_base_url()

BROCHURES_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "brochures"

# Topic keyword → brochure filename mapping
BROCHURE_TOPIC_MAP: dict[str, str] = {
    "cse":          "cse.pdf",
    "computer":     "cse.pdf",
    "cs":           "cse.pdf",
    "ece":          "ece.pdf",
    "electronics":  "ece.pdf",
    "communication":"ece.pdf",
    "ise":          "ise.pdf",
    "information":  "ise.pdf",
    "me":           "me.pdf",
    "mechanical":   "me.pdf",
    "civil":        "civil.pdf",
    "eee":          "eee.pdf",
    "electrical":   "eee.pdf",
    "mba":          "mba.pdf",
    "management":   "mba.pdf",
    "admission":    "admissions.pdf",
    "fee":          "fees.pdf",
    "fees":         "fees.pdf",
    "hostel":       "hostel.pdf",
    "placement":    "placements.pdf",
    "library":      "general.pdf",
    "sports":       "general.pdf",
    "canteen":      "general.pdf",
}

# ── LLM / RAG imports (lazy — avoids circular import at module load) ─────────
_RAG_SERVICE_URL: str = os.getenv("RAG_SERVICE_URL", "http://localhost:8600").rstrip("/")
_RAG_COLLECTION: str  = os.getenv("RAG_COLLECTION", "kiosk-rnsit")

router = APIRouter(prefix="/companion", tags=["companion"])


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_redis():
    """Return the shared Redis client from main.py (avoids circular import)."""
    try:
        from backend.main import redis_client
        return redis_client
    except Exception:
        return None


def _companion_key(token: str) -> str:
    return f"companion:token:{token}"


def _issue_token(session_snapshot: dict) -> str:
    """Mint a new companion token, store in Redis, return the token string."""
    token = secrets.token_hex(20)
    rc = _get_redis()
    if rc:
        try:
            rc.set(
                _companion_key(token),
                json.dumps(session_snapshot, default=str),
                ex=COMPANION_TOKEN_TTL_MINUTES * 60,
            )
        except Exception as e:
            logger.warning("[COMPANION] Redis write failed: %s", e)
    else:
        # Fallback: store in module-level dict (dev/no-Redis mode)
        _TOKEN_STORE[token] = session_snapshot
    logger.info(
        "[COMPANION] Token issued for session=%s TTL=%dm",
        (session_snapshot.get("session_id") or "?")[:8],
        COMPANION_TOKEN_TTL_MINUTES,
    )
    return token


# In-memory fallback when Redis is unavailable
_TOKEN_STORE: dict[str, dict] = {}


def _lookup_token(token: str) -> dict | None:
    """Return the session snapshot for a token, or None if expired/invalid."""
    rc = _get_redis()
    if rc:
        try:
            raw = rc.get(_companion_key(token))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("[COMPANION] Redis read failed: %s", e)
    # Fallback
    return _TOKEN_STORE.get(token)


def _revoke_token(token: str) -> None:
    rc = _get_redis()
    if rc:
        try:
            rc.delete(_companion_key(token))
        except Exception:
            pass
    _TOKEN_STORE.pop(token, None)


def _invalidate_session_tokens(session_id: str) -> None:
    """Revoke all companion tokens belonging to a session (called on session end)."""
    rc = _get_redis()
    if rc:
        try:
            # Scan for any keys matching companion:token:* and check their session_id
            cursor = 0
            while True:
                cursor, keys = rc.scan(cursor, match="companion:token:*", count=100)
                for k in keys:
                    raw = rc.get(k)
                    if raw:
                        snap = json.loads(raw)
                        if snap.get("session_id") == session_id:
                            rc.delete(k)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("[COMPANION] Token cleanup scan failed: %s", e)
    # Also clean in-memory fallback store
    to_delete = [t for t, snap in _TOKEN_STORE.items()
                 if snap.get("session_id") == session_id]
    for t in to_delete:
        del _TOKEN_STORE[t]


def _pick_brochure(topics: str) -> Path:
    """Return the Path of the best-matching brochure PDF for the given topic string."""
    topics_lower = topics.lower()
    for keyword, filename in BROCHURE_TOPIC_MAP.items():
        if keyword in topics_lower:
            candidate = BROCHURES_DIR / filename
            if candidate.exists():
                return candidate
    # Fallback to general
    general = BROCHURES_DIR / "general.pdf"
    if general.exists():
        return general
    return BROCHURES_DIR / "general.pdf"


def _clean_user_name(raw: str) -> str:
    """Sanitize visitor name, rejecting false/incidental names."""
    if not raw or not isinstance(raw, str):
        return "Visitor"
    cleaned = raw.strip()
    lower = cleaned.lower()
    if lower in ("guest", "unknown", "there", "none", "null", "undefined"):
        return "Visitor"
    if any(bad in lower for bad in ("student of", "all right", "alright", "doctor", "bengaluru", "bangalore", "wait", "stop", "task")):
        return "Visitor"
    words = [w for w in cleaned.split() if w.isalpha()]
    if 1 <= len(words) <= 3:
        return " ".join(words)
    return "Visitor"


async def _summarize_interactions(interactions: list[dict], user_name: str = "Guest") -> str:
    """
    Produce an accurate, personalized 2–4 sentence recap of what the visitor discussed.
    Uses chat_completion_with_fallback directly (never RAG retrieval) to prevent hallucination.
    """
    user_name = _clean_user_name(user_name)

    if not interactions:
        return (
            f"Welcome to RNSIT, {user_name}! You have started an interaction with Nova, the digital receptionist. "
            "Ask questions at the kiosk about admissions, courses, departments, placements, or facilities, "
            "and your live conversation summary will appear here."
        )

    # Build clean transcript, filtering out internal name changes and noise
    lines = []
    questions = []
    has_staff_request = False

    for ia in interactions[-12:]:
        q = (ia.get("input_text") or "").strip()
        a = (ia.get("response_text") or "").strip()
        if not q:
            continue
        # Skip name change operations from polluting the session recap
        if "changed your name to" in a or re.search(r"\b(?:change|update|set|rename)\s+(?:my\s+)?name\b", q, re.I):
            continue
        if re.search(r"\b(?:staff|human|person|reception|front\s*desk|supervisor)\b", q, re.I):
            has_staff_request = True
        questions.append(q)
        lines.append(f"Visitor: {q}")
        if a:
            lines.append(f"Nova: {a}")

    if not lines:
        if has_staff_request:
            return f"{user_name}, during your visit to RNSIT, you requested to connect with our front desk staff. Our reception team at the Admin Block is available to assist you."
        return f"{user_name}, you had an interaction with Nova at the RNSIT reception kiosk."

    transcript_text = "\n".join(lines)

    try:
        from backend.llm import chat_completion_with_fallback
        name_clause = f" for {user_name}" if user_name and user_name != "Visitor" else ""
        staff_note = "Explicitly note that the visitor requested to connect with front desk staff for direct assistance. " if has_staff_request else ""
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are the session summarizer for the RNS Institute of Technology (RNSIT) digital receptionist kiosk. "
                    f"The visitor's name is '{user_name}'. "
                    "Write an accurate, friendly, and concise 2 to 3 sentence recap of what the visitor asked about "
                    "and the key campus information Nova provided (e.g. courses, admissions, facilities, hostel). "
                    f"Address the visitor directly using their name (e.g., '{user_name}, during your visit to RNSIT, you asked about...'). "
                    f"{staff_note}"
                    "Stick strictly to the facts mentioned in the transcript below. "
                    "Do NOT invent details, do NOT include greetings or farewells, and do NOT repeat back the transcript verbatim."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation transcript:\n{transcript_text}\n\nWrite a personalized summary{name_clause}:",
            },
        ]
        res = await chat_completion_with_fallback(messages=messages, temperature=0.2, max_tokens=250)
        summary = res[0] if isinstance(res, (tuple, list)) else res
        if summary and isinstance(summary, str):
            cleaned = re.sub(r"^(?:Summary:|Nova:|Recap:|Here is|Here's)\s*", "", summary.strip(), flags=re.I).strip()
            if cleaned:
                return cleaned
    except Exception as e:
        logger.warning("[COMPANION] LLM summarization failed: %s", e)

    # Fallback to accurate bullet list of questions asked
    if questions:
        bullets = "\n".join(f"• {q}" for q in questions[-5:])
        staff_msg = "\n• Requested human staff connection at the front desk." if has_staff_request else ""
        return f"During your visit to the RNSIT kiosk, you inquired about:\n{bullets}{staff_msg}\n\nFeel free to explore the official brochures below or contact the campus admissions desk."

    return f"You had a session at the RNSIT kiosk."


def _extract_topics_from_interactions(interactions: list[dict]) -> str:
    """Simple keyword extraction from visitor questions for brochure matching."""
    combined = " ".join(
        ia.get("input_text", "").lower()
        for ia in interactions
        if ia.get("input_text")
    )
    return combined


def _generate_placeholder_pdf(title: str, topics: str, summary: str) -> bytes:
    """
    Generate a minimal brochure PDF with ReportLab when no static file exists.
    This is the fallback for sessions with no matching pre-built brochure.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2.5*cm, rightMargin=2.5*cm,
                                topMargin=2.5*cm, bottomMargin=2.5*cm)
        styles = getSampleStyleSheet()

        header_style = ParagraphStyle(
            "Header", parent=styles["Heading1"],
            fontSize=22, textColor=colors.HexColor("#1a56db"),
            spaceAfter=6, alignment=TA_CENTER,
        )
        sub_style = ParagraphStyle(
            "Sub", parent=styles["Normal"],
            fontSize=11, textColor=colors.HexColor("#6b7280"),
            spaceAfter=20, alignment=TA_CENTER,
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=11, leading=18, textColor=colors.HexColor("#1f2937"),
            spaceAfter=12,
        )
        label_style = ParagraphStyle(
            "Label", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#6b7280"),
            spaceBefore=24, spaceAfter=4,
        )

        story = [
            Spacer(1, 0.5*cm),
            Paragraph("RNS Institute of Technology", header_style),
            Paragraph("Bengaluru – 560098 | www.rnsit.ac.in", sub_style),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")),
            Spacer(1, 0.5*cm),
            Paragraph(title, ParagraphStyle(
                "Title", parent=styles["Heading2"],
                fontSize=16, textColor=colors.HexColor("#111827"),
                spaceAfter=16,
            )),
            Paragraph(label_style.spaceAfter and "YOUR SESSION SUMMARY", label_style),
            Paragraph(summary, body_style),
            Spacer(1, 0.5*cm),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")),
            Paragraph("CONTACT US", label_style),
            Paragraph(
                "📞 +91-80-23190000 &nbsp;&nbsp; ✉️ principal@rnsit.ac.in",
                body_style,
            ),
            Paragraph(
                "🌐 www.rnsit.ac.in &nbsp;&nbsp; 📍 Dr. Vishnuvardhan Road, Channasandra, Bengaluru",
                body_style,
            ),
            Spacer(1, 0.5*cm),
            Paragraph(
                f"<i>Generated on {datetime.now().strftime('%d %B %Y')} · RNSIT Digital Kiosk</i>",
                ParagraphStyle("Footer", parent=styles["Normal"],
                               fontSize=8, textColor=colors.HexColor("#9ca3af"),
                               alignment=TA_CENTER),
            ),
        ]

        doc.build(story)
        return buf.getvalue()

    except Exception as e:
        logger.error("[COMPANION] ReportLab PDF generation failed: %s", e)
        return b""


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/token")
async def issue_companion_token():
    """
    Mint a companion token for the currently active kiosk session.
    Called by WelcomeScreen.js on session start (or on demand).
    Returns: { token, url, expires_in_seconds }
    """
    # Import active_session lazily to avoid circular import
    try:
        from backend.main import active_session
    except ImportError:
        active_session = None

    if not active_session:
        raise HTTPException(status_code=404, detail="No active session to generate a companion token for.")

    snapshot = {
        "session_id": active_session.get("session_id", ""),
        "user_name":  active_session.get("user_name", "Guest"),
        "face_id":    active_session.get("face_id", ""),
        "issued_at":  datetime.now().isoformat(),
    }

    token = _issue_token(snapshot)
    url = f"{COMPANION_BASE_URL}/companion/{token}"

    return {
        "token":             token,
        "url":               url,
        "expires_in_seconds": COMPANION_TOKEN_TTL_MINUTES * 60,
    }


@router.get("/qr/{token}")
async def get_companion_qr_image(token: str):
    """
    Generate and stream a PNG QR code for the given companion token URL.
    Used by the kiosk WelcomeScreen to display the QR code without external client-side libs.
    """
    try:
        import qrcode
        url = f"{COMPANION_BASE_URL}/companion/{token}"
        img = qrcode.make(url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error("[COMPANION] QR generation error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate QR code.")


@router.get("/validate/{token}")
async def validate_companion_token(token: str):
    """
    Validate a companion token and return the enriched session data.
    Called by the mobile landing page JS to populate the summary view.
    """
    snapshot = _lookup_token(token)
    if not snapshot:
        raise HTTPException(status_code=410, detail="Token expired or invalid.")

    session_id = snapshot.get("session_id", "")
    user_name  = snapshot.get("user_name", "Guest")
    face_id    = snapshot.get("face_id", "")

    # Fetch interaction history from MongoDB (most recent 25 turns)
    try:
        from backend.database import interactions_collection
        cursor = interactions_collection.find(
            {"session_id": session_id},
            sort=[("timestamp", -1)],
            limit=25,
        )
        interactions = [doc async for doc in cursor]
        if not interactions and face_id:
            cursor = interactions_collection.find(
                {"face_id": face_id},
                sort=[("timestamp", -1)],
                limit=25,
            )
            interactions = [doc async for doc in cursor]
        interactions.reverse()
        # Serialize ObjectId
        for doc in interactions:
            doc.pop("_id", None)
    except Exception as e:
        logger.warning("[COMPANION] Could not fetch interactions: %s", e)
        interactions = []

    # Merge live in-memory turns from active_session if current
    try:
        from backend.main import active_session, message_log
        if active_session and (
            active_session.get("session_id") == session_id
            or (face_id and active_session.get("face_id") == face_id)
        ):
            live_pairs = []
            cur_q = None
            for m in (message_log or []):
                spk = m.get("speaker", "")
                txt = m.get("text", "")
                if spk in ("user", "visitor"):
                    cur_q = txt
                elif spk in ("kiosk", "nova") and cur_q:
                    live_pairs.append({
                        "session_id": session_id,
                        "input_text": cur_q,
                        "response_text": txt,
                        "user_name": user_name,
                        "timestamp": m.get("timestamp", datetime.now().isoformat()),
                    })
                    cur_q = None
            if live_pairs:
                # Always use live pairs for the current session — they reflect
                # exactly what was said, avoiding stale DB data from prior sessions
                # with the same face_id that outnumber the current live turns.
                existing_qs = {ia.get("input_text", "") for ia in interactions
                               if ia.get("session_id") == session_id}
                for lp in live_pairs:
                    if lp["input_text"] not in existing_qs:
                        interactions.append(lp)
                        existing_qs.add(lp["input_text"])
                # If no DB rows for current session exist at all, use live only
                db_has_current = any(ia.get("session_id") == session_id for ia in interactions)
                if not db_has_current:
                    interactions = live_pairs
    except Exception as ex:
        logger.warning("[COMPANION] Live message merge failed: %s", ex)

    # Always prefer the active session name (most up-to-date)
    try:
        from backend.main import active_session
        if active_session and (
            active_session.get("session_id") == session_id
            or (face_id and active_session.get("face_id") == face_id)
        ):
            live_name = active_session.get("user_name")
            if live_name and live_name not in ("Guest", "Unknown", ""):
                user_name = live_name
    except Exception:
        pass

    # Fall back to checking DB interactions for a real name
    if user_name in ("Guest", "Unknown", "Visitor", ""):
        for ia in interactions:
            ia_name = ia.get("user_name")
            if ia_name and ia_name not in ("Guest", "Unknown", ""):
                user_name = ia_name
                break

    user_name = _clean_user_name(user_name)
    summary = await _summarize_interactions(interactions, user_name=user_name)
    topics  = _extract_topics_from_interactions(interactions)

    # Pick best brochure
    brochure_path = _pick_brochure(topics)
    brochure_available = brochure_path.exists()
    brochure_url = f"{COMPANION_BASE_URL}/companion/brochure/{token}" if brochure_available or True else None

    return {
        "valid":              True,
        "user_name":          user_name,
        "session_id":         session_id,
        "summary":            summary,
        "topics":             topics[:200],
        "brochure_url":       brochure_url,
        "brochure_available": True,
        "issued_at":          snapshot.get("issued_at", ""),
        "expires_in_minutes": COMPANION_TOKEN_TTL_MINUTES,
    }


@router.get("/brochure/{token}")
async def get_companion_brochure(token: str):
    """
    Serve the best-matched brochure PDF for this session.
    Falls back to a dynamically generated summary PDF via ReportLab.
    """
    snapshot = _lookup_token(token)
    if not snapshot:
        raise HTTPException(status_code=410, detail="Token expired or invalid.")

    session_id = snapshot.get("session_id", "")
    user_name  = snapshot.get("user_name", "Guest")
    face_id    = snapshot.get("face_id", "")

    # Fetch interactions for topic detection + summary
    try:
        from backend.database import interactions_collection
        cursor = interactions_collection.find(
            {"session_id": session_id},
            sort=[("timestamp", 1)],
            limit=25,
        )
        interactions = [doc async for doc in cursor]
        if not interactions and face_id:
            cursor = interactions_collection.find(
                {"face_id": face_id},
                sort=[("timestamp", 1)],
                limit=25,
            )
            interactions = [doc async for doc in cursor]
        for doc in interactions:
            doc.pop("_id", None)
    except Exception as e:
        logger.warning("[COMPANION] Brochure: could not fetch interactions: %s", e)
        interactions = []

    # Check for updated user_name
    for ia in interactions:
        ia_name = ia.get("user_name")
        if ia_name and ia_name not in ("Guest", "Unknown", ""):
            user_name = ia_name

    topics     = _extract_topics_from_interactions(interactions)
    brochure   = _pick_brochure(topics)

    # If a real static PDF exists, serve it directly
    if brochure.exists():
        pdf_bytes = brochure.read_bytes()
        fname = brochure.name
    else:
        # Generate a personalised placeholder PDF
        summary = await _summarize_interactions(interactions, user_name=user_name)
        pdf_bytes = _generate_placeholder_pdf(
            title   = f"RNSIT — Your Visit Summary for {user_name}",
            topics  = topics,
            summary = summary,
        )
        fname = "rnsit_brochure.pdf"

    if not pdf_bytes:
        raise HTTPException(status_code=500, detail="Could not generate brochure PDF.")

    return Response(
        content     = pdf_bytes,
        media_type  = "application/pdf",
        headers     = {
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Cache-Control":       "no-store",
        },
    )


@router.get("/{token}", response_class=HTMLResponse)
async def companion_landing_page(token: str):
    """
    Mobile landing page served as plain HTML — no React build required.
    Validates the token and renders the session summary with a brochure download link.
    Degrades gracefully on expired/invalid tokens.
    """
    snapshot = _lookup_token(token)

    if not snapshot:
        # Expired or invalid token — show a clear, helpful error page
        return HTMLResponse(content=_render_expired_page(), status_code=410)

    # Valid token — render full companion page; JS fetches summary async
    return HTMLResponse(content=_render_companion_page(token, snapshot))


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def _render_expired_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Session Expired — RNSIT</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
      min-height: 100vh; display: flex; align-items: center; justify-content: center;
      color: #f1f5f9; padding: 24px;
    }
    .card {
      background: rgba(255,255,255,0.07); backdrop-filter: blur(16px);
      border: 1px solid rgba(255,255,255,0.12); border-radius: 20px;
      padding: 40px 32px; max-width: 400px; width: 100%; text-align: center;
    }
    .icon { font-size: 56px; margin-bottom: 20px; }
    h1 { font-size: 22px; font-weight: 700; margin-bottom: 12px; color: #f8fafc; }
    p  { font-size: 14px; color: #94a3b8; line-height: 1.6; margin-bottom: 8px; }
    .contact {
      margin-top: 28px; padding: 16px; background: rgba(255,255,255,0.05);
      border-radius: 12px; font-size: 13px; color: #cbd5e1;
    }
    .contact a { color: #60a5fa; text-decoration: none; }
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">⏰</div>
    <h1>Session Link Expired</h1>
    <p>This QR companion link has expired or is no longer valid.</p>
    <p>Companion links are active for up to 20 minutes from your kiosk session.</p>
    <div class="contact">
      <strong>Visit us at RNSIT</strong><br>
      Dr. Vishnuvardhan Road, Channasandra<br>
      Bengaluru – 560098<br><br>
      📞 <a href="tel:+918023190000">+91-80-23190000</a><br>
      🌐 <a href="https://www.rnsit.ac.in" target="_blank">www.rnsit.ac.in</a>
    </div>
  </div>
</body>
</html>"""


def _render_companion_page(token: str, snapshot: dict) -> str:
    user_name  = snapshot.get("user_name", "Guest")
    issued_at  = snapshot.get("issued_at", "")
    backend_url = COMPANION_BASE_URL

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Your RNSIT Visit Summary</title>
  <meta name="description" content="Continue your RNSIT kiosk session on your phone — view your session summary and download campus brochures.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --primary:   #1a56db;
      --primary-light: #3b82f6;
      --accent:    #10b981;
      --bg:        #f8fafc;
      --surface:   #ffffff;
      --text:      #111827;
      --muted:     #6b7280;
      --border:    #e5e7eb;
      --radius:    16px;
    }}

    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }}

    /* ── Top hero banner ── */
    .hero {{
      background: linear-gradient(135deg, #0f172a 0%, #1a56db 50%, #1e40af 100%);
      padding: 32px 24px 48px;
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: '';
      position: absolute;
      inset: 0;
      background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='30'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }}
    .hero-logo {{ font-size: 13px; color: rgba(255,255,255,0.7); font-weight: 500; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px; }}
    .hero h1 {{ font-size: 26px; font-weight: 800; color: #fff; line-height: 1.2; margin-bottom: 8px; }}
    .hero-sub {{ font-size: 14px; color: rgba(255,255,255,0.7); }}
    .avatar {{
      width: 56px; height: 56px; border-radius: 50%;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      display: flex; align-items: center; justify-content: center;
      font-size: 24px; margin-bottom: 16px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}

    /* ── Cards ── */
    .content {{ padding: 0 16px 32px; margin-top: -24px; }}
    .card {{
      background: var(--surface);
      border-radius: var(--radius);
      padding: 20px;
      margin-bottom: 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      border: 1px solid var(--border);
    }}
    .card-label {{
      font-size: 11px; font-weight: 700; letter-spacing: 1px;
      color: var(--muted); text-transform: uppercase; margin-bottom: 10px;
    }}
    .card-body {{ font-size: 15px; line-height: 1.7; color: var(--text); }}

    /* ── Summary skeleton loader ── */
    .skeleton {{
      background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
      border-radius: 6px; height: 14px; margin-bottom: 8px;
    }}
    .skeleton:last-child {{ width: 70%; }}
    @keyframes shimmer {{
      0%   {{ background-position: 200% 0; }}
      100% {{ background-position: -200% 0; }}
    }}

    /* ── Download button ── */
    .btn-download {{
      display: flex; align-items: center; justify-content: center; gap: 10px;
      width: 100%; padding: 16px 20px;
      background: linear-gradient(135deg, var(--accent), #059669);
      color: white; font-size: 16px; font-weight: 700;
      border: none; border-radius: 14px; cursor: pointer;
      text-decoration: none;
      box-shadow: 0 4px 16px rgba(16,185,129,0.35);
      transition: transform 0.15s, box-shadow 0.15s;
    }}
    .btn-download:active {{ transform: scale(0.97); box-shadow: 0 2px 8px rgba(16,185,129,0.2); }}
    .btn-download .icon {{ font-size: 20px; }}

    /* ── Quick links ── */
    .links-grid {{
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 10px; margin-top: 4px;
    }}
    .link-item {{
      display: flex; align-items: center; gap: 8px;
      padding: 12px 14px; background: #f8fafc;
      border: 1px solid var(--border); border-radius: 10px;
      font-size: 13px; font-weight: 500; color: var(--primary);
      text-decoration: none;
      transition: background 0.15s;
    }}
    .link-item:active {{ background: #dbeafe; }}

    /* ── Expiry badge ── */
    .expiry {{
      display: flex; align-items: center; gap: 8px;
      font-size: 12px; color: var(--muted); margin-top: 8px;
    }}
    .expiry-dot {{
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(16,185,129,0.2);
    }}

    /* ── Footer ── */
    footer {{
      text-align: center; padding: 24px 16px;
      font-size: 12px; color: var(--muted);
    }}
    footer a {{ color: var(--primary-light); text-decoration: none; }}
  </style>
</head>
<body>

  <div class="hero">
    <div class="hero-logo">RNS Institute of Technology</div>
    <div class="avatar">🎓</div>
    <h1 id="hero-greeting">Hello{', ' + user_name if user_name not in ('Guest', 'Unknown', '') else ''}!</h1>
    <p class="hero-sub">Here's a recap of your kiosk session with Nova.</p>
  </div>

  <div class="content">

    <!-- Session Summary -->
    <div class="card" id="summary-card">
      <div class="card-label">📋 Your Session Summary</div>
      <div class="card-body" id="summary-body">
        <div class="skeleton"></div>
        <div class="skeleton"></div>
        <div class="skeleton"></div>
        <div class="skeleton" style="width:80%"></div>
        <div class="skeleton"></div>
      </div>
    </div>

    <!-- Download Brochure -->
    <a id="brochure-btn" class="btn-download" href="{backend_url}/companion/brochure/{token}" download>
      <span class="icon">📄</span>
      <span id="brochure-btn-text">Download Official Brochure (PDF)</span>
    </a>

    <!-- Quick Links -->
    <div class="card" style="margin-top:16px">
      <div class="card-label">🔗 Quick Links</div>
      <div class="links-grid">
        <a class="link-item" href="https://www.rnsit.ac.in" target="_blank">🌐 Website</a>
        <a class="link-item" href="https://www.rnsit.ac.in/admissions" target="_blank">📝 Admissions</a>
        <a class="link-item" href="tel:+918023190000">📞 Call Us</a>
        <a class="link-item" href="https://goo.gl/maps/rnsit" target="_blank">📍 Directions</a>
      </div>
    </div>

    <!-- Expiry indicator -->
    <div class="expiry">
      <div class="expiry-dot" id="expiry-dot"></div>
      <span id="expiry-text">This link is active · checking…</span>
    </div>

  </div>

  <footer>
    <p>RNS Institute of Technology, Bengaluru — 560098</p>
    <p style="margin-top:4px"><a href="https://www.rnsit.ac.in">www.rnsit.ac.in</a> · 📞 +91-80-23190000</p>
  </footer>

  <script>
    const BACKEND   = '{backend_url}';
    const TOKEN     = '{token}';
    const TTL_MIN   = {COMPANION_TOKEN_TTL_MINUTES};

    async function loadSummary() {{
      try {{
        const r = await fetch(BACKEND + '/companion/validate/' + TOKEN);
        if (!r.ok) {{
          if (r.status === 410) {{ showExpired(); return; }}
          throw new Error('HTTP ' + r.status);
        }}
        const data = await r.json();

        // Dynamically update greeting if user name was detected during conversation
        if (data.user_name && data.user_name !== 'Guest' && data.user_name !== 'Unknown') {{
          const heroH1 = document.getElementById('hero-greeting');
          if (heroH1) heroH1.textContent = 'Hello, ' + data.user_name + '!';
        }}

        // Render summary
        const body = document.getElementById('summary-body');
        body.innerHTML = data.summary
          ? data.summary.replace(/\\n/g, '<br>')
          : 'No summary available yet.';

        // Expiry badge
        const dot  = document.getElementById('expiry-dot');
        const text = document.getElementById('expiry-text');
        const issuedAt = new Date(data.issued_at);
        if (!isNaN(issuedAt)) {{
          const expiresAt = new Date(issuedAt.getTime() + TTL_MIN * 60 * 1000);
          const now = Date.now();
          const remaining = Math.max(0, Math.floor((expiresAt - now) / 60000));
          if (remaining > 0) {{
            text.textContent = `Link active for ${{remaining}} more minute${{remaining === 1 ? '' : 's'}}`;
          }} else {{
            showExpired();
          }}
        }} else {{
          text.textContent = `Link active for ${{TTL_MIN}} minutes from scan`;
        }}
      }} catch (e) {{
        document.getElementById('summary-body').textContent =
          'Unable to load session summary. Please try again.';
      }}
    }}

    function showExpired() {{
      const dot  = document.getElementById('expiry-dot');
      const text = document.getElementById('expiry-text');
      dot.style.background  = '#ef4444';
      dot.style.boxShadow   = '0 0 0 3px rgba(239,68,68,0.2)';
      text.textContent      = 'This link has expired';
      document.getElementById('brochure-btn').style.opacity = '0.5';
      document.getElementById('brochure-btn').style.pointerEvents = 'none';
    }}

    loadSummary();
  </script>

</body>
</html>"""
