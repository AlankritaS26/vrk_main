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
import zipfile
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

# Comprehensive catalog of all official college & department brochures
BROCHURE_CATALOG: list[dict[str, str]] = [
    {
        "id": "general",
        "filename": "general.pdf",
        "title": "General Campus Prospectus",
        "subtitle": "Overview, autonomous curriculum, infrastructure & student life",
        "category": "Overview",
        "icon": "🏫",
        "tag": "Campus Guide",
    },
    {
        "id": "admissions",
        "filename": "admissions.pdf",
        "title": "Admissions & Eligibility Guide",
        "subtitle": "KCET, COMEDK, Management quotas, eligibility & seat matrices",
        "category": "Admissions & Fees",
        "icon": "🎓",
        "tag": "Admissions",
    },
    {
        "id": "fees",
        "filename": "fees.pdf",
        "title": "Fee Structure & Scholarships",
        "subtitle": "Tuition fees, payment schedules, scholarships & concessions",
        "category": "Admissions & Fees",
        "icon": "💳",
        "tag": "Finance",
    },
    {
        "id": "placements",
        "filename": "placements.pdf",
        "title": "Training & Placements Report",
        "subtitle": "50 LPA highest package, 92%+ offers & top tier-1 recruiters",
        "category": "Campus Life & Careers",
        "icon": "🏆",
        "tag": "Placements",
    },
    {
        "id": "cse",
        "filename": "cse.pdf",
        "title": "Computer Science & Engg (CSE)",
        "subtitle": "AI, Cloud Computing, Full-Stack, NBA accredited curriculum",
        "category": "Engineering",
        "icon": "💻",
        "tag": "Department",
    },
    {
        "id": "ise",
        "filename": "ise.pdf",
        "title": "Information Science & Engg (ISE)",
        "subtitle": "Data engineering, software architecture, cyber security",
        "category": "Engineering",
        "icon": "💡",
        "tag": "Department",
    },
    {
        "id": "ece",
        "filename": "ece.pdf",
        "title": "Electronics & Communication (ECE)",
        "subtitle": "VLSI design, embedded systems, robotics, IoT & wireless tech",
        "category": "Engineering",
        "icon": "📡",
        "tag": "Department",
    },
    {
        "id": "eee",
        "filename": "eee.pdf",
        "title": "Electrical & Electronics (EEE)",
        "subtitle": "Electric vehicles, renewable power, smart grids & automation",
        "category": "Engineering",
        "icon": "⚡",
        "tag": "Department",
    },
    {
        "id": "me",
        "filename": "me.pdf",
        "title": "Mechanical Engineering (ME)",
        "subtitle": "Industrial robotics, CAD/CAM/CAE, thermal engineering & labs",
        "category": "Engineering",
        "icon": "⚙️",
        "tag": "Department",
    },
    {
        "id": "civil",
        "filename": "civil.pdf",
        "title": "Civil Engineering",
        "subtitle": "Structural engineering, GIS/surveying, sustainable building",
        "category": "Engineering",
        "icon": "🏗️",
        "tag": "Department",
    },
    {
        "id": "mba",
        "filename": "mba.pdf",
        "title": "Management Studies (MBA)",
        "subtitle": "Finance, Marketing, HR, Business Analytics & corporate immersion",
        "category": "Postgraduate",
        "icon": "📊",
        "tag": "Management",
    },
    {
        "id": "hostel",
        "filename": "hostel.pdf",
        "title": "Hostels & Campus Accommodation",
        "subtitle": "Boys & girls hostel blocks, dining, 24/7 security & amenities",
        "category": "Campus Life & Careers",
        "icon": "🏠",
        "tag": "Facilities",
    },
]

# Topic keyword → brochure filename mapping
BROCHURE_TOPIC_MAP: dict[str, str] = {
    "cse":          "cse.pdf",
    "computer":     "cse.pdf",
    "cs":           "cse.pdf",
    "coding":       "cse.pdf",
    "software":     "cse.pdf",
    "ece":          "ece.pdf",
    "electronics":  "ece.pdf",
    "communication":"ece.pdf",
    "vlsi":         "ece.pdf",
    "ise":          "ise.pdf",
    "information":  "ise.pdf",
    "me":           "me.pdf",
    "mechanical":   "me.pdf",
    "civil":        "civil.pdf",
    "construction": "civil.pdf",
    "eee":          "eee.pdf",
    "electrical":   "eee.pdf",
    "mba":          "mba.pdf",
    "management":   "mba.pdf",
    "business":     "mba.pdf",
    "admission":    "admissions.pdf",
    "eligibility":  "admissions.pdf",
    "seat":         "admissions.pdf",
    "cutoff":       "admissions.pdf",
    "quota":        "admissions.pdf",
    "fee":          "fees.pdf",
    "fees":         "fees.pdf",
    "scholarship":  "fees.pdf",
    "cost":         "fees.pdf",
    "hostel":       "hostel.pdf",
    "room":         "hostel.pdf",
    "mess":         "hostel.pdf",
    "food":         "hostel.pdf",
    "stay":         "hostel.pdf",
    "accommodation":"hostel.pdf",
    "placement":    "placements.pdf",
    "package":      "placements.pdf",
    "salary":       "placements.pdf",
    "recruiter":    "placements.pdf",
    "job":          "placements.pdf",
    "career":       "placements.pdf",
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


def _get_catalog_item(brochure_id: str) -> dict | None:
    if not brochure_id:
        return None
    bid = brochure_id.lower().strip().replace(".pdf", "")
    for item in BROCHURE_CATALOG:
        if item["id"] == bid or item["filename"].replace(".pdf", "") == bid:
            return item
    return None


def _pick_brochure_id(topics: str) -> str:
    """Return the catalog ID of the best-matching brochure for the given topic string."""
    topics_lower = (topics or "").lower()
    for keyword, filename in BROCHURE_TOPIC_MAP.items():
        if keyword in topics_lower:
            return filename.replace(".pdf", "")
    return "general"


def _pick_brochure(topics: str) -> Path:
    """Return the Path of the best-matching brochure PDF for the given topic string."""
    rec_id = _pick_brochure_id(topics)
    item = _get_catalog_item(rec_id)
    if item:
        candidate = BROCHURES_DIR / item["filename"]
        if candidate.exists():
            return candidate
    general = BROCHURES_DIR / "general.pdf"
    if general.exists():
        return general
    return BROCHURES_DIR / "general.pdf"


def _build_brochures_data(token: str, rec_id: str = "general") -> list[dict]:
    """Build serializable metadata for all catalog brochures."""
    res = []
    for item in BROCHURE_CATALOG:
        p = BROCHURES_DIR / item["filename"]
        exists = p.exists()
        is_rec = (item["id"] == rec_id)
        res.append({
            "id":             item["id"],
            "title":          item["title"],
            "subtitle":       item["subtitle"],
            "category":       item["category"],
            "icon":           item["icon"],
            "tag":            item["tag"],
            "filename":       item["filename"],
            "available":      exists,
            "is_recommended": is_rec,
            "download_url":   f"{COMPANION_BASE_URL}/companion/brochure/{token}/{item['id']}",
        })
    return res


def _render_brochure_cards_html(token: str, recommended_id: str = "general") -> str:
    """Pre-render static HTML cards for all brochures so mobile visitors can select and download immediately."""
    cards = []
    for item in BROCHURE_CATALOG:
        is_rec = (item["id"] == recommended_id)
        rec_badge = '<span class="badge badge-rec">⭐ Recommended</span>' if is_rec else ''
        cat_badge = f'<span class="badge badge-cat">{item["category"]}</span>'
        tag_badge = f'<span class="badge badge-tag">{item["tag"]}</span>'
        download_url = f"{COMPANION_BASE_URL}/companion/brochure/{token}/{item['id']}"
        search_terms = f"{item['title']} {item['subtitle']} {item['category']} {item['id']} {item['tag']}".lower()

        cards.append(f"""
        <div class="brochure-card selectable{' is-recommended' if is_rec else ''}" data-id="{item['id']}" data-category="{item['category']}" data-search="{search_terms}" onclick="toggleBrochureSelection('{item['id']}', event)">
          <div class="select-indicator">
            <div class="checkbox-circle" id="check-{item['id']}">
              <span class="checkmark">✓</span>
            </div>
          </div>
          <div class="card-main">
            <div class="brochure-icon">{item['icon']}</div>
            <div class="brochure-meta">
              <div class="badges-row">{cat_badge}{tag_badge}{rec_badge}</div>
              <h3 class="brochure-title">{item['title']}</h3>
              <p class="brochure-sub">{item['subtitle']}</p>
            </div>
          </div>
          <div class="card-action">
            <a class="btn-single-dl" href="{download_url}" download="RNSIT_{item['filename']}" title="Download this PDF directly" onclick="event.stopPropagation();">
              <span class="dl-icon">⬇️</span>
              <span class="dl-text">PDF</span>
            </a>
          </div>
        </div>""")
    return "\\n".join(cards)


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

    # Pick recommended brochure for session
    rec_id = _pick_brochure_id(topics)
    rec_item = _get_catalog_item(rec_id) or _get_catalog_item("general")
    brochures_list = _build_brochures_data(token, rec_id)

    rec_download_url = f"{COMPANION_BASE_URL}/companion/brochure/{token}/{rec_id}"
    zip_download_url = f"{COMPANION_BASE_URL}/companion/brochure/{token}/all"

    return {
        "valid":                True,
        "user_name":            user_name,
        "session_id":           session_id,
        "summary":              summary,
        "topics":               topics[:200],
        "recommended_brochure": {
            "id":           rec_item["id"],
            "title":        rec_item["title"],
            "subtitle":     rec_item["subtitle"],
            "category":     rec_item["category"],
            "icon":         rec_item["icon"],
            "download_url": rec_download_url,
        },
        "brochures":            brochures_list,
        "zip_download_url":     zip_download_url,
        "brochure_url":         rec_download_url,
        "brochure_available":   True,
        "issued_at":            snapshot.get("issued_at", ""),
        "expires_in_minutes":   COMPANION_TOKEN_TTL_MINUTES,
    }


@router.get("/brochure/{token}")
@router.get("/brochure/{token}/{brochure_id}")
async def get_companion_brochure(
    token: str,
    brochure_id: str | None = None,
    doc: str | None = None,
    docs: str | None = None,
    ids: str | None = None,
):
    """
    Serve official brochure PDF(s) for this session:
      - Multi-selection: ?docs=cse,hostel bundles ONLY user-selected brochures into a custom ZIP (or single PDF).
      - 'all' or 'zip': bundles all available official campus brochures into a ZIP file.
      - brochure_id / doc specified: streams the requested brochure PDF.
      - omitted: streams the best-matched brochure based on session topics.
    """
    snapshot = _lookup_token(token)
    if not snapshot:
        raise HTTPException(status_code=410, detail="Token expired or invalid.")

    # 1. Multi-selection download: stream ONLY the user-selected brochures
    raw_list = docs or ids or ""
    if not raw_list and brochure_id and ("," in brochure_id or brochure_id.lower() == "selected"):
        raw_list = doc if brochure_id.lower() == "selected" else brochure_id
    if not raw_list and doc and "," in doc:
        raw_list = doc

    if raw_list:
        requested = [x.strip().lower() for x in raw_list.split(",") if x.strip()]
        matched_items = []
        for rid in requested:
            item = _get_catalog_item(rid)
            if item:
                matched_items.append(item)
            else:
                p = BROCHURES_DIR / f"{rid}.pdf"
                if p.exists():
                    matched_items.append({"id": rid, "filename": f"{rid}.pdf", "title": rid.upper()})

        if not matched_items:
            raise HTTPException(status_code=404, detail="No matching brochures found for selected items.")

        # If exactly 1 selected, stream that 1 PDF directly
        if len(matched_items) == 1:
            it = matched_items[0]
            cand = BROCHURES_DIR / it["filename"]
            if cand.exists():
                return Response(
                    content=cand.read_bytes(),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f'attachment; filename="RNSIT_{it["filename"]}"',
                        "Cache-Control": "no-store",
                    },
                )

        # If multiple selected, stream a tailored ZIP bundle of ONLY the chosen brochures
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for it in matched_items:
                p = BROCHURES_DIR / it["filename"]
                if p.exists():
                    zf.write(p, arcname=f"RNSIT_{it['filename']}")
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="RNSIT_Selected_Brochures.zip"',
                "Cache-Control": "no-store",
            },
        )

    target_id = (brochure_id or doc or "").strip().lower()

    # 2. Download all brochures as a ZIP archive
    if target_id in ("all", "zip", "all.zip", "bundle"):
        buf = io.BytesIO()
        added_count = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in BROCHURE_CATALOG:
                p = BROCHURES_DIR / item["filename"]
                if p.exists():
                    zf.write(p, arcname=f"RNSIT_{item['filename']}")
                    added_count += 1
        if added_count == 0:
            raise HTTPException(status_code=404, detail="No brochure files found to package.")
        buf.seek(0)
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="RNSIT_Campus_Brochures.zip"',
                "Cache-Control": "no-store",
            },
        )

    # 2. Specific brochure requested by ID or filename
    if target_id and target_id not in ("default", "recommended", "auto"):
        item = _get_catalog_item(target_id)
        candidate = None
        fname = None
        if item:
            candidate = BROCHURES_DIR / item["filename"]
            fname = f"RNSIT_{item['filename']}"
        else:
            safe_fname = Path(target_id).name
            if not safe_fname.endswith(".pdf"):
                safe_fname += ".pdf"
            candidate = BROCHURES_DIR / safe_fname
            fname = f"RNSIT_{safe_fname}"

        if candidate and candidate.exists():
            return Response(
                content=candidate.read_bytes(),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}"',
                    "Cache-Control": "no-store",
                },
            )

    # 3. Topic-based match for session
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
        fname = f"RNSIT_{brochure.name}"
    else:
        # Generate a personalised placeholder PDF
        summary = await _summarize_interactions(interactions, user_name=user_name)
        pdf_bytes = _generate_placeholder_pdf(
            title   = f"RNSIT — Your Visit Summary for {user_name}",
            topics  = topics,
            summary = summary,
        )
        fname = "RNSIT_brochure.pdf"

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
    user_name = snapshot.get("user_name", "Guest")
    backend_url = COMPANION_BASE_URL
    brochure_cards_html = _render_brochure_cards_html(token, "general")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>RNSIT Campus Companion — Choose & Download Brochures</title>
  <meta name="description" content="Select and download the specific official RNSIT department and campus PDF brochures you want directly on your phone.">
  <meta name="theme-color" content="#0b192c">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --primary: #1e3a8a;
      --primary-light: #3b82f6;
      --accent: #10b981;
      --accent-dark: #059669;
      --gold: #d97706;
      --gold-bg: #fffbeb;
      --bg: #f8fafc;
      --surface: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --radius: 16px;
      --shadow-sm: 0 2px 6px rgba(15, 23, 42, 0.05);
      --shadow-md: 0 4px 16px rgba(15, 23, 42, 0.08);
    }}

    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      padding-bottom: 96px; /* Space for floating selection bar */
    }}

    /* ── Hero Banner ── */
    .hero {{
      background: linear-gradient(135deg, #0b192c 0%, #1e3a8a 55%, #1d4ed8 100%);
      padding: 30px 20px 42px;
      color: #fff;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 85% 20%, rgba(217, 119, 6, 0.18), transparent 45%);
      pointer-events: none;
    }}
    .hero-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }}
    .hero-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(255, 255, 255, 0.12);
      backdrop-filter: blur(8px);
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.5px;
      color: #93c5fd;
      border: 1px solid rgba(255, 255, 255, 0.16);
    }}
    .live-pulse {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.35);
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ transform: scale(1); opacity: 1; }}
      50% {{ transform: scale(1.3); opacity: 0.7; }}
    }}
    .avatar {{
      width: 44px;
      height: 44px;
      border-radius: 50%;
      background: linear-gradient(135deg, #f59e0b, #d97706);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }}
    .hero h1 {{
      font-size: 23px;
      font-weight: 800;
      line-height: 1.25;
      margin-bottom: 6px;
    }}
    .hero-sub {{
      font-size: 13px;
      color: rgba(255, 255, 255, 0.85);
      line-height: 1.5;
    }}

    /* ── Main Container ── */
    .container {{
      max-width: 640px;
      margin: -24px auto 0;
      padding: 0 16px;
    }}

    /* ── Cards ── */
    .card {{
      background: var(--surface);
      border-radius: var(--radius);
      padding: 18px 20px;
      margin-bottom: 16px;
      box-shadow: var(--shadow-sm);
      border: 1px solid var(--border);
    }}
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }}
    .card-title {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.8px;
      color: var(--muted);
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .card-body {{
      font-size: 14px;
      line-height: 1.65;
      color: var(--text);
    }}
    .btn-copy {{
      background: #f1f5f9;
      border: none;
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 600;
      color: var(--primary);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: background 0.15s;
    }}
    .btn-copy:active {{ background: #e2e8f0; }}

    /* ── Skeleton ── */
    .skeleton {{
      background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%);
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
      border-radius: 6px;
      height: 14px;
      margin-bottom: 8px;
    }}
    .skeleton:last-child {{ width: 65%; }}
    @keyframes shimmer {{
      0%   {{ background-position: 200% 0; }}
      100% {{ background-position: -200% 0; }}
    }}

    /* ── Featured Recommended Brochure Card ── */
    .card-featured {{
      background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
      border: 1.5px solid #86efac;
      border-radius: var(--radius);
      padding: 16px 18px;
      margin-bottom: 20px;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.12);
    }}
    .featured-label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #dcfce7;
      color: #15803d;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 999px;
      margin-bottom: 10px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }}
    .featured-content {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .featured-icon {{
      font-size: 30px;
      width: 48px;
      height: 48px;
      background: #ffffff;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      flex-shrink: 0;
    }}
    .featured-details h2 {{
      font-size: 15px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 2px;
    }}
    .featured-details p {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .btn-featured-download {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      width: 100%;
      padding: 11px 16px;
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
      color: #fff;
      font-size: 14px;
      font-weight: 700;
      border: none;
      border-radius: 12px;
      text-decoration: none;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
      cursor: pointer;
      transition: transform 0.12s, box-shadow 0.12s;
    }}
    .btn-featured-download:active {{
      transform: scale(0.98);
      box-shadow: 0 2px 6px rgba(16, 185, 129, 0.2);
    }}

    /* ── Section Title ── */
    .section-header {{
      margin: 24px 0 10px;
    }}
    .section-title {{
      font-size: 17px;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .section-title-tag {{
      font-size: 11px;
      font-weight: 700;
      color: var(--primary);
      background: #eff6ff;
      padding: 3px 10px;
      border-radius: 999px;
    }}
    .section-sub {{
      font-size: 12.5px;
      color: var(--muted);
      margin-top: 4px;
    }}

    /* ── Selection Action Controls (Select Recommended / Clear) ── */
    .selection-actions-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
      padding: 8px 12px;
      background: #f1f5f9;
      border-radius: 12px;
      font-size: 12px;
    }}
    .sel-count-text {{
      font-weight: 700;
      color: var(--text);
    }}
    .sel-buttons-row {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .btn-sel-action {{
      border: 1px solid var(--border);
      background: #ffffff;
      padding: 4px 10px;
      border-radius: 8px;
      font-size: 11px;
      font-weight: 600;
      color: var(--primary);
      cursor: pointer;
      transition: background 0.15s;
    }}
    .btn-sel-action:active {{
      background: #e2e8f0;
    }}

    /* ── Search & Filter Controls ── */
    .search-box {{
      position: relative;
      margin-bottom: 12px;
    }}
    .search-input {{
      width: 100%;
      padding: 11px 14px 11px 38px;
      font-size: 13px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
      outline: none;
      transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .search-input:focus {{
      border-color: var(--primary-light);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
    }}
    .search-icon {{
      position: absolute;
      left: 13px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 15px;
      opacity: 0.6;
      pointer-events: none;
    }}

    .filter-pills {{
      display: flex;
      gap: 6px;
      overflow-x: auto;
      padding-bottom: 6px;
      margin-bottom: 14px;
      scrollbar-width: none;
    }}
    .filter-pills::-webkit-scrollbar {{ display: none; }}
    .pill {{
      padding: 6px 12px;
      font-size: 11.5px;
      font-weight: 600;
      border-radius: 999px;
      background: #ffffff;
      color: var(--muted);
      border: 1px solid var(--border);
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s;
      flex-shrink: 0;
    }}
    .pill.active {{
      background: var(--primary);
      color: #ffffff;
      border-color: var(--primary);
      box-shadow: 0 2px 8px rgba(30, 58, 138, 0.25);
    }}

    /* ── Selectable Brochure Cards List ── */
    .brochures-grid {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .brochure-card {{
      background: var(--surface);
      border: 1.5px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: var(--shadow-sm);
      cursor: pointer;
      user-select: none;
      -webkit-tap-highlight-color: transparent;
      transition: border-color 0.15s, background 0.15s, box-shadow 0.15s, transform 0.1s;
    }}
    .brochure-card:active {{
      transform: scale(0.99);
    }}
    .brochure-card.is-selected {{
      border-color: var(--accent);
      background: #f0fdf4 !important;
      box-shadow: 0 3px 12px rgba(16, 185, 129, 0.15);
    }}
    .brochure-card.hidden {{
      display: none !important;
    }}

    /* ── Checkbox on Card ── */
    .select-indicator {{
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }}
    .checkbox-circle {{
      width: 24px;
      height: 24px;
      border-radius: 50%;
      border: 2px solid #cbd5e1;
      background: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;
    }}
    .checkbox-circle .checkmark {{
      font-size: 13px;
      color: #ffffff;
      font-weight: 800;
      opacity: 0;
      transform: scale(0.5);
      transition: all 0.15s ease;
    }}
    .brochure-card.is-selected .checkbox-circle {{
      border-color: var(--accent);
      background: var(--accent);
    }}
    .brochure-card.is-selected .checkbox-circle .checkmark {{
      opacity: 1;
      transform: scale(1);
    }}

    .card-main {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
      flex: 1;
    }}
    .brochure-icon {{
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: #f1f5f9;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 20px;
      flex-shrink: 0;
    }}
    .brochure-card.is-selected .brochure-icon {{
      background: #dcfce7;
    }}
    .brochure-meta {{
      min-width: 0;
    }}
    .badges-row {{
      display: flex;
      align-items: center;
      gap: 4px;
      margin-bottom: 2px;
      flex-wrap: wrap;
    }}
    .badge {{
      font-size: 9px;
      font-weight: 700;
      padding: 1px 6px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}
    .badge-cat {{
      background: #eff6ff;
      color: #1d4ed8;
    }}
    .badge-tag {{
      background: #f8fafc;
      color: #64748b;
      border: 1px solid #e2e8f0;
    }}
    .badge-rec {{
      background: #fef08a;
      color: #854d0e;
    }}
    .brochure-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: 1px;
    }}
    .brochure-sub {{
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 280px;
    }}

    /* ── Single Direct Download Icon Button ── */
    .card-action {{
      flex-shrink: 0;
    }}
    .btn-single-dl {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 3px;
      background: #f8fafc;
      color: var(--primary);
      border: 1px solid #cbd5e1;
      padding: 7px 10px;
      border-radius: 9px;
      font-size: 11px;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .btn-single-dl:active {{
      background: var(--primary);
      color: #ffffff;
      transform: scale(0.95);
    }}
    .btn-single-dl .dl-icon {{
      font-size: 12px;
    }}

    /* ── Floating Sticky Action Bar for Selected Brochures ── */
    .floating-bar {{
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(14px);
      border-top: 1px solid var(--border);
      box-shadow: 0 -4px 20px rgba(15, 23, 42, 0.12);
      padding: 12px 16px;
      z-index: 999;
    }}
    .floating-inner {{
      max-width: 640px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .bar-info {{
      min-width: 0;
    }}
    .bar-count {{
      font-size: 13.5px;
      font-weight: 800;
      color: var(--text);
    }}
    .bar-hint {{
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .btn-bar-download {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 18px;
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
      color: #ffffff;
      font-size: 13.5px;
      font-weight: 700;
      border-radius: 12px;
      text-decoration: none;
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
      white-space: nowrap;
      transition: all 0.15s ease;
    }}
    .btn-bar-download:active {{
      transform: scale(0.97);
    }}
    .btn-bar-download.disabled {{
      background: #cbd5e1;
      color: #64748b;
      box-shadow: none;
      cursor: not-allowed;
      pointer-events: none;
      opacity: 0.65;
    }}

    /* ── Quick Links ── */
    .links-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 4px;
    }}
    .link-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 11px 12px;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      font-size: 12.5px;
      font-weight: 600;
      color: var(--primary);
      text-decoration: none;
      transition: background 0.15s;
    }}
    .link-item:active {{ background: #eff6ff; }}

    /* ── Expiry Badge ── */
    .expiry {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11.5px;
      color: var(--muted);
      margin-top: 12px;
      padding: 0 4px;
    }}
    .expiry-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(16,185,129,0.2);
    }}

    /* ── Toast Notification ── */
    .toast {{
      position: fixed;
      bottom: 84px;
      left: 50%;
      transform: translateX(-50%) translateY(100px);
      background: #0f172a;
      color: #ffffff;
      padding: 10px 18px;
      border-radius: 999px;
      font-size: 12.5px;
      font-weight: 600;
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
      display: flex;
      align-items: center;
      gap: 8px;
      transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.25s;
      opacity: 0;
      z-index: 1000;
      pointer-events: none;
    }}
    .toast.show {{
      transform: translateX(-50%) translateY(0);
      opacity: 1;
    }}

    /* ── Footer ── */
    footer {{
      text-align: center;
      padding: 24px 16px 10px;
      font-size: 11.5px;
      color: var(--muted);
      line-height: 1.6;
    }}
    footer a {{ color: var(--primary-light); text-decoration: none; font-weight: 500; }}
  </style>
</head>
<body>

  <!-- Hero Header -->
  <div class="hero">
    <div class="hero-top">
      <div class="hero-badge">
        <span class="live-pulse"></span>
        <span>RNSIT Digital Kiosk Companion</span>
      </div>
      <div class="avatar">🎓</div>
    </div>
    <h1 id="hero-greeting">Hello{', ' + user_name if user_name not in ('Guest', 'Unknown', '') else ''}!</h1>
    <p class="hero-sub">Here is your visit summary with Nova. Select only the brochures you want and download them directly to your phone.</p>
  </div>

  <div class="container">

    <!-- Session Summary Card -->
    <div class="card" id="summary-card">
      <div class="card-header">
        <div class="card-title">📋 Your Session Summary</div>
        <button class="btn-copy" id="copy-btn" onclick="copySummary()">Copy</button>
      </div>
      <div class="card-body" id="summary-body">
        <div class="skeleton"></div>
        <div class="skeleton"></div>
        <div class="skeleton" style="width:85%"></div>
      </div>
    </div>

    <!-- Featured Recommendation Card (Matches visitor questions) -->
    <div class="card-featured" id="featured-card">
      <div class="featured-label">⭐ Recommended For Your Visit</div>
      <div class="featured-content">
        <div class="featured-icon" id="featured-icon">🏫</div>
        <div class="featured-details">
          <h2 id="featured-title">General Campus Prospectus</h2>
          <p id="featured-sub">Overview, autonomous curriculum, infrastructure & student life</p>
        </div>
      </div>
      <a id="featured-brochure-btn" class="btn-featured-download" href="{backend_url}/companion/brochure/{token}/general" download="RNSIT_general.pdf">
        <span>📥</span>
        <span id="featured-btn-text">Download Recommended Brochure (PDF)</span>
      </a>
    </div>

    <!-- Section Header -->
    <div class="section-header">
      <div class="section-title">
        <span>Select Which Brochures to Download</span>
        <span class="section-title-tag">Choose &amp; Download</span>
      </div>
      <p class="section-sub">Tap any card to select it, or tap the PDF button on the right to download individually.</p>
    </div>

    <!-- Selection Bar with Recommended / Clear Actions -->
    <div class="selection-actions-bar">
      <div class="sel-count-text" id="selection-summary-text">0 selected</div>
      <div class="sel-buttons-row">
        <button type="button" class="btn-sel-action" onclick="selectRecommended()">⭐ Select Recommended</button>
        <button type="button" class="btn-sel-action" onclick="clearSelection()">Clear</button>
      </div>
    </div>

    <!-- Search Input -->
    <div class="search-box">
      <span class="search-icon">🔍</span>
      <input type="text" class="search-input" id="brochure-search" placeholder="Search brochures (CSE, hostel, fees, civil...)" oninput="filterBrochures()" />
    </div>

    <!-- Category Filter Tabs -->
    <div class="filter-pills" id="filter-pills">
      <button class="pill active" data-cat="all" onclick="setCategory('all', this)">All</button>
      <button class="pill" data-cat="rec" onclick="setCategory('rec', this)">⭐ Recommended</button>
      <button class="pill" data-cat="Engineering" onclick="setCategory('Engineering', this)">💻 Engineering</button>
      <button class="pill" data-cat="Admissions & Fees" onclick="setCategory('Admissions & Fees', this)">🎓 Admissions & Fees</button>
      <button class="pill" data-cat="Campus Life & Careers" onclick="setCategory('Campus Life & Careers', this)">🏆 Life & Careers</button>
    </div>

    <!-- Grid of Selectable Brochure Cards -->
    <div class="brochures-grid" id="brochures-grid">
      {brochure_cards_html}
    </div>

    <!-- Quick Links -->
    <div class="card" style="margin-top:20px">
      <div class="card-title" style="margin-bottom:12px">🔗 Campus Quick Links</div>
      <div class="links-grid">
        <a class="link-item" href="https://www.rnsit.ac.in" target="_blank">🌐 Official Website</a>
        <a class="link-item" href="https://www.rnsit.ac.in/admissions" target="_blank">📝 Admissions Portal</a>
        <a class="link-item" href="tel:+918023190000">📞 Call Reception</a>
        <a class="link-item" href="https://goo.gl/maps/rnsit" target="_blank">📍 Campus Maps</a>
      </div>
    </div>

    <!-- Expiry indicator -->
    <div class="expiry">
      <div class="expiry-dot" id="expiry-dot"></div>
      <span id="expiry-text">This link is active · checking…</span>
    </div>

  </div>

  <!-- Floating Sticky Action Bar for Selected Downloads -->
  <div class="floating-bar" id="floating-bar">
    <div class="floating-inner">
      <div class="bar-info">
        <div class="bar-count" id="bar-count">0 Selected</div>
        <div class="bar-hint" id="bar-hint">Tap brochures above to choose</div>
      </div>
      <a class="btn-bar-download disabled" id="btn-bar-download" href="javascript:void(0)" onclick="downloadSelected(event)">
        <span>📥</span>
        <span id="btn-selected-text">Download Selected</span>
      </a>
    </div>
  </div>

  <footer>
    <p><strong>RNS Institute of Technology, Bengaluru — 560098</strong></p>
    <p style="margin-top:4px"><a href="https://www.rnsit.ac.in" target="_blank">www.rnsit.ac.in</a> · 📞 +91-80-23190000 · ✉️ principal@rnsit.ac.in</p>
  </footer>

  <div class="toast" id="toast">
    <span>📥</span>
    <span id="toast-text">Starting download...</span>
  </div>

  <script>
    const BACKEND = '{backend_url}';
    const TOKEN = '{token}';
    const TTL_MIN = {COMPANION_TOKEN_TTL_MINUTES};
    let currentCategory = 'all';
    let recommendedId = 'general';
    const selectedBrochures = new Set();

    async function loadCompanionData() {{
      try {{
        const r = await fetch(BACKEND + '/companion/validate/' + TOKEN);
        if (!r.ok) {{
          if (r.status === 410) {{ showExpired(); return; }}
          throw new Error('HTTP ' + r.status);
        }}
        const data = await r.json();

        // Update Greeting
        if (data.user_name && data.user_name !== 'Guest' && data.user_name !== 'Unknown') {{
          const h1 = document.getElementById('hero-greeting');
          if (h1) h1.textContent = 'Hello, ' + data.user_name + '!';
        }}

        // Update Session Summary
        const body = document.getElementById('summary-body');
        if (body) {{
          body.innerHTML = data.summary
            ? data.summary.replace(/\\n/g, '<br>')
            : 'No summary available yet.';
        }}

        // Update Recommended Brochure
        if (data.recommended_brochure) {{
          const rb = data.recommended_brochure;
          recommendedId = rb.id;
          const fTitle = document.getElementById('featured-title');
          const fSub = document.getElementById('featured-sub');
          const fIcon = document.getElementById('featured-icon');
          const fBtn = document.getElementById('featured-brochure-btn');
          const fBtnText = document.getElementById('featured-btn-text');

          if (fTitle) fTitle.textContent = rb.title;
          if (fSub) fSub.textContent = rb.subtitle;
          if (fIcon) fIcon.textContent = rb.icon || '📄';
          if (fBtn) {{
            fBtn.href = rb.download_url;
            fBtn.setAttribute('download', 'RNSIT_' + rb.id + '.pdf');
          }}
          if (fBtnText) fBtnText.textContent = 'Download ' + rb.title + ' (PDF)';

          // Mark card in list
          document.querySelectorAll('.brochure-card').forEach(card => {{
            if (card.dataset.id === rb.id) {{
              card.classList.add('is-recommended');
              if (!card.querySelector('.badge-rec')) {{
                const badge = document.createElement('span');
                badge.className = 'badge badge-rec';
                badge.textContent = '⭐ Recommended';
                const row = card.querySelector('.badges-row');
                if (row) row.appendChild(badge);
              }}
            }}
          }});

          // Automatically select the recommended brochure by default so user has 1 ready
          if (selectedBrochures.size === 0) {{
            selectBrochure(rb.id, true);
          }}
        }}

        updateExpiry(data.issued_at);
      }} catch (e) {{
        const body = document.getElementById('summary-body');
        if (body) {{
          body.textContent = 'Unable to refresh session summary. All brochure download links below remain fully active.';
        }}
      }}
    }}

    function toggleBrochureSelection(id, event) {{
      if (selectedBrochures.has(id)) {{
        selectBrochure(id, false);
      }} else {{
        selectBrochure(id, true);
      }}
    }}

    function selectBrochure(id, shouldSelect) {{
      const card = document.querySelector(`.brochure-card[data-id="${{id}}"]`);
      if (shouldSelect) {{
        selectedBrochures.add(id);
        if (card) card.classList.add('is-selected');
      }} else {{
        selectedBrochures.delete(id);
        if (card) card.classList.remove('is-selected');
      }}
      updateSelectionUI();
    }}

    function selectRecommended() {{
      if (recommendedId) {{
        selectBrochure(recommendedId, true);
        showToast('Selected recommended brochure!');
      }}
    }}

    function clearSelection() {{
      selectedBrochures.clear();
      document.querySelectorAll('.brochure-card').forEach(c => c.classList.remove('is-selected'));
      updateSelectionUI();
      showToast('Cleared brochure selections');
    }}

    function updateSelectionUI() {{
      const count = selectedBrochures.size;
      const barCount = document.getElementById('bar-count');
      const barHint = document.getElementById('bar-hint');
      const btn = document.getElementById('btn-bar-download');
      const btnText = document.getElementById('btn-selected-text');
      const summaryText = document.getElementById('selection-summary-text');

      if (summaryText) {{
        summaryText.textContent = count === 0
          ? '0 selected (tap cards to choose)'
          : `${{count}} brochure${{count === 1 ? '' : 's'}} selected`;
      }}

      if (barCount) {{
        barCount.textContent = count === 0 ? '0 Selected' : `${{count}} Selected`;
      }}

      if (barHint) {{
        barHint.textContent = count === 0
          ? 'Tap any brochure above to select'
          : count === 1
            ? 'Ready for direct PDF download'
            : 'Will download as a tailored ZIP bundle';
      }}

      if (btn && btnText) {{
        if (count === 0) {{
          btn.classList.add('disabled');
          btnText.textContent = 'Select Brochures';
          btn.removeAttribute('href');
        }} else if (count === 1) {{
          btn.classList.remove('disabled');
          const singleId = Array.from(selectedBrochures)[0];
          btnText.textContent = 'Download Selected (1 PDF)';
          btn.href = `${{BACKEND}}/companion/brochure/${{TOKEN}}/${{singleId}}`;
          btn.setAttribute('download', `RNSIT_${{singleId}}.pdf`);
        }} else {{
          btn.classList.remove('disabled');
          const idList = Array.from(selectedBrochures).join(',');
          btnText.textContent = `Download Selected (${{count}} PDFs)`;
          btn.href = `${{BACKEND}}/companion/brochure/${{TOKEN}}/selected?docs=${{idList}}`;
          btn.setAttribute('download', 'RNSIT_Selected_Brochures.zip');
        }}
      }}
    }}

    function downloadSelected(e) {{
      if (selectedBrochures.size === 0) {{
        if (e) e.preventDefault();
        showToast('Please tap on at least one brochure to select it!');
        return;
      }}
      const count = selectedBrochures.size;
      showToast(count === 1 ? 'Downloading selected PDF...' : `Downloading ${{count}} selected brochures...`);
    }}

    function updateExpiry(issuedAtStr) {{
      const dot = document.getElementById('expiry-dot');
      const text = document.getElementById('expiry-text');
      const issuedAt = new Date(issuedAtStr);
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
    }}

    function filterBrochures() {{
      const q = (document.getElementById('brochure-search').value || '').toLowerCase().trim();
      const cards = document.querySelectorAll('.brochure-card');

      cards.forEach(card => {{
        const text = (card.dataset.search || '').toLowerCase();
        const cat = card.dataset.category || '';
        const isRec = card.classList.contains('is-recommended');

        const matchesQuery = !q || text.includes(q);
        let matchesCat = true;

        if (currentCategory === 'rec') {{
          matchesCat = isRec;
        }} else if (currentCategory !== 'all') {{
          matchesCat = (cat === currentCategory);
        }}

        if (matchesQuery && matchesCat) {{
          card.classList.remove('hidden');
        }} else {{
          card.classList.add('hidden');
        }}
      }});
    }}

    function setCategory(cat, el) {{
      currentCategory = cat;
      document.querySelectorAll('.filter-pills .pill').forEach(p => p.classList.remove('active'));
      if (el) el.classList.add('active');
      filterBrochures();
    }}

    function copySummary() {{
      const text = document.getElementById('summary-body').innerText;
      if (navigator.clipboard) {{
        navigator.clipboard.writeText(text).then(() => showToast('Summary copied to clipboard!'));
      }} else {{
        showToast('Summary ready to copy');
      }}
    }}

    function showToast(msg) {{
      const toast = document.getElementById('toast');
      const toastText = document.getElementById('toast-text');
      if (toast && toastText) {{
        toastText.textContent = msg;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 2400);
      }}
    }}

    function showExpired() {{
      const dot = document.getElementById('expiry-dot');
      const text = document.getElementById('expiry-text');
      if (dot) {{
        dot.style.background = '#ef4444';
        dot.style.boxShadow = '0 0 0 3px rgba(239,68,68,0.2)';
      }}
      if (text) text.textContent = 'This link has expired';
      document.querySelectorAll('.btn-single-dl, #featured-brochure-btn, #btn-bar-download').forEach(btn => {{
        btn.style.opacity = '0.4';
        btn.style.pointerEvents = 'none';
      }});
    }}

    // Direct single download buttons click feedback
    document.addEventListener('click', (e) => {{
      const singleBtn = e.target.closest('.btn-single-dl, #featured-brochure-btn');
      if (singleBtn && singleBtn.style.pointerEvents !== 'none') {{
        showToast('Downloading brochure PDF...');
      }}
    }});

    // Initialize UI and fetch session details
    updateSelectionUI();
    loadCompanionData();
  </script>

</body>
</html>"""
