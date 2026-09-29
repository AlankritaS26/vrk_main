"""
backend/escalation.py — Human Handover & Escalation System
===========================================================

State machine:
  BOT_HANDLING → ESCALATION_REQUESTED → STAFF_NOTIFIED
                                          ↓ (staff accepts)     → STAFF_CONNECTED → RESOLVED
                                          ↓ (60s timeout fires) → BOT_HANDLING

Live state: Redis key  escalation:active:{session_id}  (TTL = ESCALATION_TIMEOUT_SECONDS + 10s buffer)
Persistent record:     MongoDB `escalations` collection (via database.save_escalation)

Endpoints mounted on main app:
  POST /escalation/request            — visitor or auto-trigger; starts the state machine
  GET  /escalation/active             — staff dashboard polls this (+ WS push for instant alerts)
  POST /escalation/accept/{session_id}— staff connects
  POST /escalation/resolve/{session_id}— staff marks resolved / times out
  GET  /staff                         — staff-facing dashboard HTML (Basic-auth protected)

Analytics hook: every escalation writes created_at, reason, resolution, resolve_time to MongoDB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

logger = logging.getLogger("RNSIT_Kiosk.Escalation")

# ── Config ───────────────────────────────────────────────────────────────────
ESCALATION_TIMEOUT_SECONDS: int = int(os.getenv("ESCALATION_TIMEOUT_SECONDS", "60"))
STAFF_USERNAME: str = os.getenv("STAFF_USERNAME", "staff")
STAFF_PASSWORD: str = os.getenv("STAFF_PASSWORD", "rnsit2024")
ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "111111")

# ── Escalation triggers (intent detection phrases) ────────────────────────────
HUMAN_INTENT_PHRASES = {
    "talk to a person", "talk to someone", "human staff", "real person",
    "speak to a human", "connect me to staff", "connect me to a staff",
    "connect to staff", "connect to a staff", "connect with staff",
    "connect me with staff", "talk to staff", "talk to a staff",
    "front desk", "need help from a person", "can i talk to someone",
    "can i speak to someone", "can i talk to staff", "i want a human",
    "speak to someone", "i need to talk to someone", "transfer me",
    "escalate", "supervisor", "manager", "receptionist", "reception desk",
    "call staff", "call a staff", "call someone", "get staff", "human help",
    "talk to human", "speak to staff", "human assistance", "front desk staff",
    "need a person", "want a person", "i want to speak to a person",
    "i want to talk to staff", "connect staff", "connect to front desk",
}

# Sensitive category keywords that auto-trigger escalation
SENSITIVE_KEYWORDS = {
    "emergency", "medical", "harassment", "complaint", "danger",
    "accident", "injured", "fire", "security", "police", "assault",
}

router = APIRouter(prefix="/escalation", tags=["escalation"])
_security = HTTPBasic(auto_error=False)
_DASHBOARD_TOKENS: dict[str, str] = {}

# ── Active timeout tasks: session_id → asyncio.Task ─────────────────────────
_timeout_tasks: dict[str, asyncio.Task] = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_redis():
    try:
        from backend.main import redis_client
        return redis_client
    except Exception:
        return None


def _get_manager():
    try:
        from backend.main import manager
        return manager
    except Exception:
        return None


def _escalation_key(session_id: str) -> str:
    return f"escalation:active:{session_id}"


def _store_escalation_state(session_id: str, state: dict) -> None:
    rc = _get_redis()
    if rc:
        try:
            rc.set(
                _escalation_key(session_id),
                json.dumps(state, default=str),
                ex=ESCALATION_TIMEOUT_SECONDS + 30,
            )
        except Exception as e:
            logger.warning("[ESC] Redis write failed: %s", e)
    # Always keep in-memory too for instant reads
    _ESCALATION_STORE[session_id] = state


def _get_escalation_state(session_id: str) -> dict | None:
    rc = _get_redis()
    if rc:
        try:
            raw = rc.get(_escalation_key(session_id))
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    return _ESCALATION_STORE.get(session_id)


def _clear_escalation_state(session_id: str) -> None:
    rc = _get_redis()
    if rc:
        try:
            rc.delete(_escalation_key(session_id))
        except Exception:
            pass
    _ESCALATION_STORE.pop(session_id, None)


def _get_all_active_escalations() -> list[dict]:
    """Return all open escalations (STAFF_NOTIFIED or STAFF_CONNECTED)."""
    rc = _get_redis()
    results: list[dict] = {}
    if rc:
        try:
            cursor = 0
            while True:
                cursor, keys = rc.scan(cursor, match="escalation:active:*", count=100)
                for k in keys:
                    raw = rc.get(k)
                    if raw:
                        state = json.loads(raw)
                        sid = state.get("session_id", "")
                        results[sid] = state
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("[ESC] Redis scan failed: %s", e)
    # Merge with in-memory store (catches no-Redis case)
    for sid, state in _ESCALATION_STORE.items():
        if sid not in results:
            results[sid] = state
    return list(results.values())


# In-memory fallback when Redis unavailable
_ESCALATION_STORE: dict[str, dict] = {}


def _authenticate_staff(
    credentials: HTTPBasicCredentials | None = Depends(_security),
    token: str | None = Query(default=None),
    x_staff_token: str | None = Header(default=None),
) -> str:
    """Accept staff token, staff Basic Auth, or admin Basic Auth."""
    active_tok = token or x_staff_token
    if active_tok and active_tok in _DASHBOARD_TOKENS:
        return _DASHBOARD_TOKENS[active_tok]

    if credentials:
        ok_staff = (secrets.compare_digest(credentials.username, STAFF_USERNAME) and
                    secrets.compare_digest(credentials.password, STAFF_PASSWORD))
        ok_admin = (secrets.compare_digest(credentials.username, ADMIN_USERNAME) and
                    secrets.compare_digest(credentials.password, ADMIN_PASSWORD))
        if ok_staff or ok_admin:
            return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid staff credentials or expired dashboard token",
        headers={"WWW-Authenticate": "Basic"},
    )


async def _escalation_timeout_task(session_id: str) -> None:
    """
    Fires after ESCALATION_TIMEOUT_SECONDS if no staff member accepted.
    Reverts the kiosk to bot mode and cleans up.
    """
    try:
        await asyncio.sleep(ESCALATION_TIMEOUT_SECONDS)
        state = _get_escalation_state(session_id)
        if not state or state.get("status") not in ("STAFF_NOTIFIED",):
            return  # staff already accepted or resolved

        logger.info("[ESC] Timeout for session=%s — reverting to bot mode", session_id[:8])
        state["status"] = "TIMEOUT"
        state["updated_at"] = datetime.now().isoformat()
        _store_escalation_state(session_id, state)

        # Update MongoDB
        try:
            from backend.database import resolve_escalation
            await resolve_escalation(session_id, resolution="timeout_no_staff")
        except Exception as e:
            logger.warning("[ESC] Timeout DB update failed: %s", e)

        # Broadcast to kiosk
        manager = _get_manager()
        if manager:
            await manager.broadcast({
                "type":       "escalation_timeout",
                "session_id": session_id,
                "message":    "No staff available right now. Nova will continue helping you.",
            })

        _clear_escalation_state(session_id)

        # Log analytics
        _log_escalation_analytics(state, outcome="timeout")

    except asyncio.CancelledError:
        pass  # task cancelled because staff accepted — normal
    finally:
        _timeout_tasks.pop(session_id, None)


async def cancel_escalation(session_id: str, reason: str = "visitor_cancelled") -> bool:
    """Cancels an active escalation for the session and notifies kiosk & dashboard."""
    state = _get_escalation_state(session_id)
    task = _timeout_tasks.pop(session_id, None)
    if task:
        task.cancel()

    if not state and not task:
        return False

    if state:
        state["status"] = "CANCELLED"
        state["updated_at"] = datetime.now().isoformat()
        _log_escalation_analytics(state, outcome="cancelled")

    try:
        from backend.database import resolve_escalation
        await resolve_escalation(session_id, resolution=reason)
    except Exception as e:
        logger.warning("[ESC] cancel: DB update failed: %s", e)

    _clear_escalation_state(session_id)

    manager = _get_manager()
    if manager:
        await manager.broadcast({
            "type":       "escalation_cancelled",
            "session_id": session_id,
            "message":    "Staff connection cancelled. Nova will continue helping you.",
        })
    logger.info("[ESC] Escalation cancelled for session=%s reason=%s", session_id[:8], reason)
    return True


def _log_escalation_analytics(state: dict, outcome: str) -> None:
    """Write a structured analytics record to the logger (extend to DB if needed)."""
    created = state.get("created_at", "")
    resolved = datetime.now().isoformat()
    reason = state.get("reason", "unknown")
    user_name = state.get("user_name", "Guest")
    session_id = state.get("session_id", "")

    try:
        from datetime import datetime as dt
        c = dt.fromisoformat(created)
        r = dt.fromisoformat(resolved)
        resolution_seconds = int((r - c).total_seconds())
    except Exception:
        resolution_seconds = -1

    logger.info(
        "[ESC ANALYTICS] session=%s user=%s reason=%s outcome=%s resolution_time=%ds",
        session_id[:8], user_name, reason, outcome, resolution_seconds,
    )


def _detect_human_intent(text: str) -> bool:
    """True if the visitor's message explicitly requests a human."""
    t = text.lower().strip()
    if any(phrase in t for phrase in HUMAN_INTENT_PHRASES):
        return True
    if re.search(r"\b(?:connect|talk|speak|transfer|call)\s+(?:me\s+)?(?:to|with)\s+(?:a\s+)?(?:staff|human|person|reception|receptionist|front\s*desk)\b", t):
        return True
    if re.search(r"\b(?:human|staff)\s+(?:help|assistance|support|member)\b", t):
        return True
    if re.search(r"\b(?:want|need)\s+(?:a\s+)?(?:human|person|staff)\b", t):
        return True
    return False


def _detect_sensitive(text: str) -> bool:
    """True if the message contains a sensitive keyword requiring escalation."""
    t = text.lower()
    return any(kw in t for kw in SENSITIVE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL API (called from main.py's /ask pipeline)
# ─────────────────────────────────────────────────────────────────────────────

async def maybe_escalate(question: str, answer: str, session_id: str,
                         face_id: str | None, user_name: str,
                         low_confidence_count: int,
                         transcript: list[dict] | None = None) -> bool:
    """
    Called from the /ask pipeline after each answer.
    Checks all escalation triggers. Returns True if escalation was triggered.
    """
    reason = None

    if _detect_sensitive(question):
        reason = "sensitive_topic"
    elif _detect_human_intent(question):
        reason = "user_request"
    elif low_confidence_count >= 3:
        reason = "repeated_low_confidence"

    if reason is None:
        return False

    # Don't double-escalate an already-open escalation for this session
    existing = _get_escalation_state(session_id)
    if existing and existing.get("status") in ("STAFF_NOTIFIED", "STAFF_CONNECTED"):
        return False

    logger.info("[ESC] Auto-escalating session=%s reason=%s", session_id[:8], reason)
    await _trigger_escalation(
        session_id=session_id,
        face_id=face_id,
        user_name=user_name,
        reason=reason,
        transcript=transcript or [],
    )
    return True


async def _trigger_escalation(
    session_id: str,
    face_id: str | None,
    user_name: str,
    reason: str,
    transcript: list[dict],
) -> dict:
    """Core escalation logic shared by the REST endpoint and the auto-escalation path."""
    now = datetime.now().isoformat()
    state = {
        "session_id": session_id,
        "face_id":    face_id or "",
        "user_name":  user_name or "Guest",
        "reason":     reason,
        "status":     "STAFF_NOTIFIED",
        "created_at": now,
        "updated_at": now,
        "transcript": transcript[-10:] if transcript else [],  # last 10 messages
    }

    # Persist to Redis (live state) + MongoDB (permanent record)
    _store_escalation_state(session_id, state)
    try:
        from backend.database import save_escalation
        await save_escalation(
            session_id=session_id,
            face_id=face_id,
            user_name=user_name,
            reason=reason,
            transcript=transcript[-10:] if transcript else [],
        )
    except Exception as e:
        logger.warning("[ESC] MongoDB save failed: %s", e)

    # Broadcast escalation_pending to kiosk frontend
    manager = _get_manager()
    if manager:
        await manager.broadcast({
            "type":       "escalation_pending",
            "session_id": session_id,
            "reason":     reason,
            "user_name":  user_name,
            "message":    "Connecting you to a staff member at the front desk. Please wait…",
        })

    # Start the timeout countdown
    task = asyncio.create_task(_escalation_timeout_task(session_id))
    if session_id in _timeout_tasks:
        _timeout_tasks[session_id].cancel()
    _timeout_tasks[session_id] = task

    logger.info("[ESC] Escalation triggered: session=%s reason=%s", session_id[:8], reason)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class EscalationRequest(BaseModel):
    reason: str = "user_request"
    message: str = ""


@router.post("/request")
async def request_escalation(payload: EscalationRequest):
    """
    Visitor or frontend triggers a human handover request.
    Can be called directly (e.g. from the 'Talk to Staff' button).
    """
    try:
        from backend.main import active_session, message_log
    except ImportError:
        active_session = None
        message_log = []

    if not active_session:
        raise HTTPException(status_code=404, detail="No active session.")

    session_id = active_session.get("session_id", "")
    face_id    = active_session.get("face_id", "")
    user_name  = active_session.get("user_name", "Guest")

    # Build transcript from message_log
    transcript = [
        {"speaker": m.get("speaker", ""), "text": m.get("text", "")}
        for m in message_log[-10:]
    ]

    # Honour the optional extra message
    if payload.message:
        transcript.append({"speaker": "visitor", "text": payload.message})

    existing = _get_escalation_state(session_id)
    if existing and existing.get("status") in ("STAFF_NOTIFIED", "STAFF_CONNECTED"):
        return {"status": "already_pending", "session_id": session_id}

    state = await _trigger_escalation(
        session_id=session_id,
        face_id=face_id,
        user_name=user_name,
        reason=payload.reason,
        transcript=transcript,
    )
    return {"status": "escalation_triggered", "session_id": session_id, "reason": state["reason"]}


@router.get("/active")
async def get_active_escalations(username: str = Depends(_authenticate_staff)):
    """Return all open escalations for the staff dashboard poll."""
    escalations = _get_all_active_escalations()
    # Filter to only truly open ones
    open_ones = [e for e in escalations if e.get("status") in ("STAFF_NOTIFIED", "STAFF_CONNECTED")]
    return {"escalations": open_ones, "count": len(open_ones)}


@router.post("/accept/{session_id}")
async def accept_escalation(session_id: str, username: str = Depends(_authenticate_staff)):
    """Staff accepts an escalation — cancels the timeout and notifies the kiosk."""
    state = _get_escalation_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="No active escalation for this session.")
    if state.get("status") not in ("STAFF_NOTIFIED",):
        raise HTTPException(status_code=400, detail=f"Escalation is in state '{state.get('status')}', cannot accept.")

    # Cancel the timeout task
    task = _timeout_tasks.pop(session_id, None)
    if task:
        task.cancel()

    state["status"] = "STAFF_CONNECTED"
    state["updated_at"] = datetime.now().isoformat()
    state["staff_id"] = username
    _store_escalation_state(session_id, state)

    # Update MongoDB
    try:
        from backend.database import escalations_collection
        # Motor's update_one does not support 'sort' — find the newest open doc first
        esc_doc = await escalations_collection.find_one(
            {"session_id": session_id, "status": "STAFF_NOTIFIED"},
            sort=[("created_at", -1)],
        )
        if esc_doc:
            await escalations_collection.update_one(
                {"_id": esc_doc["_id"]},
                {"$set": {"status": "STAFF_CONNECTED", "staff_id": username,
                          "updated_at": state["updated_at"]}},
            )
    except Exception as e:
        logger.warning("[ESC] accept: DB update failed: %s", e)

    # Notify kiosk
    manager = _get_manager()
    if manager:
        await manager.broadcast({
            "type":       "escalation_connected",
            "session_id": session_id,
            "staff_id":   username,
            "message":    "A staff member has connected. Please hold.",
        })

    logger.info("[ESC] Staff %s accepted escalation for session=%s", username, session_id[:8])
    return {"status": "accepted", "session_id": session_id}


@router.post("/resolve/{session_id}")
async def resolve_escalation_endpoint(session_id: str, resolution: str = "resolved",
                                      username: str = Depends(_authenticate_staff)):
    """Staff marks the escalation as resolved."""
    state = _get_escalation_state(session_id)
    if not state:
        # May already be cleaned up — still try DB update
        try:
            from backend.database import resolve_escalation
            await resolve_escalation(session_id, resolution=resolution, staff_id=username)
        except Exception:
            pass
        return {"status": "resolved", "session_id": session_id}

    # Cancel any pending timeout
    task = _timeout_tasks.pop(session_id, None)
    if task:
        task.cancel()

    state["status"] = "RESOLVED"
    state["updated_at"] = datetime.now().isoformat()
    _log_escalation_analytics(state, outcome=resolution)

    try:
        from backend.database import resolve_escalation
        await resolve_escalation(session_id, resolution=resolution, staff_id=username)
    except Exception as e:
        logger.warning("[ESC] resolve: DB update failed: %s", e)

    _clear_escalation_state(session_id)

    # Notify kiosk to return to bot mode
    manager = _get_manager()
    if manager:
        await manager.broadcast({
            "type":       "escalation_resolved",
            "session_id": session_id,
            "resolution": resolution,
            "message":    "Your query has been addressed. Nova is ready to help you further.",
        })

    logger.info("[ESC] Resolved session=%s by %s resolution=%s", session_id[:8], username, resolution)
    return {"status": "resolved", "session_id": session_id}


@router.post("/cancel/{session_id}")
async def cancel_escalation_endpoint(session_id: str):
    """Visitor voice command ('stop') or client cancels the pending escalation."""
    ok = await cancel_escalation(session_id, reason="visitor_cancelled")
    return {"status": "cancelled" if ok else "not_found", "session_id": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# STAFF DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
async def staff_dashboard(username: str = Depends(_authenticate_staff)):
    """
    Staff-facing live dashboard showing open escalation requests.
    Polls /escalation/active every 5s AND connects to the WebSocket for instant alerts.
    """
    dash_token = secrets.token_urlsafe(32)
    _DASHBOARD_TOKENS[dash_token] = username

    try:
        from backend.main import BACKEND_URL
        backend_url = BACKEND_URL
    except Exception:
        backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")

    ws_url = backend_url.replace("http://", "ws://").replace("https://", "wss://")

    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RNSIT — Staff Escalation Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: #0f172a; color: #f1f5f9;
      min-height: 100vh;
    }}
    header {{
      background: linear-gradient(90deg, #1e3a5f, #1a56db);
      padding: 18px 28px;
      display: flex; align-items: center; justify-content: space-between;
    }}
    header h1 {{ font-size: 20px; font-weight: 700; }}
    header .badge {{
      background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 20px;
      font-size: 12px; font-weight: 600;
    }}
    .status-bar {{
      background: #1e293b; padding: 8px 28px;
      display: flex; align-items: center; gap: 12px; font-size: 13px; color: #94a3b8;
    }}
    .dot {{ width:8px; height:8px; border-radius:50%; background:#10b981;
            box-shadow:0 0 0 3px rgba(16,185,129,.2); animation: pulse 2s infinite; }}
    .dot.red {{ background:#ef4444; box-shadow:0 0 0 3px rgba(239,68,68,.2); }}
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}

    main {{ padding: 28px; max-width: 1000px; margin: 0 auto; }}
    h2 {{ font-size: 16px; font-weight: 600; color: #94a3b8; margin-bottom: 16px;
           text-transform: uppercase; letter-spacing: 1px; }}

    #queue {{ display: flex; flex-direction: column; gap: 14px; }}

    .card {{
      background: #1e293b; border: 1px solid #334155;
      border-radius: 14px; padding: 20px 24px;
      transition: border-color 0.2s;
    }}
    .card.urgent {{ border-color: #ef4444; }}
    .card.connected {{ border-color: #10b981; }}

    .card-top {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; }}
    .visitor-name {{ font-size:18px; font-weight:700; color:#f8fafc; }}
    .reason-badge {{
      padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
      background: rgba(239,68,68,.15); color: #fca5a5;
    }}
    .reason-badge.sensitive {{ background: rgba(245,158,11,.15); color: #fcd34d; }}
    .reason-badge.connected {{ background: rgba(16,185,129,.15); color: #6ee7b7; }}

    .meta {{ font-size: 13px; color: #64748b; margin-bottom: 14px; line-height: 1.6; }}
    .transcript {{
      background: #0f172a; border-radius: 8px; padding: 12px 14px;
      font-size: 12px; color: #94a3b8; max-height: 160px; overflow-y: auto;
      margin-bottom: 14px; line-height: 1.7;
    }}
    .transcript .visitor {{ color: #93c5fd; }}
    .transcript .nova {{ color: #86efac; }}

    .actions {{ display:flex; gap:10px; flex-wrap:wrap; }}
    .btn {{
      padding: 8px 18px; border:none; border-radius: 8px;
      font-size: 13px; font-weight: 600; cursor: pointer; transition: .15s;
    }}
    .btn-accept {{ background:#10b981; color:#fff; }}
    .btn-accept:hover {{ background:#059669; }}
    .btn-resolve {{ background:#6366f1; color:#fff; }}
    .btn-resolve:hover {{ background:#4f46e5; }}
    .btn-timeout {{ background:#374151; color:#9ca3af; }}
    .btn-timeout:hover {{ background:#4b5563; }}

    .empty {{
      text-align:center; padding: 60px 20px; color:#475569;
    }}
    .empty .icon {{ font-size:48px; margin-bottom:12px; }}
    .empty p {{ font-size:16px; }}

    #alert-banner {{
      display:none; position:fixed; top:0; left:0; right:0;
      background:#ef4444; color:#fff; text-align:center;
      padding:12px; font-size:15px; font-weight:600; z-index:999;
      animation: slidein .3s ease;
    }}
    @keyframes slidein {{ from{{transform:translateY(-100%)}} to{{transform:translateY(0)}} }}
  </style>
</head>
<body>
  <div id="alert-banner">🚨 New escalation request!</div>

  <header>
    <h1>🎯 Staff Escalation Dashboard</h1>
    <span class="badge">Logged in as: {username}</span>
  </header>

  <div class="status-bar">
    <div class="dot" id="ws-dot"></div>
    <span id="ws-status">Connecting…</span>
    &nbsp;·&nbsp;
    <span id="last-refresh">Polling every 5s</span>
  </div>

  <main>
    <h2 id="queue-title">Active Escalations</h2>
    <div id="queue"><div class="empty"><div class="icon">✅</div><p>No active escalations. All clear!</p></div></div>
  </main>

  <script>
    const BACKEND = window.location.origin || '{backend_url}';
    const WS_URL  = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + (window.location.host || '127.0.0.1:8001') + '/ws';
    const TOKEN   = '{dash_token}';

    let queue = {{}};

    function timeAgo(iso) {{
      const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
      if (diff < 60) return `${{diff}}s ago`;
      if (diff < 3600) return `${{Math.floor(diff/60)}}m ago`;
      return `${{Math.floor(diff/3600)}}h ago`;
    }}

    function reasonLabel(reason) {{
      const map = {{
        user_request: 'User Request',
        repeated_low_confidence: 'Nova Struggling',
        sensitive_topic: 'Sensitive Topic ⚠️',
        auto: 'Auto-Escalated',
      }};
      return map[reason] || reason;
    }}

    function renderQueue() {{
      const el = document.getElementById('queue');
      const items = Object.values(queue);
      if (items.length === 0) {{
        el.innerHTML = '<div class="empty"><div class="icon">✅</div><p>No active escalations. All clear!</p></div>';
        document.getElementById('queue-title').textContent = 'Active Escalations';
        return;
      }}
      document.getElementById('queue-title').textContent = `Active Escalations (${{items.length}})`;
      el.innerHTML = items.map(e => {{
        const isConnected = e.status === 'STAFF_CONNECTED';
        const isSensitive = e.reason === 'sensitive_topic';
        const transcript  = (e.transcript || []).map(m =>
          `<div class="${{m.speaker === 'visitor' ? 'visitor' : 'nova'}}"><strong>${{m.speaker === 'visitor' ? '👤 Visitor' : '🤖 Nova'}}:</strong> ${{m.text || ''}}</div>`
        ).join('');
        return `
        <div class="card ${{isConnected ? 'connected' : isSensitive ? 'urgent' : ''}}" id="card-${{e.session_id}}">
          <div class="card-top">
            <span class="visitor-name">👤 ${{e.user_name || 'Guest'}}</span>
            <span class="reason-badge ${{isConnected ? 'connected' : isSensitive ? 'sensitive' : ''}}">${{isConnected ? '✅ Connected' : reasonLabel(e.reason)}}</span>
          </div>
          <div class="meta">
            Session: <code>${{(e.session_id||'').slice(0,8)}}</code>
            &nbsp;·&nbsp; Escalated ${{timeAgo(e.created_at)}}
            &nbsp;·&nbsp; Status: <strong>${{e.status}}</strong>
          </div>
          ${{transcript ? `<div class="transcript">${{transcript}}</div>` : ''}}
          <div class="actions">
            ${{!isConnected ? `<button class="btn btn-accept" onclick="accept('${{e.session_id}}')">📞 Accept</button>` : ''}}
            <button class="btn btn-resolve" onclick="resolve('${{e.session_id}}', 'resolved')">✅ Mark Resolved</button>
            <button class="btn btn-timeout" onclick="resolve('${{e.session_id}}', 'dismissed')">✖ Dismiss</button>
          </div>
        </div>`;
      }}).join('');
    }}

    async function fetchQueue() {{
      try {{
        const r = await fetch(BACKEND + '/escalation/active?token=' + encodeURIComponent(TOKEN));
        if (r.ok) {{
          const data = await r.json();
          queue = {{}};
          for (const e of (data.escalations || [])) {{
            queue[e.session_id] = e;
          }}
          renderQueue();
          document.getElementById('last-refresh').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
        }}
      }} catch(e) {{ console.error('Poll error', e); }}
    }}

    async function accept(sessionId) {{
      await fetch(BACKEND + '/escalation/accept/' + sessionId + '?token=' + encodeURIComponent(TOKEN), {{
        method: 'POST'
      }});
      await fetchQueue();
    }}

    async function resolve(sessionId, resolution) {{
      await fetch(BACKEND + '/escalation/resolve/' + sessionId + '?resolution=' + encodeURIComponent(resolution) + '&token=' + encodeURIComponent(TOKEN), {{
        method: 'POST'
      }});
      delete queue[sessionId];
      renderQueue();
    }}

    // WebSocket for instant push alerts
    function connectWs() {{
      const ws = new WebSocket(WS_URL);
      const dot = document.getElementById('ws-dot');
      const statusEl = document.getElementById('ws-status');
      ws.onopen = () => {{
        dot.classList.remove('red');
        statusEl.textContent = 'Live · WebSocket connected';
      }};
      ws.onmessage = (e) => {{
        try {{
          const msg = JSON.parse(e.data);
          if (msg.type === 'escalation_pending') {{
            const banner = document.getElementById('alert-banner');
            banner.textContent = `🚨 New escalation from ${{msg.user_name || 'Visitor'}}: ${{msg.reason || ''}}`;
            banner.style.display = 'block';
            setTimeout(() => {{ banner.style.display = 'none'; }}, 6000);
            fetchQueue();
          }} else if (msg.type === 'escalation_resolved' || msg.type === 'escalation_timeout' || msg.type === 'escalation_cancelled') {{
            delete queue[msg.session_id];
            renderQueue();
          }}
        }} catch(_) {{}}
      }};
      ws.onclose = () => {{
        dot.classList.add('red');
        statusEl.textContent = 'WebSocket disconnected — reconnecting…';
        setTimeout(connectWs, 3000);
      }};
    }}

    fetchQueue();
    setInterval(fetchQueue, 5000);
    connectWs();
  </script>
</body>
</html>""")
