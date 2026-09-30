"""
backend/rag_calibration_log.py — structured, append-only decision log for
threshold calibration.

Writes one CSV row per confidence-RAG routing decision to
<project_root>/logs/rag_calibration.csv — separate from the normal
human-readable application log (main.py's logging.basicConfig), because
that log is for reading, this one is for LOADING INTO PANDAS/EXCEL and
sweeping threshold values against real traffic without re-running
anything.

Schema (CSV header):
    decision_id, timestamp, session_id, question, top_score, second_score,
    score_gap, top_entity, second_entity, confidence_band, ambiguous,
    route, event_type, user_reply, resolution, answer_source

Two fields need explaining:
  - confidence_band vs route: confidence_band is the PURE score bucket
    (HIGH/MEDIUM/LOW) before any ambiguity/scope logic runs; route is the
    actual final outcome (e.g. HIGH_AMBIGUOUS, MEDIUM_CONFIRM,
    LOW_OUT_OF_SCOPE). Keeping both lets you spot cases where the score
    said HIGH but the ambiguity check changed the outcome.
  - event_type / resolution: MEDIUM confirmations and HIGH ambiguity
    questions can't have a "did the user confirm?" answer in the SAME
    row they were asked in — that only exists on the NEXT turn. So this
    is an append-only event log: one row at decision time
    (event_type="decision", resolution="pending" for MEDIUM/AMBIGUOUS,
    or already-final for HIGH/LOW/OUT_OF_SCOPE), and a SECOND row when a
    pending one gets resolved (event_type="resolution"), joined by the
    same decision_id. Load both event types and join on decision_id for
    full picture; or just look at decision_id + resolution for a quick
    read.
"""
from __future__ import annotations

import os
import csv
import uuid
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("RNSIT_Kiosk.CalibrationLog")

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
CSV_PATH = os.path.join(_LOG_DIR, "rag_calibration.csv")

_HEADER = [
    "decision_id", "timestamp", "session_id", "question",
    "top_score", "second_score", "score_gap",
    "top_entity", "second_entity",
    "confidence_band", "ambiguous", "route",
    "event_type", "user_reply", "resolution", "answer_source",
]

_write_lock = asyncio.Lock()


def _ensure_file():
    os.makedirs(_LOG_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_HEADER)


async def log_decision(
    *, session_id: str | None, question: str,
    top_score: float, second_score: float | None,
    top_entity: str, second_entity: str,
    confidence_band: str, ambiguous: bool, route: str,
    resolution: str, answer_source: str = "",
) -> str:
    """
    Called once per query, at routing-decision time.
    Returns a decision_id — MEDIUM/AMBIGUOUS callers should hang onto it
    (store it in session_state["pending"]["decision_id"]) so the eventual
    log_resolution() call can be linked back to this row.
    `resolution` here should be "pending" for MEDIUM/AMBIGUOUS (awaiting
    the next turn), or the already-final outcome for HIGH/LOW/OUT_OF_SCOPE
    (e.g. "answered", "unknown_logged", "out_of_scope").
    """
    decision_id = uuid.uuid4().hex[:12]
    score_gap = (top_score - second_score) if second_score is not None else None
    row = [
        decision_id, datetime.now().isoformat(timespec="seconds"),
        session_id or "unknown", question,
        f"{top_score:.4f}", f"{second_score:.4f}" if second_score is not None else "",
        f"{score_gap:.4f}" if score_gap is not None else "",
        top_entity, second_entity,
        confidence_band, ambiguous, route,
        "decision", "", resolution, answer_source,
    ]
    await _append_row(row)
    return decision_id


async def log_resolution(
    *, decision_id: str, session_id: str | None, question: str,
    user_reply: str, resolution: str, answer_source: str = "",
) -> None:
    """
    Called when a PENDING MEDIUM confirmation or HIGH ambiguity question
    gets resolved on a later turn. `resolution` should be one of:
    "confirmed", "declined", "disambiguated_a", "disambiguated_b",
    "neither", or "abandoned" (user asked something unrelated instead of
    replying yes/no — the pending marker was cleared and a fresh search
    ran; log this so abandoned clarifications show up in your calibration
    data too, since a high abandon rate on a route is itself a signal).
    """
    row = [
        decision_id, datetime.now().isoformat(timespec="seconds"),
        session_id or "unknown", question,
        "", "", "",  # scores not re-logged on the resolution row
        "", "",
        "", "", "",  # confidence_band / ambiguous / route not repeated
        "resolution", user_reply, resolution, answer_source,
    ]
    await _append_row(row)


async def _append_row(row: list) -> None:
    try:
        async with _write_lock:
            await asyncio.to_thread(_write_row_sync, row)
    except Exception as e:
        logger.warning("[CALIBRATION-LOG] Failed to write row: %s", e)


def _write_row_sync(row: list) -> None:
    _ensure_file()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def read_recent_rows(limit: int = 200) -> list[dict]:
    """Best-effort read of the most recent `limit` rows, newest first —
    used by the admin dashboard's Calibration Log tab. Returns [] if the
    file doesn't exist yet (no queries logged since this feature shipped)."""
    if not os.path.exists(CSV_PATH):
        return []
    try:
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return list(reversed(rows))[:limit]
    except Exception as e:
        logger.warning("[CALIBRATION-LOG] Failed to read: %s", e)
        return []