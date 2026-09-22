"""
backend/admin_knowledge.py — Admin Knowledge-Update Loop
==========================================================
Implements the right-hand side of the architecture diagram:

  ADMIN reviews unanswered questions
    -> VERIFIED ANSWER (admin provides correct answer)
    -> knowledge_entries (MongoDB)
    -> ChromaDB upsert + embedding      (via RAGService, entry-keyed —
                                          see rag_upsert_entry)
    -> FUTURE QUERIES available in next RAG search

Mounted into backend/main.py as an APIRouter under /api/admin/knowledge.
Every write endpoint requires the same HTTP-Basic admin auth already used
elsewhere in the app (duplicated here, not imported from main.py, to
avoid a circular import — main.py is the one that includes this router).
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.admin_auth import authenticate_admin
from backend.database import (
    get_unanswered_questions,
    dismiss_unanswered_question,
    save_knowledge_entry,
    get_knowledge_entries,
    get_knowledge_entry,
    delete_knowledge_entry,
    mark_unanswered_resolved,
    get_rag_settings,
    save_rag_settings,
)
from backend.llm import rag_upsert_entry, rag_delete_entry
from backend.confidence_rag import invalidate_settings_cache
from backend.rag_calibration_log import read_recent_rows, CSV_PATH

logger = logging.getLogger("RNSIT_Kiosk.AdminKnowledge")
router = APIRouter(prefix="/api/admin/knowledge", tags=["Admin Knowledge"])


# ── Request/response models ─────────────────────────────────────────
class VerifyAnswerRequest(BaseModel):
    answer: str        = Field(..., min_length=1, description="Admin-verified answer text")
    category: str       = Field("General", description="e.g. Admissions, Placements, Hostel")
    entity_type: str    = Field("", description="e.g. department, facility, office")
    entity_name: str    = Field("", description="Structured label used by the ambiguity check, e.g. 'CSE Department'")


class KnowledgeEntryUpdateRequest(BaseModel):
    question: str | None = None
    answer: str | None = None
    category: str | None = None
    entity_type: str | None = None
    entity_name: str | None = None


class KnowledgeEntryCreateRequest(BaseModel):
    question: str
    answer: str
    category: str = "General"
    entity_type: str = ""
    entity_name: str = ""


class RagSettingsUpdateRequest(BaseModel):
    high_threshold: float | None = Field(None, ge=0, le=1, description="Score at/above this = HIGH confidence")
    near_threshold: float | None = Field(None, ge=0, le=1, description="Score at/above this = MEDIUM confidence")
    scope_threshold: float | None = Field(None, ge=0, le=1, description="LOW-branch scope-check score lower bound")
    ambiguity_gap: float | None = Field(None, ge=0, le=1, description="Max top1-top2 score gap still treated as ambiguous")
    text_similarity_threshold: float | None = Field(
        None, ge=0, le=1,
        description="Below this text-overlap ratio between top1/top2, chunks are treated as 'different topics' "
                     "when metadata is missing (fallback ambiguity signal)",
    )


# ── Confidence-RAG routing settings (admin-configurable, no redeploy) ──
@router.get("/settings")
async def get_settings(username: str = Depends(authenticate_admin)):
    """Current effective thresholds (DB override > env var > hardcoded default)."""
    return await get_rag_settings()


@router.put("/settings")
async def update_settings(body: RagSettingsUpdateRequest, username: str = Depends(authenticate_admin)):
    """
    Update one or more routing thresholds at runtime — no code change, no
    restart. Only the fields provided are changed; omitted fields keep
    their current value. Takes effect within ~30s (confidence_rag.py's
    settings cache TTL) or immediately if invalidate_settings_cache() below
    succeeds, which it always should in-process.
    """
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    new_settings = await save_rag_settings(updates)
    invalidate_settings_cache()
    logger.info("[ADMIN-KB] RAG settings updated by %s: %s", username, updates)
    return new_settings


# ── Unanswered-question queue ───────────────────────────────────────
@router.get("/unanswered")
async def list_unanswered(status_filter: str = "pending", limit: int = 200,
                           username: str = Depends(authenticate_admin)):
    """Admin review queue, sorted by ask_count desc (most-asked unknowns first)."""
    return {"questions": await get_unanswered_questions(status=status_filter, limit=limit)}


@router.post("/unanswered/{question_id}/verify")
async def verify_answer(question_id: str, body: VerifyAnswerRequest,
                         username: str = Depends(authenticate_admin)):
    """
    ADMIN VERIFIED ANSWER step. Looks up the original question text,
    writes a knowledge_entries doc, embeds+upserts it into ChromaDB under
    a stable entry_id (replacing any prior chunks for that entry_id), and
    marks the unanswered_questions row as resolved. From this point on the
    fact is available to future RAG searches — no retraining, no restart.
    """
    pending = await get_unanswered_questions(status="all", limit=10000)
    match = next((q for q in pending if q.get("question_id") == question_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Unanswered question not found")

    entry = await save_knowledge_entry(
        question=match["question_text"],
        answer=body.answer,
        category=body.category,
        entity_type=body.entity_type,
        entity_name=body.entity_name,
        created_by=username,
    )

    combined_text = f"Q: {entry['question']}\nA: {entry['answer']}"
    metadata = {
        "source": "admin_verified",
        "category": entry["category"],
        "entity_type": entry["entity_type"],
        "entity_name": entry["entity_name"],
    }
    try:
        added = await rag_upsert_entry(entry["entry_id"], combined_text, metadata)
    except Exception as e:
        logger.error("[ADMIN-KB] RAGService upsert failed for entry %s: %s", entry["entry_id"], e)
        raise HTTPException(status_code=502, detail=f"Saved to MongoDB but vector upsert failed: {e}")

    await mark_unanswered_resolved(question_id, entry["entry_id"])
    logger.info("[ADMIN-KB] Verified '%s' by %s -> entry_id=%s (%d chunk(s))",
                match["question_text"], username, entry["entry_id"], added)
    return {"message": "Answer verified and published.", "entry": entry, "chunks_added": added}


@router.delete("/unanswered/{question_id}")
async def dismiss_question(question_id: str, username: str = Depends(authenticate_admin)):
    """Admin dismisses an unknown question without answering it."""
    ok = await dismiss_unanswered_question(question_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Unanswered question not found")
    return {"message": "Dismissed."}


# ── Knowledge entries (verified, self-updating source of truth) ─────
@router.get("/entries")
async def list_entries(limit: int = 500, username: str = Depends(authenticate_admin)):
    return {"entries": await get_knowledge_entries(limit=limit)}


@router.post("/entries")
async def create_entry(body: KnowledgeEntryCreateRequest, username: str = Depends(authenticate_admin)):
    """
    Lets an admin add brand-new knowledge directly (not from the
    unanswered-questions queue) — e.g. proactively documenting a new
    facility. Same replace-on-upsert guarantee as verify_answer.
    """
    entry = await save_knowledge_entry(
        question=body.question, answer=body.answer, category=body.category,
        entity_type=body.entity_type, entity_name=body.entity_name, created_by=username,
    )
    combined_text = f"Q: {entry['question']}\nA: {entry['answer']}"
    metadata = {"source": "admin_verified", "category": entry["category"],
                "entity_type": entry["entity_type"], "entity_name": entry["entity_name"]}
    added = await rag_upsert_entry(entry["entry_id"], combined_text, metadata)
    return {"message": "Knowledge entry created.", "entry": entry, "chunks_added": added}


@router.put("/entries/{entry_id}")
async def update_entry(entry_id: str, body: KnowledgeEntryUpdateRequest,
                        username: str = Depends(authenticate_admin)):
    """
    Edit a previously-verified entry. Re-upserting under the SAME
    entry_id deletes the old chunk(s) first (see RAGService/rag_store.py
    ::upsert_entry) — this is the concrete mechanism behind "stale-data
    prevention": there is never a moment where both the old and new
    wording of the same fact are both searchable.
    """
    existing = await get_knowledge_entry(entry_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")

    entry = await save_knowledge_entry(
        entry_id=entry_id,
        question=body.question or existing["question"],
        answer=body.answer or existing["answer"],
        category=body.category or existing["category"],
        entity_type=body.entity_type if body.entity_type is not None else existing["entity_type"],
        entity_name=body.entity_name if body.entity_name is not None else existing["entity_name"],
        created_by=username,
    )
    combined_text = f"Q: {entry['question']}\nA: {entry['answer']}"
    metadata = {"source": "admin_verified", "category": entry["category"],
                "entity_type": entry["entity_type"], "entity_name": entry["entity_name"]}
    added = await rag_upsert_entry(entry["entry_id"], combined_text, metadata)
    return {"message": "Knowledge entry updated.", "entry": entry, "chunks_added": added}


@router.delete("/entries/{entry_id}")
async def remove_entry(entry_id: str, username: str = Depends(authenticate_admin)):
    ok = await delete_knowledge_entry(entry_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")
    try:
        await rag_delete_entry(entry_id)
    except Exception as e:
        logger.warning("[ADMIN-KB] Mongo entry deleted but RAGService cleanup failed: %s", e)
    return {"message": "Knowledge entry deleted."}


# ── Calibration log (see backend/rag_calibration_log.py) ───────────────
@router.get("/calibration-log")
async def view_calibration_log(limit: int = 200, username: str = Depends(authenticate_admin)):
    """Recent routing-decision rows, newest first — for the dashboard's
    Calibration Log tab. For real analysis (histograms, threshold
    sweeps), use /calibration-log/download and load it into pandas."""
    return {"rows": read_recent_rows(limit=limit)}


@router.get("/calibration-log/download")
async def download_calibration_log(username: str = Depends(authenticate_admin)):
    """Raw CSV file — every routing decision + resolution ever logged,
    structured for pandas/Excel. Columns documented in
    backend/rag_calibration_log.py's module docstring."""
    import os
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="No calibration data logged yet.")
    return FileResponse(CSV_PATH, media_type="text/csv", filename="rag_calibration.csv")