import os
import re
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("RNSIT_Kiosk.Database")

# Secure connection setup
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)

# Your global database and collections
db = client["rnsit_db"]
college_collection = db["college_profile"]  # Stores RNSIT details / FAQs
faces_collection = db["faces"]              # Stores face embeddings/IDs
sessions_collection = db["sessions"]        # Track active/inactive kiosk sessions
interactions_collection = db["interactions"]# Chat history logs

# ── NEW (Confidence-RAG / Admin Knowledge-Update Loop) ─────────────────
# unanswered_questions: LOW-confidence queries that were judged RNSIT-
#   related (in-domain) but had no good retrieval match. Deduplicated by
#   normalized text; each repeat asking just bumps `ask_count` instead of
#   creating a new row, so the admin dashboard can sort by "most asked
#   unknowns" — the queue an admin actually wants to work through first.
# knowledge_entries: admin-verified Q&A written through the admin panel
#   (no code change needed). Each entry also gets embedded + upserted into
#   the RAGService ChromaDB collection under the SAME entry_id, so editing
#   an entry here always deletes-then-reinserts its vector chunk(s) —
#   preventing stale/duplicate chunks from an old wording of the same fact.
unanswered_collection      = db["unanswered_questions"]
knowledge_entries_collection = db["knowledge_entries"]
settings_collection        = db["kiosk_settings"]

# Hardcoded fallback if neither the DB nor an env var has a value.
# Precedence used by get_rag_settings(): DB value > env var > this default.
_RAG_SETTINGS_HARDCODED_DEFAULTS = {
    "high_threshold":            0.55,
    "near_threshold":            0.35,
    "scope_threshold":           0.15,
    "ambiguity_gap":             0.06,
    "text_similarity_threshold": 0.40,
}
_RAG_SETTINGS_ENV_VARS = {
    "high_threshold":            "RAG_HIGH_THRESHOLD",
    "near_threshold":            "RAG_NEAR_THRESHOLD",
    "scope_threshold":           "RAG_SCOPE_THRESHOLD",
    "ambiguity_gap":             "RAG_AMBIGUITY_GAP",
    "text_similarity_threshold": "RAG_TEXT_SIMILARITY_THRESHOLD",
}
_RAG_SETTINGS_DOC_ID = "rag_thresholds"

# ==========================================
# CORE DB LIFECYCLE HANDLERS
# ==========================================

async def get_kiosk_data():
    """Fetches the single main RNSIT document containing all info and FAQs."""
    try:
        data = await college_collection.find_one({})
        if data:
            data["_id"] = str(data["_id"])
        return data
    except Exception as e:
        logger.error(f"MongoDB Error fetching kiosk data: {e}")
        return None

# ==========================================
# SESSION MANAGEMENT (Replaces PostgreSQL tables)
# ==========================================

async def save_session(session_id: str, face_id: str | None, user_name: str,
                       is_returning: bool, visit_count: int,
                       continued_from: str | None = None):
    """
    Saves or logs a kiosk session. 
    MongoDB handles creating the session record dynamically without a strict schema definition.
    """
    try:
        await sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "face_id": face_id,
                    "user_name": user_name,
                    "is_returning": is_returning,
                    "visit_count": visit_count,
                    "is_active": True,
                    "last_activity": datetime.now().isoformat(),
                    **({"continued_from": continued_from} if continued_from else {}),
                },
                "$setOnInsert": {
                    "started_at": datetime.now().isoformat()
                }
            },
            upsert=True
        )
        logger.info(f"[MongoDB] Session indexed: {session_id}")
    except Exception as e:
        logger.error(f"Error tracking session: {e}")

async def save_interaction(session_id: str, question: str, answer: str, face_id: str | None = None, user_name: str | None = None):
    """
    Logs individual conversational components directly into cloud transactions.
    Records session_id, user_name, input_text, response_text, timestamp, and face_id.
    """
    try:
        doc = {
            "session_id": session_id,
            "input_text": question,
            "response_text": answer,
            "user_name": user_name or "Guest",
            "timestamp": datetime.now().isoformat(),
        }
        if face_id:
            doc["face_id"] = face_id
        await interactions_collection.insert_one(doc)
    except Exception as e:
        logger.error(f"Error logging conversational interaction: {e}")

# ==========================================
# BIOMETRICS & FACE RETRIEVAL HANDLERS
# ==========================================

async def save_face_encoding(face_id: str, name: str, encoding: list,
                             encodings: list | None = None):
    """Saves or updates a biometric profile mapping vector representations directly.
    
    When `encoding` is an empty list, saves a name-only placeholder record
    (has_encoding=False) so the visitor appears in the admin Face Tracks tab
    without affecting the face recognition matching pipeline.
    """
    try:
        now_str = datetime.now().isoformat()
        has_enc = bool(encoding and len(encoding) > 0)
        set_doc = {
            "name": name,
            "encoding": encoding,
            "encodings": (encodings or ([encoding] if has_enc else [])),
            "last_seen": now_str,
            "detected_at": now_str,
            "has_encoding": has_enc,
        }
        await faces_collection.update_one(
            {"face_id": face_id},
            {
                "$set": set_doc,
                "$setOnInsert": {
                    "registered_at": now_str,
                    "visit_count": 1
                }
            },
            upsert=True
        )
        tag = "with encoding" if has_enc else "(name-only placeholder)"
        logger.info(f"[MongoDB] Face Registered into Face Tracks {tag}: {name} (face_id={face_id[:8]})")
    except Exception as e:
        logger.error(f"Error updating biometric vector signature: {e}")


async def update_face_name_with_alias(face_id: str, new_name: str) -> bool:
    """Update the face record's display name and preserve the old name in
    `name_history` so the admin dashboard can show both the original and
    any visitor-changed names. Also syncs user_name across sessions with this face_id.
    """
    try:
        # First fetch the current name so we can archive it
        doc = await faces_collection.find_one({"face_id": face_id}, {"name": 1})
        if not doc:
            logger.warning(f"[MongoDB] update_face_name_with_alias: face_id {face_id} not found")
            return False
        old_name = doc.get("name", "")
        result = await faces_collection.update_one(
            {"face_id": face_id},
            {
                "$set": {
                    "name": new_name,
                    "name_updated_at": datetime.now().isoformat(),
                },
                # Push the old name into name_history (deduplicating with $addToSet)
                "$addToSet": {"name_history": old_name} if old_name and old_name != new_name else {},
            }
        )
        # Also sync user_name across sessions collection for this face_id
        await sessions_collection.update_many(
            {"face_id": face_id},
            {"$set": {"user_name": new_name}}
        )
        logger.info(f"[MongoDB] Face name updated across DB: '{old_name}' -> '{new_name}' ({face_id[:8]})")
        return result.matched_count > 0
    except Exception as e:
        logger.error(f"Error updating face name with alias: {e}")
        return False

async def update_session_user_name(session_id: str, new_name: str) -> bool:
    """Updates user_name in sessions collection by session_id."""
    try:
        res = await sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"user_name": new_name}}
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error(f"Error updating session user_name: {e}")
        return False

async def get_all_face_encodings():
    """Retrieves all registered biometric keys for local processing frames.
    
    Only returns records that have a real encoding vector (has_encoding=True or
    legacy records without the flag but with a non-empty encoding list). 
    Name-only placeholder records (has_encoding=False, empty encoding) are 
    intentionally excluded — passing a zero-length vector to _match() in
    detection.py would give garbage cosine similarities.

    Returns 'encoding' AND 'encodings' (plural) so detection._match() can
    use multi-template matching (best-of-N similarity) for better accuracy.
    """
    try:
        # Use $type: 4 (array) + $not $size 0 to correctly filter non-empty arrays.
        # NOTE: chaining two $ne on the same field is invalid MongoDB — the second
        # silently overrides the first, so "$ne": None, "$ne": [] would only keep
        # the $ne: [] check, allowing null values through.
        cursor = faces_collection.find({
            "has_encoding": {"$ne": False},
            "encoding": {
                "$type": 4,           # must be an array
                "$not": {"$size": 0}  # must be non-empty
            },
        })
        results = []
        async for doc in cursor:
            enc = doc.get("encoding")
            if not enc or len(enc) == 0:
                continue   # extra guard — skip truly empty lists
            encs = doc.get("encodings") or [enc]
            # filter any empty sub-vectors that might have crept in
            encs = [e for e in encs if e and len(e) > 0]
            if not encs:
                encs = [enc]
            results.append({
                "face_id": doc["face_id"],
                "name": doc["name"],
                "encoding": enc,
                "encodings": encs,
                "visit_count": doc.get("visit_count", 1),
            })
        logger.info(f"[DB] get_all_face_encodings: returning {len(results)} biometric profile(s)")
        return results
    except Exception as e:
        logger.error(f"Error querying biometric records: {e}")
        return []

async def update_face_seen(face_id: str):
    """
    Increments visit metrics asynchronously when recognized by the camera loop.
    Fixes the ImportError in main.py.
    """
    try:
        await faces_collection.update_one(
            {"face_id": face_id},
            {
                "$inc": {"visit_count": 1},
                "$set": {"last_seen": datetime.now().isoformat()}
            }
        )
        logger.info(f"[MongoDB] Biometric presence incremented for profile ID: {face_id}")
    except Exception as e:
        logger.error(f"Error updating presence timestamp: {e}")

async def delete_face_by_name(name: str):
    """
    Deletes all face records matching a visitor name (case-insensitive).
    Returns the list of deleted face_ids so the caller can clean up
    any on-disk face image directories.
    """
    try:
        cursor = faces_collection.find(
            {"name": {"$regex": f"^{name}$", "$options": "i"}}, {"face_id": 1}
        )
        face_ids = [doc["face_id"] async for doc in cursor if doc.get("face_id")]
        if face_ids:
            await faces_collection.delete_many({"face_id": {"$in": face_ids}})
            logger.info(f"[DB] Deleted {len(face_ids)} face record(s) for '{name}'")
        return face_ids
    except Exception as e:
        logger.error(f"MongoDB Error deleting faces for '{name}': {e}")
        return []


# ==========================================
# SESSION CONTINUITY (30-day resume)
# ==========================================

async def find_recent_session_by_face(face_id: str, days: int = 30):
    """
    Most recent session for this PERSON within `days`.
    This is what makes a returning visitor continue their existing
    session_id instead of getting a brand-new one. Mongo is the source
    of truth here; Redis (if present) is only ever a cache in front.
    """
    if not face_id:
        return None
    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        return await sessions_collection.find_one(
            {"face_id": face_id, "last_activity": {"$gte": cutoff}},
            sort=[("last_activity", -1)],
        )
    except Exception as e:
        logger.error(f"[DB] find_recent_session_by_face failed: {e}")
        return None


async def get_last_interaction(session_id: str):
    """
    Returns the single most recent stored {question, answer, timestamp}
    for a given session_id, or None if nothing is on file. Kept for the
    explicit "what was my last session about" recall route in main.py.
    Reads the LITERAL last question straight from Mongo (no LLM, no
    guessing) — either a real prior question exists or it doesn't.
    """
    if not session_id:
        return None
    try:
        return await interactions_collection.find_one(
            {"session_id": session_id},
            sort=[("timestamp", -1)],
        )
    except Exception as e:
        logger.error(f"[DB] get_last_interaction failed: {e}")
        return None


async def get_recent_interactions(session_id: str | None = None, face_id: str | None = None, limit: int = 3):
    """
    Returns up to `limit` most recent stored interactions, OLDEST first
    (chronological order) — used to summarize the *topics* of a visitor's
    previous session(s) for the "welcome back, last time you were asking
    about X" greeting, rather than just echoing their final message.

    Prefers `face_id` when given: it's the durable per-person identity,
    so it finds a visitor's real history even across any session_id
    fragmentation (e.g. a stray interaction that got logged under
    session_id="unknown" because no session was active yet at that
    moment — that Q&A is otherwise permanently orphaned from the
    session-based lookup). Falls back to session_id only if no face_id
    is available. Empty list if nothing is on file; never guesses.
    """
    if not face_id and not session_id:
        return []
    try:
        query = {"face_id": face_id} if face_id else {"session_id": session_id}
        cursor = interactions_collection.find(
            query,
            sort=[("timestamp", -1)],
            limit=limit,
        )
        docs = [doc async for doc in cursor]
        docs.reverse()  # oldest first, so the topic summary reads naturally
        return docs
    except Exception as e:
        logger.error(f"[DB] get_recent_interactions failed: {e}")
        return []


async def touch_session(session_id: str):
    """Bump last_activity so the 30-day window is measured from real use."""
    try:
        await sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"last_activity": datetime.now().isoformat()}},
        )
    except Exception as e:
        logger.error(f"[DB] touch_session failed: {e}")


async def deactivate_session(session_id: str):
    try:
        await sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"is_active": False,
                      "last_activity": datetime.now().isoformat()}},
        )
    except Exception as e:
        logger.error(f"[DB] deactivate_session failed: {e}")


async def update_session_context(session_id: str, entry: dict) -> None:
    """
    Session-context write-back (runs after EVERY terminal kiosk response).

    Appends {query, answer, route, entity, timestamp} to a bounded
    `context_history` list on the session document. This is what powers
    context-aware interaction on the NEXT turn: query condensing
    (backend.llm.condense_query) and pending-clarification resolution
    (backend.confidence_rag) both read the in-memory copy kept on
    active_session, but persisting it here means a resumed session (same
    face_id within 30 days) can also recover its last topic from Mongo,
    not just from the in-memory dict that gets wiped on server restart.
    Keeps only the most recent 10 turns — this is short-term conversational
    memory, not a permanent transcript (that's what `interactions` is for).
    """
    try:
        await sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "context_history": {
                        "$each": [entry],
                        "$slice": -10,
                    }
                },
                "$set": {"last_activity": datetime.now().isoformat()},
            },
        )
    except Exception as e:
        logger.error(f"[DB] update_session_context failed: {e}")


async def get_session_context(session_id: str) -> list[dict]:
    """Returns the stored context_history for a session (oldest first), or []."""
    try:
        doc = await sessions_collection.find_one(
            {"session_id": session_id}, {"context_history": 1}
        )
        return (doc or {}).get("context_history", []) or []
    except Exception as e:
        logger.error(f"[DB] get_session_context failed: {e}")
        return []


# ==========================================
# UNANSWERED-QUESTION TRACKING (LOW confidence, in-domain)
# ==========================================
def _normalize_for_dedup(text: str) -> str:
    """Lowercase + strip punctuation/extra whitespace so near-identical
    phrasings of the same unknown question collapse into one tracked row
    instead of spamming the admin queue with duplicates."""
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


async def save_unanswered_question(question: str, related: bool = True,
                                    session_id: str | None = None,
                                    face_id: str | None = None) -> str:
    """
    Logs (or bumps the ask-count of) a question the confidence-RAG pipeline
    could not answer. `related=True` means the LOW-confidence Scope Check
    judged it RNSIT-related (RNSIT_UNKNOWN in the routing diagram) — these
    are the ones that show up in the admin review queue. Out-of-scope
    questions are intentionally NEVER passed in here (see backend/
    confidence_rag.py's LOW branch) — they must never pollute the
    knowledge-update loop.

    Dedup key = normalized question text (hash), so "where's the hostel"
    and "Where is the hostel?" collapse into a single tracked row whose
    ask_count increments — this is what lets the admin dashboard surface
    "most frequently asked unknowns" for review priority.
    """
    norm = _normalize_for_dedup(question)
    if not norm:
        return ""
    qid = hashlib.sha256(norm.encode()).hexdigest()[:24]
    now = datetime.now().isoformat()
    try:
        await unanswered_collection.update_one(
            {"question_id": qid},
            {
                "$set": {
                    "question_text": question.strip(),
                    "normalized_text": norm,
                    "related": bool(related),
                    "last_asked_at": now,
                    "status": "pending",
                },
                "$setOnInsert": {
                    "question_id": qid,
                    "first_asked_at": now,
                },
                "$inc": {"ask_count": 1},
                "$addToSet": {
                    "session_ids": session_id or "unknown",
                },
            },
            upsert=True,
        )
        logger.info(f"[DB] Unanswered question tracked (related={related}): '{question}'")
        return qid
    except Exception as e:
        logger.error(f"[DB] save_unanswered_question failed: {e}")
        return ""


async def get_unanswered_questions(status: str = "pending", limit: int = 200) -> list[dict]:
    """Admin queue — sorted by ask_count desc so the most-frequently-asked
    unknowns (the ones worth an admin's time) surface first."""
    try:
        query = {} if status == "all" else {"status": status}
        cursor = unanswered_collection.find(query).sort(
            [("ask_count", -1), ("last_asked_at", -1)]
        ).limit(limit)
        docs = [d async for d in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs
    except Exception as e:
        logger.error(f"[DB] get_unanswered_questions failed: {e}")
        return []


async def mark_unanswered_resolved(question_id: str, knowledge_entry_id: str) -> bool:
    """Flips a tracked unknown question to 'answered' once the admin has
    verified and published a knowledge_entries answer for it."""
    try:
        res = await unanswered_collection.update_one(
            {"question_id": question_id},
            {"$set": {
                "status": "answered",
                "resolved_at": datetime.now().isoformat(),
                "knowledge_entry_id": knowledge_entry_id,
            }},
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error(f"[DB] mark_unanswered_resolved failed: {e}")
        return False


async def dismiss_unanswered_question(question_id: str) -> bool:
    """Admin dismisses an unknown question without answering it (e.g. it
    was noise, or out-of-scope but slipped through the keyword allowlist)."""
    try:
        res = await unanswered_collection.update_one(
            {"question_id": question_id},
            {"$set": {"status": "dismissed", "resolved_at": datetime.now().isoformat()}},
        )
        return res.matched_count > 0
    except Exception as e:
        logger.error(f"[DB] dismiss_unanswered_question failed: {e}")
        return False


# ==========================================
# ADMIN-VERIFIED KNOWLEDGE ENTRIES (self-updating RAG source of truth)
# ==========================================
async def save_knowledge_entry(question: str, answer: str, category: str = "General",
                                entity_type: str = "", entity_name: str = "",
                                entry_id: str | None = None,
                                created_by: str = "admin") -> dict:
    """
    Create OR update a verified knowledge entry. Reusing the SAME entry_id
    on update (rather than minting a new one) is what lets the RAGService
    upsert-by-entry replace the old vector chunk instead of leaving it
    alongside the new one — the actual mechanism behind "stale-data
    prevention" in the architecture: one entry_id <-> one authoritative
    set of chunks in ChromaDB, always.
    """
    now = datetime.now().isoformat()
    eid = entry_id or str(uuid.uuid4())
    doc = {
        "entry_id": eid,
        "question": question.strip(),
        "answer": answer.strip(),
        "category": category or "General",
        "entity_type": entity_type or "",
        "entity_name": entity_name or "",
        "updated_at": now,
        "updated_by": created_by,
    }
    try:
        await knowledge_entries_collection.update_one(
            {"entry_id": eid},
            {"$set": doc, "$setOnInsert": {"created_at": now, "created_by": created_by}},
            upsert=True,
        )
        logger.info(f"[DB] Knowledge entry upserted: entry_id={eid} question={question!r}")
        return doc
    except Exception as e:
        logger.error(f"[DB] save_knowledge_entry failed: {e}")
        raise


async def get_knowledge_entries(limit: int = 500) -> list[dict]:
    try:
        cursor = knowledge_entries_collection.find().sort("updated_at", -1).limit(limit)
        docs = [d async for d in cursor]
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs
    except Exception as e:
        logger.error(f"[DB] get_knowledge_entries failed: {e}")
        return []


async def get_knowledge_entry(entry_id: str) -> dict | None:
    try:
        doc = await knowledge_entries_collection.find_one({"entry_id": entry_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"[DB] get_knowledge_entry failed: {e}")
        return None


async def delete_knowledge_entry(entry_id: str) -> bool:
    try:
        res = await knowledge_entries_collection.delete_one({"entry_id": entry_id})
        return res.deleted_count > 0
    except Exception as e:
        logger.error(f"[DB] delete_knowledge_entry failed: {e}")
        return False


async def ensure_indexes():
    """Called once at startup — keeps face/session lookups fast."""
    try:
        await faces_collection.create_index("face_id", unique=True)
        await sessions_collection.create_index("session_id", unique=True)
        await sessions_collection.create_index([("face_id", 1), ("last_activity", -1)])
        await unanswered_collection.create_index("question_id", unique=True)
        await unanswered_collection.create_index([("status", 1), ("ask_count", -1)])
        await knowledge_entries_collection.create_index("entry_id", unique=True)
        logger.info("[DB] Indexes ensured.")
    except Exception as e:
        logger.warning(f"[DB] ensure_indexes: {e}")


# ==========================================
# CONFIGURABLE CONFIDENCE-RAG ROUTING THRESHOLDS
# ==========================================
# Precedence: value stored in MongoDB (admin-set via the dashboard) >
# env var > hardcoded default. This is what makes the HIGH/MEDIUM/LOW
# bands and the ambiguity-gap tunable WITHOUT a code change or restart —
# only backend/confidence_rag.py's short-lived in-memory cache needs to
# expire (or be explicitly invalidated right after a save, which the
# admin PUT endpoint does).
def _default_for(key: str) -> float:
    env_name = _RAG_SETTINGS_ENV_VARS[key]
    return float(os.getenv(env_name, _RAG_SETTINGS_HARDCODED_DEFAULTS[key]))


async def get_rag_settings() -> dict:
    """Returns ALL threshold keys, each resolved via the precedence above."""
    resolved = {key: _default_for(key) for key in _RAG_SETTINGS_HARDCODED_DEFAULTS}
    try:
        doc = await settings_collection.find_one({"_id": _RAG_SETTINGS_DOC_ID})
        if doc:
            for key in resolved:
                if doc.get(key) is not None:
                    resolved[key] = float(doc[key])
            resolved["updated_at"] = doc.get("updated_at")
            resolved["updated_by"] = doc.get("updated_by")
    except Exception as e:
        logger.error(f"[DB] get_rag_settings failed, using env/hardcoded defaults: {e}")
    return resolved


async def save_rag_settings(updates: dict, updated_by: str = "admin") -> dict:
    """Persists a PARTIAL update (only the keys the admin actually changed);
    unset keys keep falling back to env-var/hardcoded defaults via
    get_rag_settings()."""
    clean = {k: float(v) for k, v in updates.items() if k in _RAG_SETTINGS_HARDCODED_DEFAULTS and v is not None}
    if not clean:
        return await get_rag_settings()
    clean["updated_at"] = datetime.now().isoformat()
    clean["updated_by"] = updated_by
    try:
        await settings_collection.update_one(
            {"_id": _RAG_SETTINGS_DOC_ID}, {"$set": clean}, upsert=True
        )
    except Exception as e:
        logger.error(f"[DB] save_rag_settings failed: {e}")
        raise
    return await get_rag_settings()