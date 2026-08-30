import os
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


async def ensure_indexes():
    """Called once at startup — keeps face/session lookups fast."""
    try:
        await faces_collection.create_index("face_id", unique=True)
        await sessions_collection.create_index("session_id", unique=True)
        await sessions_collection.create_index([("face_id", 1), ("last_activity", -1)])
        logger.info("[DB] Indexes ensured.")
    except Exception as e:
        logger.warning(f"[DB] ensure_indexes: {e}")