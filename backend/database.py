import os
import logging
from datetime import datetime
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

async def save_session(session_id: str, face_id: str | None, user_name: str, is_returning: bool, visit_count: int):
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
                    "last_activity": datetime.now().isoformat()
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

async def save_interaction(session_id: str, question: str, answer: str):
    """Logs individual conversational components directly into cloud transactions."""
    try:
        await interactions_collection.insert_one({
            "session_id": session_id,
            "input_text": question,
            "response_text": answer,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error logging conversational interaction: {e}")

# ==========================================
# BIOMETRICS & FACE RETRIEVAL HANDLERS
# ==========================================

async def save_face_encoding(face_id: str, name: str, encoding: list):
    """Saves or updates a biometric profile mapping vector representations directly."""
    try:
        await faces_collection.update_one(
            {"face_id": face_id},
            {
                "$set": {
                    "name": name,
                    "encoding": encoding,
                    "last_seen": datetime.now().isoformat()
                },
                "$setOnInsert": {
                    "registered_at": datetime.now().isoformat(),
                    "visit_count": 1
                }
            },
            upsert=True
        )
        logger.info(f"[MongoDB] Face Registered: {name}")
    except Exception as e:
        logger.error(f"Error updating biometric vector signature: {e}")

async def get_all_face_encodings():
    """Retrieves all registered biometric keys for local processing frames."""
    try:
        cursor = faces_collection.find({"encoding": {"$ne": None}})
        results = []
        async for doc in cursor:
            results.append({
                "face_id": doc["face_id"],
                "name": doc["name"],
                "encoding": doc["encoding"]
            })
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
