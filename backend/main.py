"""
RNSIT Digital Receptionist - Backend Server

HOW TO RUN (always from VRK_MVP/ folder):
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

import os
import uuid
import shutil
import logging
import hashlib
import string
import asyncio
import sys
import base64
from pathlib import Path
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Notice everything is fully async matching your MongoDB database.py layout
# Change lines 33-37 in backend/main.py to this exact block:
from backend.database import (
    get_kiosk_data,
    save_session, save_interaction,
    update_face_seen, save_face_encoding, get_all_face_encodings,
    delete_face_by_name,
)
from backend.llm import initialize_rag_knowledge_base, generate_rag_kiosk_response, close_llm_client
from backend.stt import transcribe_audio, transcribe_pcm
from backend.tts import text_to_speech

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RNSIT_Kiosk")

ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_QUERY_LENGTH: int = 300  # Lowered slightly to guard against buffer/token manipulation attacks
SESSION_TIMEOUT_SECONDS: int = 120

DOMAINS_CORRECTIONS = {
    "pricipal":  "principal",
    "prinsipal": "principal",
    "libary":    "library",
    "placment":  "placement",
    "fees":      "fee",
}

# Redis / Memurai caching
try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    logger.info("[REDIS] Connected to Memurai caching engine.")
except Exception as e:
    logger.warning("[REDIS] Memurai unreachable: %s", e)
    redis_client = None

# Shared state
active_session: dict | None = None
message_log: list[dict] = []
visitor_name_response: dict = {"ready": False, "name": "", "save": True}
_last_activity_ts: float = 0.0


# ==========================================
# BACKGROUND TIMEOUT LOOP
# ==========================================
async def _session_timeout_loop():
    global active_session, _last_activity_ts
    try:
        while True:
            await asyncio.sleep(30)
            if active_session and _last_activity_ts > 0:
                idle = datetime.now().timestamp() - _last_activity_ts
                if idle >= SESSION_TIMEOUT_SECONDS:
                    logger.info(f"[SESSION] Timeout after {idle:.0f}s idle — ending session")
                    sid = active_session.get("session_id")
                    active_session    = None
                    _last_activity_ts = 0.0
                    await manager.broadcast({"type": "session_end", "session_id": sid, "reason": "timeout"})
    except asyncio.CancelledError:
        logger.info("[SESSION] Session timeout background task loop stopped cleanly.")


# ==========================================
# SECURITY LAYER (CUSTOM INCOMING GUARDRAIL)
# ==========================================
def verify_input_safety(query: str) -> bool:
    """
    A backend validation method intercepting malicious overrides or system prompt exploitation
    before passing query contexts down to any intelligence framework modules.
    """
    if len(query) > MAX_QUERY_LENGTH:
        return False
        
    # Block structural context injection strings
    malicious_sequences = [
        "ignore previous", "system prompt", "override rules", 
        "act as a", "you are now", "delete from", "drop collection"
    ]
    
    normalized_q = query.lower()
    if any(sequence in normalized_q for sequence in malicious_sequences):
        logger.warning(f"[SECURITY ALERT] Prompt injection signature intercepted: '{query}'")
        return False
        
    return True


# ==========================================
# SERVER LIFECYCLE
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[SYSTEM] Booting server resources...")
    try:
        await initialize_rag_knowledge_base()
        logger.info("[SYSTEM] RAG vector cache loaded successfully.")
    except Exception as e:
        logger.error("[SYSTEM] RAG initialization failed during startup: %s", e)

    timeout_task = asyncio.create_task(_session_timeout_loop())

    yield

    logger.info("[SYSTEM] Triggering cleanup hooks...")
    timeout_task.cancel()
    try:
        await timeout_task
    except asyncio.CancelledError:
        pass

    await close_llm_client()
    logger.info("[SYSTEM] Server teardown complete.")


app = FastAPI(title="RNSIT Digital Receptionist", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# WEBSOCKET BROADCASTER
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in self.active[:]:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ==========================================
# UTILITY HELPERS
# ==========================================
def _log_message(text: str, speaker: str) -> dict:
    entry = {
        "index":     len(message_log),
        "text":      text,
        "speaker":   speaker,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    message_log.append(entry)
    return entry


# ==========================================
# HEALTH ENDPOINT
# ==========================================
@app.get("/health")
def health():
    """Liveness probe for the launcher and future monitoring."""
    return {"status": "healthy"}


@app.get("/")
def root():
    return {"status": "RNSIT Kiosk Backend is Live"}


# ==========================================
# SESSION MANAGEMENT ENDPOINTS
# ==========================================
@app.post("/session/start")
async def start_session(
    trigger: str = "camera",
    user_name: str = "Guest",
    is_returning: bool = False,
    visit_count: int = 1,
    face_id: str = "",
    session_id: str = "",
):
    global active_session, message_log, _last_activity_ts
    final_session_id = face_id or session_id or str(uuid.uuid4())

    if active_session and active_session.get("session_id") == final_session_id:
        _last_activity_ts = datetime.now().timestamp()
        return {
            "status":     "already_active",
            "session_id": final_session_id,
            "session":    active_session,
        }

    active_session = {
        "session_id":   final_session_id,
        "user_name":    user_name,
        "is_returning": is_returning,
        "visit_count":  visit_count,
        "face_id":      face_id,
        "trigger":      trigger,
        "asking_name":  False,
    }
    message_log       = []
    _last_activity_ts = datetime.now().timestamp()

    db_face_id = face_id.strip() if face_id and face_id.strip() else None
    
    # Executed asynchronously matching your new async MongoDB driver configuration
    await save_session(final_session_id, db_face_id, user_name, is_returning, visit_count)
    await manager.broadcast({"type": "session_start", "session": active_session})
    return {"status": "success", "session_id": final_session_id, "session": active_session}


@app.post("/session/end")
async def end_session_endpoint(session_id: str = None):
    global active_session, _last_activity_ts
    sid = session_id or (active_session["session_id"] if active_session else None)
    
    active_session    = None
    _last_activity_ts = 0.0
    await manager.broadcast({"type": "session_end", "session_id": sid})
    return {"status": "success"}


@app.get("/session/current")
def get_current_session():
    if active_session:
        return {"active": True, **active_session}
    return {"active": False}


@app.get("/session/messages/{session_id}")
def get_session_messages(session_id: str, after: int = 0):
    msgs = [m for m in message_log if m.get("index", 0) > after]
    return {"messages": msgs}


# ==========================================
# MESSAGE ROUTING
# ==========================================
class MessagePayload(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    text: str = Field(..., description="Message text")
    speaker: str = Field(..., description="Speaker id/name")

    @field_validator("text")
    @classmethod
    def _validate_text(cls, v: str) -> str:
        v = v.strip()
        return v[:MAX_QUERY_LENGTH] if len(v) > MAX_QUERY_LENGTH else v


@app.post("/message")
async def post_message(payload: MessagePayload):
    entry = _log_message(payload.text, payload.speaker)
    await manager.broadcast({"type": "message", **entry})
    return {"status": "ok"}


# ==========================================
# CORE WORKFLOW ROUTING ENGINE (ASK)
# ==========================================
@app.get("/ask")
async def ask_kiosk(question: str = Query(..., description="Visitor question")):
    global _last_activity_ts, active_session
    _last_activity_ts = datetime.now().timestamp()

    # Apply the backend security guardrail validation immediately
    if not verify_input_safety(question):
        raise HTTPException(status_code=400, detail="Security Exception: Request contains blocked sequences.")

    q_clean = question.lower().strip()
    q_clean = q_clean.translate(str.maketrans('', '', string.punctuation)).strip()

    words           = q_clean.split()
    corrected_words = [DOMAINS_CORRECTIONS.get(w, w) for w in words]
    q_normalized    = " ".join(corrected_words)

    sid          = active_session["session_id"] if active_session else "unknown"
    visitor_name = (active_session.get("user_name") or "there") if active_session else "there"

    visitor_entry = _log_message(question, "visitor")
    await manager.broadcast({"type": "message", **visitor_entry})

    async def _respond(answer: str, source: str = "") -> dict:
        try:
            await save_interaction(sid, question, answer)
        except Exception as exc:
            logger.error("[DATABASE ERROR] Failed to log interaction: %s", exc)
        kiosk_entry = _log_message(answer, "kiosk")
        await manager.broadcast({"type": "message", **kiosk_entry})
        result = {"question": question, "answer": answer}
        if source:
            result["source"] = source
        return result

    # ─── Thank you → end session immediately ─────────────────────────────────
    THANK_YOU_PHRASES = {
        "thank you", "thanks", "thank u", "thankyou",
        "ok thanks", "okay thanks", "ok thank you", "okay thank you",
        "thats all", "thats all thanks", "bye", "goodbye", "that is all",
    }
    if any(phrase in q_normalized for phrase in THANK_YOU_PHRASES):
        farewell = (
            f"You're welcome{', ' + visitor_name if visitor_name != 'there' else ''}! "
            "Have a great day. Goodbye!"
        )
        active_session    = None
        _last_activity_ts = 0.0
        await manager.broadcast({
            "type":       "session_end",
            "session_id": sid,
            "reason":     "thank_you",
        })
        return await _respond(farewell)

    # ─── Dynamic Cloud MongoDB Lookup ─────────────────────────────────────────
    # Pull fresh structured details straight from MongoDB Compass/Atlas instead of local SQL scripts
    kiosk_cloud_data = await get_kiosk_data()
    if kiosk_cloud_data and "faqs" in kiosk_cloud_data:
        for faq in kiosk_cloud_data["faqs"]:
            if faq["question"].lower() in q_normalized:
                return await _respond(faq["answer"], source="mongodb_atlas")

    # ─── Redis cache fallback ──────────────────────────────────────────────────
    cache_key = f"kiosk:cache:{hashlib.md5(q_normalized.encode()).hexdigest()}"
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info("[REDIS HIT] For normalized key: '%s'", q_normalized)
                return await _respond(cached, source="redis_cache")
        except Exception as e:
            logger.warning("Redis read error: %s", e)

    # ─── RAG Intelligent GenAI Fallback ─────────────────────────────────────────
    logger.info("[CACHE MISS] Running RAG for processed string: '%s'", q_normalized)
    try:
        answer = await generate_rag_kiosk_response(q_normalized, history=message_log[:-1][-6:])
        if redis_client and answer:
            try:
                redis_client.set(cache_key, answer, ex=3600)
            except Exception as e:
                logger.warning("Redis write error: %s", e)
    except Exception as exc:
        logger.error("RAG inference failed: %s", exc)
        answer = "I'm having trouble processing that. Please visit the Admin Block for assistance."

    return await _respond(answer, source="local_llm")


# ==========================================
# BIOMETRICS / FACE REGISTRATION ENDPOINTS
# ==========================================
class RegisterFacePayload(BaseModel):
    face_id:  str         = Field(..., description="Unique face id")
    name:     str         = Field(..., description="Person's name")
    encoding: List[float] = Field(..., description="Face encoding vector")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip()


@app.get("/faces/all")
async def get_all_faces_endpoint():
    faces = await get_all_face_encodings()
    logger.info(f"[FACE] /faces/all returning {len(faces)} faces from MongoDB")
    return {"faces": faces}


@app.post("/faces/register")
async def register_face(payload: RegisterFacePayload):
    await save_face_encoding(payload.face_id, payload.name, payload.encoding)
    logger.info(f"[FACE] Registered in MongoDB: {payload.name} ({payload.face_id})")
    return {"status": "ok", "face_id": payload.face_id}


@app.post("/faces/visit")
async def record_face_visit(face_id: str):
    await update_face_seen(face_id)
    logger.info(f"[FACE] Visit count updated for face_id={face_id}")
    return {"status": "ok"}

# ==========================================
# SPEECH ENDPOINTS (STT / TTS)
# ==========================================
@app.post("/stt/pcm")
async def speech_to_text_pcm(request: Request):
    """
    Primary STT path: raw 16 kHz mono int16 PCM from the browser VAD.
    No WebM, no ffmpeg — bytes go straight into numpy → Whisper.
    """
    try:
        pcm_bytes = await request.body()
        if not pcm_bytes or len(pcm_bytes) < 4800:  # < 150 ms of audio
            return {"text": "", "confidence": 0.0, "error": "no_audio"}
        # Whisper is blocking — run in a thread so the event loop never stalls
        result = await asyncio.to_thread(transcribe_pcm, pcm_bytes, "en")
        return result
    except Exception as e:
        logger.error(f"[STT/PCM] Endpoint error: {e}")
        return {"text": "", "confidence": 0.0, "error": str(e)}


@app.websocket("/ws/stt")
async def stt_websocket_endpoint(ws: WebSocket):
    """
    Alternative STT transport:
      browser -> binary frame : one COMPLETE utterance (16 kHz mono int16 PCM)
      backend -> JSON frame   : {text, confidence, language, latency_ms}
    """
    await ws.accept()
    logger.info("[WS/STT] Kiosk connected")
    try:
        while True:
            pcm_bytes = await ws.receive_bytes()
            result = await asyncio.to_thread(transcribe_pcm, pcm_bytes, "en")
            await ws.send_json(result)
    except WebSocketDisconnect:
        logger.info("[WS/STT] Kiosk disconnected")


@app.post("/tts")
async def tts_endpoint(request: Request):
    """Text → base64 WAV (Kokoro). Empty audio → frontend falls back to browser voice."""
    try:
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text:
            return {"audio": None}
        wav = await asyncio.to_thread(text_to_speech, text)
        return {"audio": base64.b64encode(wav).decode("utf-8") if wav else None}
    except Exception as e:
        logger.error(f"[TTS] Endpoint error: {e}")
        return {"audio": None}


# ==========================================
# VISITOR MANAGEMENT ENDPOINTS
# ==========================================
@app.post("/visitor/unknown")
async def visitor_unknown():
    """Called by detection.py when an unrecognized face appears."""
    global visitor_name_response, active_session
    if not visitor_name_response.get("ready"):
        visitor_name_response = {"ready": False, "name": "", "save": True}

    if active_session is None:
        active_session = {
            "session_id":   str(uuid.uuid4()),
            "user_name":    "Unknown",
            "is_returning": False,
            "visit_count":  1,
            "face_id":      "",
            "trigger":      "camera",
            "asking_name":  True,
        }
    else:
        active_session["asking_name"] = True

    await manager.broadcast({"type": "asking_name"})
    return {"status": "asking"}


@app.post("/visitor/submit_name")
async def submit_name(name: str = "Guest", save: bool = True):
    global visitor_name_response, active_session
    visitor_name_response = {"ready": True, "name": name, "save": save}
    if active_session:
        active_session["asking_name"] = False
        active_session["user_name"]   = name
    logger.info(f"[VISITOR] Name submitted: '{name}' save={save}")
    return {"status": "ok"}


@app.get("/visitor/name_response")
def get_name_response():
    return visitor_name_response


@app.post("/visitor/clear_response")
def clear_response():
    global visitor_name_response
    visitor_name_response = {"ready": False, "name": "", "save": True}
    return {"status": "cleared"}


@app.post("/visitor/delete_my_data")
async def delete_my_data(name: str):
    """Erase a visitor's face data (GDPR-style right to be forgotten)."""
    try:
        face_ids = await delete_face_by_name(name)
        if not face_ids:
            return {"success": False, "message": f"No data found for '{name}'."}

        for face_id in face_ids:
            face_dir = PROJECT_ROOT / "faces" / face_id
            if face_dir.exists():
                shutil.rmtree(face_dir)
                logger.info(f"[DELETE] Removed face dir: {face_dir}")

        await manager.broadcast({"type": "cache_reload"})
        return {"success": True,
                "message": f"All data for '{name}' has been permanently deleted."}
    except Exception as e:
        logger.error(f"[DELETE] Error: {e}")
        return {"success": False, "message": "Deletion failed. Please contact staff."}