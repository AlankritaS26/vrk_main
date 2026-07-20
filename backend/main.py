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
import secrets
from pathlib import Path
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports matching your async MongoDB database layout
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

# --- REDIS / MEMURAI CACHING ---
try:
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    logger.info("[REDIS] Connected to Memurai caching engine.")
except Exception as e:
    logger.warning("[REDIS] Memurai unreachable: %s", e)
    redis_client = None

# --- DIRECT MONGO CLIENT FOR ADMIN DASHBOARD ---
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.error("[SYSTEM] MONGO_URI environmental variable is missing!")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.rnsit_db if mongo_client else None

# --- SECURITY GATE CONFIGURATION ---
security = HTTPBasic()
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "111111"

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={},
        )
    return credentials.username

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

# 1. Read the comma-separated string from .env and split it into an actual list
origins_raw = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]

# 2. Add the middleware with the processed list
if not ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
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


# ==========================================================
# WRITE API SCHEMA AND ENDPOINTS FOR ADMINISTRATIVE WRITE CONTROLS
# ==========================================================
class FaceUpdateRequest(BaseModel):
    name: str

@app.delete("/api/admin/interactions/{session_id}")
async def delete_interaction(session_id: str, username: str = Depends(authenticate_admin)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection is inactive.")
    result = await db.interactions.delete_many({"session_id": session_id})
    return {"message": f"Purged {result.deleted_count} logs for session {session_id}."}

@app.delete("/api/admin/faces/{face_id}")
async def delete_face(face_id: str, username: str = Depends(authenticate_admin)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection is inactive.")
    # Target and eliminate facial record matching the ID
    result = await db.faces.delete_one({"face_id": face_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Face target profile not found.")
    
    # Cascade clean corresponding active session caches to stay structured
    await db.sessions.delete_many({"face_id": face_id})
    return {"message": "Facial record successfully deleted."}

@app.put("/api/admin/faces/{face_id}")
async def update_face_name(face_id: str, payload: FaceUpdateRequest, username: str = Depends(authenticate_admin)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection is inactive.")
    result = await db.faces.update_one(
        {"face_id": face_id},
        {"$set": {"name": payload.name}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Face record not found.")
    return {"message": f"Renamed profile to {payload.name}"}

@app.delete("/api/admin/sessions/{session_id}")
async def delete_session(session_id: str, username: str = Depends(authenticate_admin)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection is inactive.")
    result = await db.sessions.delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session record not found.")
    return {"message": f"Session {session_id} ended and dropped from database."}

@app.delete("/api/admin/clear-all")
async def clear_all_test_data(username: str = Depends(authenticate_admin)):
    """A master administrative trigger to drop interactions and transient sessions for clean demos."""
    global active_session, message_log, _last_activity_ts
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection is inactive.")
    
    await db.interactions.delete_many({})
    await db.sessions.delete_many({})
    
    # Flush current runtime values as well
    active_session = None
    message_log = []
    _last_activity_ts = 0.0
    
    await manager.broadcast({"type": "session_end", "session_id": "all", "reason": "admin_reset"})
    return {"message": "All database session registries and live interaction history wiped!"}


# ==========================================
# SECURE ADMIN DASHBOARD (MULTI-COLLECTION)
# ==========================================
@app.get("/logs-dashboard", response_class=HTMLResponse)
async def view_admin_dashboard(username: str = Depends(authenticate_admin)):
    if db is None:
        return HTMLResponse(
            content="<h2>Database configuration error. MONGO_URI is not set up correctly.</h2>",
            status_code=500
        )

    # 1. Pull data concurrently from all four MongoDB collections
    interactions_list = await db.interactions.find().sort("timestamp", -1).limit(50).to_list(length=50)
    faces_list = await db.faces.find().sort("detected_at", -1).limit(50).to_list(length=50)
    sessions_list = await db.sessions.find().sort("start_time", -1).limit(50).to_list(length=50)
    profile_list = await db.college_profile.find().limit(100).to_list(length=100)

    # --- Tab 1: Build Interactions rows ---
    interaction_rows = ""
    for idx, item in enumerate(interactions_list):
        ts = item.get("timestamp")
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts or "N/A")
        sess_id = item.get('session_id', 'N/A')
        interaction_rows += f"""
        <tr id="interaction-{sess_id}">
            <td>{idx + 1}</td>
            <td><code>{sess_id}</code></td>
            <td><strong>{item.get('input_text', 'N/A')}</strong></td>
            <td>{item.get('response_text', 'N/A')}</td>
            <td><span class="badge">{time_str}</span></td>
            <td>
                <button class="btn btn-danger" onclick="deleteInteraction('{sess_id}')">Delete Log</button>
            </td>
        </tr>
        """

    # --- Tab 2: Build Faces rows ---
    face_rows = ""
    for idx, item in enumerate(faces_list):
        ts = item.get("detected_at") or item.get("last_seen")
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts or "N/A")
        face_id = item.get('face_id', 'N/A')
        current_name = item.get('name', 'Unknown Visitor')
        face_rows += f"""
        <tr id="face-{face_id}">
            <td>{idx + 1}</td>
            <td><code>{face_id}</code></td>
            <td><strong id="face-name-text-{face_id}">{current_name}</strong></td>
            <td>{item.get('visit_count', 1)}</td>
            <td><span class="badge">{time_str}</span></td>
            <td>
                <button class="btn btn-edit" onclick="editFaceName('{face_id}', '{current_name}')">Rename</button>
                <button class="btn btn-danger" onclick="deleteFace('{face_id}')">Delete</button>
            </td>
        </tr>
        """

    # --- Tab 3: Build Sessions rows ---
    session_rows = ""
    for idx, item in enumerate(sessions_list):
        ts = item.get("start_time")
        time_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts or "N/A")
        sess_id = item.get('session_id', 'N/A')
        session_rows += f"""
        <tr id="session-{sess_id}">
            <td><code>{sess_id}</code></td>
            <td>{item.get('user_name', 'Guest')}</td>
            <td>{item.get('visit_count', 1)}</td>
            <td><span class="badge">{time_str}</span></td>
            <td>
                <button class="btn btn-danger" onclick="deleteSession('{sess_id}')">End & Delete</button>
            </td>
        </tr>
        """

    # --- Tab 4: Build Knowledge Base rows ---
    profile_rows = ""
    for idx, item in enumerate(profile_list):
        profile_rows += f"""
        <tr>
            <td>{idx + 1}</td>
            <td><span class="badge" style="background:#0066cc; color:white;">{item.get('category', 'General')}</span></td>
            <td><strong>{item.get('question_or_key', item.get('question', 'N/A'))}</strong></td>
            <td>{item.get('fact_details', item.get('answer', 'N/A'))}</td>
            <td><span class="badge" style="background:#e2e8f0; color:#475569;">Static</span></td>
        </tr>
        """

    # Modern responsive HTML layout with CSS-only tabs and interactive controls
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RNSIT Kiosk - Admin Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 30px; background-color: #f4f6f9; color: #333; }}
            .container {{ max-width: 1300px; margin: 0 auto; }}
            .header-flex {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
            h1 {{ color: #0066cc; margin: 0; font-size: 28px; }}
            .subtitle {{ color: #666; margin-top: 5px; margin-bottom: 30px; font-size: 15px; }}
            
            /* Clean CSS Tabs layout */
            .tabs {{ display: flex; flex-wrap: wrap; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            .tabs label {{ padding: 15px 25px; cursor: pointer; background: #f8fafc; font-weight: 600; border-bottom: 3px solid transparent; transition: ease 0.2s; order: 1; }}
            .tabs input[type="radio"] {{ display: none; }}
            .tab-content {{ width: 100%; padding: 25px; background: #fff; border-top: 1px solid #e2e8f0; display: none; order: 99; overflow-x: auto; }}
            
            .tabs input[type="radio"]:checked + label {{ border-bottom: 3px solid #0066cc; background: #fff; color: #0066cc; }}
            .tabs input[type="radio"]:checked + label + .tab-content {{ display: block; }}
            
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; min-width: 800px; }}
            th, td {{ padding: 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
            th {{ background-color: #f8fafc; color: #475569; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; font-weight: 700; }}
            tr:hover {{ background-color: #f8fafc; }}
            code {{ background: #f1f5f9; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-family: Courier, monospace; color: #0066cc; }}
            .badge {{ background: #e2e8f0; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
            
            /* Modernized Action Button Designs */
            .btn {{ padding: 6px 12px; border: none; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer; transition: 0.2s ease; margin-right: 5px; }}
            .btn-danger {{ background-color: #fee2e2; color: #dc2626; }}
            .btn-danger:hover {{ background-color: #fca5a5; }}
            .btn-edit {{ background-color: #e0f2fe; color: #0284c7; }}
            .btn-edit:hover {{ background-color: #bae6fd; }}
            .btn-reset {{ background-color: #dc2626; color: white; padding: 10px 20px; font-size: 14px; border-radius: 6px; box-shadow: 0 4px 6px rgba(220, 38, 38, 0.2); }}
            .btn-reset:hover {{ background-color: #b91c1c; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-flex">
                <div>
                    <h1>RNSIT Kiosk - Admin Dashboard</h1>
                    <p class="subtitle">Secure Read/Write control plane managing collections, tracked users, and kiosk interactions.</p>
                </div>
                <button class="btn btn-reset" onclick="clearAllTestData()">🔄 Clear All Sessions & Interactions</button>
            </div>
            
            <div class="tabs">
                <input type="radio" name="admin_tabs" id="tab_interactions" checked>
                <label for="tab_interactions"> Interactions ({len(interactions_list)})</label>
                <div class="tab-content">
                    <h3> Live Interactions Log (`interactions` collection)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 5%">#</th>
                                <th style="width: 15%">Session ID</th>
                                <th style="width: 25%">User Query</th>
                                <th style="width: 35%">Kiosk Response</th>
                                <th style="width: 12%">Timestamp</th>
                                <th style="width: 8%">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {interaction_rows if interaction_rows else "<tr><td colspan='6' style='text-align:center;'>No interactions recorded yet.</td></tr>"}
                        </tbody>
                    </table>
                </div>

                <input type="radio" name="admin_tabs" id="tab_faces">
                <label for="tab_faces"> Face Tracks ({len(faces_list)})</label>
                <div class="tab-content">
                    <h3> Registered Facial Profiles (`faces` collection)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 5%">#</th>
                                <th style="width: 25%">Face ID Token</th>
                                <th style="width: 25%">Identified Name</th>
                                <th style="width: 15%">Visit Count</th>
                                <th style="width: 18%">Last Spotted</th>
                                <th style="width: 12%">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {face_rows if face_rows else "<tr><td colspan='6' style='text-align:center;'>No facial profiles tracked yet.</td></tr>"}
                        </tbody>
                    </table>
                </div>

                <input type="radio" name="admin_tabs" id="tab_sessions">
                <label for="tab_sessions"> Active Sessions ({len(sessions_list)})</label>
                <div class="tab-content">
                    <h3>Session Registries (`sessions` collection)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 25%">Session ID</th>
                                <th style="width: 25%">Visitor Name</th>
                                <th style="width: 20%">Session Visit Count</th>
                                <th style="width: 18%">Started At</th>
                                <th style="width: 12%">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {session_rows if session_rows else "<tr><td colspan='5' style='text-align:center;'>No active sessions.</td></tr>"}
                        </tbody>
                    </table>
                </div>

                <input type="radio" name="admin_tabs" id="tab_profile">
                <label for="tab_profile"> Knowledge Base ({len(profile_list)})</label>
                <div class="tab-content">
                    <h3>College Profile Documents (`college_profile` collection)</h3>
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 5%">#</th>
                                <th style="width: 15%">Category</th>
                                <th style="width: 25%">Topic Key / FAQ Question</th>
                                <th style="width: 45%">Stored Fact Details / FAQ Answer</th>
                                <th style="width: 10%">Type</th>
                            </tr>
                        </thead>
                        <tbody>
                            {profile_rows if profile_rows else "<tr><td colspan='5' style='text-align:center;'>Knowledge base is empty.</td></tr>"}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // Helper configuration using Basic Authentication matching your credentials
            async function makeRequest(url, method, body = null) {{
                const headers = {{
                    "Content-Type": "application/json",
                    "Authorization": "Basic " + btoa("admin:111111")
                }};
                
                const config = {{ method, headers }};
                if (body) {{
                    config.body = JSON.stringify(body);
                }}

                try {{
                    const response = await fetch(url, config);
                    if (!response.ok) {{
                        const errorData = await response.json();
                        throw new Error(errorData.detail || "Server error occurred");
                    }}
                    return await response.json();
                }} catch (err) {{
                    alert("Operation failed: " + err.message);
                    return null;
                }}
            }}

            // Delete Interactions API link
            async function deleteInteraction(sessionId) {{
                if (confirm(`Do you want to purge interaction logs for session: ${{sessionId}}?`)) {{
                    const result = await makeRequest(`/api/admin/interactions/${{sessionId}}`, "DELETE");
                    if (result) {{
                        alert(result.message);
                        location.reload();
                    }}
                }}
            }}

            // Delete face targets and cleanup sessions
            async function deleteFace(faceId) {{
                if (confirm(`Are you sure you want to delete profile ${{faceId}}? This action resets their history.`)) {{
                    const result = await makeRequest(`/api/admin/faces/${{faceId}}`, "DELETE");
                    if (result) {{
                        alert(result.message);
                        document.getElementById(`face-${{faceId}}`)?.remove();
                    }}
                }}
            }}

            // Edit and update name parameters dynamically
            async function editFaceName(faceId, currentName) {{
                const newName = prompt(`Enter a new display name for ${{currentName}}:`, currentName);
                if (newName && newName.trim() !== "" && newName !== currentName) {{
                    const result = await makeRequest(`/api/admin/faces/${{faceId}}`, "PUT", {{ name: newName.trim() }});
                    if (result) {{
                        document.getElementById(`face-name-text-${{faceId}}`).textContent = newName.trim();
                    }}
                }}
            }}

            // Terminate ongoing sessions
            async function deleteSession(sessionId) {{
                if (confirm(`Terminate registry profile for session ID ${{sessionId}}?`)) {{
                    const result = await makeRequest(`/api/admin/sessions/${{sessionId}}`, "DELETE");
                    if (result) {{
                        alert("Session successfully dropped.");
                        document.getElementById(`session-${{sessionId}}`)?.remove();
                    }}
                }}
            }}

            // Global Master Reset
            async function clearAllTestData() {{
                if (confirm("MASTER DESTRUCTION WARNING: This clears all user sessions and live interaction records. Are you sure you want to clean up for a new presentation run?")) {{
                    const result = await makeRequest("/api/admin/clear-all", "DELETE");
                    if (result) {{
                        alert(result.message);
                        location.reload();
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
            "session":     active_session,
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
# =========================================
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

    # ─── External Team LLM RAG Pipeline Handoff ─────────────────────────────────
    logger.info("[CACHE MISS] Invoking external team's custom RAG pipeline for: '%s'", q_normalized)
    
    # Process memory bounds safely matching the structure they parse
    safe_history = message_log[-6:] if len(message_log) > 0 else []
    
    try:
        # Call the other team's function directly. It reads its own env vars, 
        # manages context from college_info.json, and contacts their LLM platform.
        answer = await generate_rag_kiosk_response(q_normalized, history=safe_history)
        
        # Cache the successful response back in Redis
        if redis_client and answer:
            try:
                redis_client.set(cache_key, answer, ex=3600)
            except Exception as e:
                logger.warning("Redis write error: %s", e)
                
    except Exception as exc:
        logger.error("External team's LLM engine failed or threw an exception: %s", exc)
        answer = "I'm having trouble processing that right now. Please visit the Admin Block for assistance."

    return await _respond(answer, source="external_team_llm")
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
class GreetVisitorPayload(BaseModel):
    face_id: str = Field("", description="Tracked face database identifier token")
    name: str = Field("Unknown", description="Identified display identity")
    is_returning: bool = Field(False, description="Flag identifying historic visitor profiling")
    visit_count: int = Field(1, description="Aggregated metric of check-ins")


@app.post("/visitor/greet")
async def greet_visitor(payload: GreetVisitorPayload):
    """
    Orchestration gateway intercepting hits from detection.py hardware loop.
    Routes tracked users directly into session pipelines or triggers identity checks.
    """
    global active_session, visitor_name_response, _last_activity_ts
    _last_activity_ts = datetime.now().timestamp()
    
    # Context handling for unrecognized/new visitors
    if not payload.face_id or payload.name.lower() == "unknown":
        logger.info("[GREET BLOCK] Unrecognized presence captured. Redirecting to initialization context.")
        if active_session is None:
            active_session = {
                "session_id": str(uuid.uuid4()),
                "user_name": "Unknown",
                "is_returning": False,
                "visit_count": 1,
                "face_id": "",
                "trigger": "camera",
                "asking_name": True,
            }
        else:
            active_session["asking_name"] = True
            
        await manager.broadcast({
            "type": "asking_name", 
            "session": active_session,
            "tts_text": "Hello! Welcome to RNSIT Kiosk. Please say your name, or say Guest to continue."
        })
        return {"status": "asking", "session_id": active_session["session_id"]}

    # Processing state metrics for verified returning users
    final_session_id = payload.face_id
    active_session = {
        "session_id": final_session_id,
        "user_name": payload.name,
        "is_returning": payload.is_returning,
        "visit_count": payload.visit_count,
        "face_id": payload.face_id,
        "trigger": "camera",
        "asking_name": False,
    }
    
    await manager.broadcast({
        "type": "session_start", 
        "session": active_session,
        "tts_text": f"Welcome back, {payload.name}! How can I help you today?"
    })
    
    logger.info(f"[GREET SUCCESS] Session established for user context: '{payload.name}'")
    return {"status": "recognized", "session_id": final_session_id, "session": active_session}


@app.post("/visitor/unknown")
async def visitor_unknown():
    """Fallback hook called by edge filters when handling non-registered footprints."""
    global visitor_name_response, active_session, _last_activity_ts
    _last_activity_ts = datetime.now().timestamp()
    
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

    await manager.broadcast({"type": "asking_name", "session": active_session})
    return {"status": "asking", "session_id": active_session["session_id"]}


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
        return {"success": True, "message": f"All data for '{name}' has been permanently deleted."}
    except Exception as e:
        logger.error(f"[DELETE] Error: {e}")
        return {"success": False, "message": "Deletion failed. Please contact staff."}