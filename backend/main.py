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
import re
import asyncio
import sys
import base64
import secrets
import json
from pathlib import Path
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import httpx

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── BUG FIX: set BEFORE detection import so detection.py captures the right URL ──
os.environ["BACKEND_URL"] = "http://127.0.0.1:8001"

# Imports matching your async MongoDB database layout
from backend.database import (
    get_kiosk_data,
    save_session, save_interaction, get_last_interaction, get_recent_interactions,
    update_face_seen, save_face_encoding, get_all_face_encodings,
    delete_face_by_name,
    find_recent_session_by_face, touch_session, deactivate_session, ensure_indexes,
)
from backend.llm import (
    initialize_rag_knowledge_base, close_llm_client, generate_rag_kiosk_response,
    generate_rag_kiosk_response_stream,
    extract_topic_label,
)
from backend.stt import transcribe_audio, transcribe_pcm
from backend.tts import text_to_speech

try:
    from backend.detection import run_pipeline as _run_pipeline
    _detection_available = True
except Exception as _det_import_err:
    import logging as _lg
    _lg.getLogger("RNSIT_Kiosk").warning(
        "[SYSTEM] Face detection module unavailable: %s  "
        "— install libgles2 (sudo apt-get install libgles2) and restart.",
        _det_import_err,
    )
    _run_pipeline = None
    _detection_available = False


def run_pipeline(frame_data):
    """Thin wrapper so the WebSocket handler always has a callable."""
    if _run_pipeline is None:
        class _NoOp:
            present = False; state = "IDLE"; identity = ""
            verified = False; bbox = None; bystanders = 0
            error = "detection_unavailable"; blink = False
        return _NoOp()
    return _run_pipeline(frame_data)

os.environ.setdefault("BACKEND_URL", "http://127.0.0.1:8001")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RNSIT_Kiosk")

MAX_QUERY_LENGTH: int = 300 
SESSION_TIMEOUT_SECONDS: int = 120

# ── RAG MICROSERVICE CONFIG ────────────────────────────────────────────────
# Read once, here, near the top of the file — everything else in this module
# (the /ask endpoint, the /api/chat endpoint, and the admin RAG management
# endpoints) reuses these two constants instead of re-reading os.environ.
RAG_SERVICE_URL: str   = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8600").rstrip("/")
RAG_COLLECTION:  str   = os.getenv("RAG_COLLECTION", "kiosk-rnsit")
RAG_TOP_K:       int   = int(os.getenv("RAG_TOP_K", "5"))
RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.35"))

DOMAINS_CORRECTIONS = {
    "pricipal":  "principal",
    "prinsipal": "principal",
    "libary":    "library",
    "placment":  "placement",
    "fees":      "fee",
    # Common STT mis-hearings of "RNSIT" (the college's own name!) — these
    # were silently NOT being fixed before: see the note further down
    # where q_normalized is built vs. what actually got sent to RAG.
    "rnsfit":    "rnsit",
    "ransit":    "rnsit",
    "rnscit":    "rnsit",
    "arnsit":    "rnsit",
    "rnsit's":   "rnsit",
    "rnsits":    "rnsit",
}

# STT sometimes splits "RNSIT" across multiple tokens instead of mishearing
# it as one word (e.g. "R N S fit", "run sit") — those can't be fixed by a
# single-word dict lookup, so they're corrected as whole phrases BEFORE the
# text is split into words.
PHRASE_CORRECTIONS = {
    "rns fit":     "rnsit",
    "r n s fit":   "rnsit",
    "run sit":     "rnsit",
    "rn sit":      "rnsit",
    "r and s fit": "rnsit",
    "r n site":    "rnsit",
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
# RAG MICROSERVICE CLIENT
# ==========================================
_RAG_STOPWORDS = {
    "who", "is", "the", "of", "a", "an", "what", "where", "when", "how", "why",
    "please", "tell", "me", "can", "you", "are", "in", "for", "to", "and", "or",
    "do", "does", "it", "was", "were", "be", "this", "that", "rnsit", "rns",
    "institute", "technology", "about",
}


_RAG_ABBREVS = {"dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "no", "vs", "etc"}


def _matches_short_phrase(q_normalized: str, phrases: set[str]) -> bool:
    """
    Word-boundary phrase match, deliberately restricted to SHORT utterances
    (<=4 words). This is the fix for the "random Goodbye" bug: the old code
    did `phrase in q_normalized`, a raw substring check across the entire
    sentence with no length cap — so a real question that happened to
    contain "bye"/"thanks" anywhere (or just drifted semantically close via
    RAG) could trigger a farewell. A genuine "bye"/"thank you" utterance is
    always short; a real campus question with those letters buried in it
    (or a coincidental partial match) is not, so capping length here is a
    cheap, robust way to tell the two apart without a big keyword tree.
    """
    q = q_normalized.strip()
    if not q or len(q.split()) > 4:
        return False
    if q in phrases:
        return True
    for phrase in phrases:
        if re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", q):
            return True
    return False


GREETING_PHRASES = {
    "hi", "hello", "hey", "hiya", "yo",
    "good morning", "good afternoon", "good evening",
}
_GREETING_RESPONSES = [
    "Hello! Welcome to RNSIT. How can I help you today?",
    "Hi there! I'm the RNSIT digital receptionist — what would you like to know?",
    "Hey! Welcome to RNS Institute of Technology. What can I help you with?",
]

# ── Easter eggs: the handful of off-topic, personality questions every
# visitor asks an assistant sooner or later ("are you a robot?", "tell me
# a joke"). This is the single thing visitors actually remember and tell
# their friends about, so these get warm, self-aware, instant replies
# instead of falling through to RAG (which has no campus-fact grounding
# for them and would either hallucinate or bounce to the offtopic fallback).
# Matched the same way as GREETING_PHRASES — via _matches_short_phrase, so
# only short, standalone utterances trigger this, never a real question
# that happens to share a few words with one of these keys.
EASTER_EGGS = {
    "are you a robot": [
        "I'm a digital receptionist, so yes and no — no body, but I do the job!",
        "Guilty as charged! But I promise I'm a friendly one.",
    ],
    "are you real": [
        "Real enough to help you find the CSE block! I'm Nova, RNSIT's digital receptionist.",
    ],
    "are you human": [
        "Not quite — I'm Nova, a digital receptionist. But I'll do my best to sound like one!",
    ],
    "tell me a joke": [
        "Why did the student bring a ladder to class? To reach the higher studies!",
        "What did the router say to the CSE student? Nothing, they just had a falling out over connection issues.",
    ],
    "who made you": [
        "I was built by the students of RNSIT to help visitors like you find your way around!",
    ],
    "what is your name": [
        "I'm Nova, the digital receptionist here at RNSIT. Nice to meet you!",
    ],
    "who are you": [
        "I'm Nova — think of me as RNSIT's always-awake front desk.",
    ],
    "i love you": [
        "That's sweet! I love helping visitors find their way around RNSIT too.",
    ],
    "do you sleep": [
        "Never! I'm here whenever a visitor needs help, day or night.",
    ],
    # ── Interactive/happy-moment additions ────────────────────────────────
    # Small talk that makes Nova feel like a person at the desk rather than
    # a search box, without drifting away from the college-assistant role —
    # deliberately short, warm, and quick to hand the conversation back to
    # campus topics.
    "how are you": [
        "I'm doing great, thanks for asking! Ready to help you explore RNSIT — what can I do for you?",
        "Feeling good and fully charged! What would you like to know about RNSIT?",
    ],
    "what is the weather today": [
        "I don't have a window, so I can't check the sky myself! But whatever it's like out there, I hope it's a good day for a campus visit.",
    ],
    "how is the weather": [
        "I don't have a window, so I can't check the sky myself! But whatever it's like out there, I hope it's a good day for a campus visit.",
    ],
    "good job": [
        "Aw, thank you! That made my day. Anything else I can help you with?",
    ],
    "you are smart": [
        "That's very kind of you to say! I try my best. What else can I help you with?",
    ],
    "you are awesome": [
        "You're pretty awesome yourself for saying that! What can I help you with next?",
    ],
    "nice to meet you": [
        "Nice to meet you too! I'm Nova, RNSIT's digital receptionist. How can I help you today?",
    ],
    "good night": [
        "Good night! It was lovely chatting with you — take care.",
    ],
}


# ── Replies to the "continue with X, or something else?" re-engagement ──
# These only carry meaning right after that specific greeting question,
# so they're intercepted deterministically (see the awaiting_topic_choice
# check in /ask) rather than being sent to RAG, where a bare "something
# else" was scoring a coincidental similarity hit and coming back as
# "I don't have that detail."
TOPIC_DECLINE_PHRASES = {
    "something else", "no", "nah", "not that", "different",
    "something different", "new topic", "no thanks", "not really",
}
TOPIC_CONTINUE_PHRASES = {
    "yes", "yeah", "yep", "sure", "continue", "yes please",
    "that one", "ok continue", "please continue", "continue with that",
}

# ── Farewell markers ──────────────────────────────────────────────────────
# THANK_YOU_PHRASES used to be declared inline inside the /ask endpoint;
# moved up here (module scope) so _is_farewell (also module scope) can see
# it, and so it sits alongside the other deterministic-route phrase sets.
THANK_YOU_PHRASES = {
    "thank you", "thanks", "thank u", "thankyou",
    "ok thanks", "okay thanks", "ok thank you", "okay thank you",
    "thats all", "thats all thanks", "bye", "goodbye", "that is all",
}

# THANK_YOU_PHRASES (below, used with the strict <=4-word _matches_short_
# phrase check) only ever caught bare "bye"/"thank you"-style utterances.
# Real visitors close a conversation in much longer, more natural ways —
# "Okay, nice talking to you.", "It was really nice talking to you. We'll
# meet you next time.", "That's it." — none of which matched, so those
# sessions never ended: the visitor's sign-off got treated as a fresh
# question, routed all the way through RAG/LLM (slow, and often answered
# with an irrelevant "I don't have that detail"), and the kiosk just sat
# there still "listening" instead of closing out.
#
# FAREWELL_MARKERS below catches those natural sign-offs as substrings
# (word-boundary matched) without the 4-word cap, since a closing remark is
# reliably short and distinctive even when the whole sentence isn't. It's
# intentionally still a fixed marker list (not "contains bye anywhere") so
# a genuine campus question is never misrouted.
FAREWELL_MARKERS = {
    "nice talking", "nice chatting", "great talking", "great chatting",
    "lovely talking", "meet you next time", "see you next time",
    "see you later", "see you soon", "catch you later", "catch you next time",
    "thats it", "that's it", "thats all", "that's all",
    "no more questions", "nothing else", "no other questions",
    "im done", "i am done", "im good", "im all set", "all set thanks",
    "gotta go", "got to go", "have to go", "need to go", "i should go",
    "ok bye", "okay bye", "alright bye", "bye bye", "gtg",
}


def _is_farewell(q_normalized: str) -> bool:
    """
    True for both the strict short farewells in THANK_YOU_PHRASES and the
    longer, natural sign-off phrasings in FAREWELL_MARKERS. See the
    FAREWELL_MARKERS comment above for why the two need different length
    rules. Capped at 15 words even for markers so a long, unrelated
    question that happens to contain one of these short phrases deep
    inside it doesn't get misrouted as a goodbye.
    """
    q = (q_normalized or "").strip()
    if not q:
        return False
    if _matches_short_phrase(q, THANK_YOU_PHRASES):
        return True
    if len(q.split()) > 15:
        return False
    for marker in FAREWELL_MARKERS:
        if re.search(rf"(?:^|\s){re.escape(marker)}(?:$|\s)", q):
            return True
    return False


# ── Q/A label stripper ────────────────────────────────────────────────────
# FAQ-derived chunks are seeded (see backend/llm.py::_json_to_text_chunks)
# as literal "Q: <question>? A: <answer>" strings, on purpose — that extra
# question text helps the embedding model match visitor phrasing. But those
# raw "Q:"/"A:" labels — and the question text itself — must never reach the
# visitor. _QA_LABEL_RE strips a leading "Q: ...? A: " block; _STRAY_LABEL_RE
# mops up any leftover "Q:"/"A:" markers (covers chunks with multiple
# Q/A pairs bundled together, or any other future FAQ-shaped content).
_QA_LABEL_RE = re.compile(r"Q:\s*.+?\?\s*A:\s*", re.IGNORECASE)
_STRAY_LABEL_RE = re.compile(r"\b[QA]:\s*", re.IGNORECASE)


def _strip_qa_labels(text: str) -> str:
    """Remove 'Q: ... A: ...' scaffolding from a chunk, leaving just the answer text."""
    text = _QA_LABEL_RE.sub("", text)
    text = _STRAY_LABEL_RE.sub("", text)
    return text.strip()


def _split_into_facts(text: str) -> list[str]:
    """
    Splits a chunk into sentence-like fragments without cutting titles and
    initials in half (e.g. "Dr. M K Venkatesha" would otherwise get chopped
    into "Dr." + "M K Venkatesha" by a naive '. ' split).
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    merged, buffer = [], ""
    for frag in raw:
        buffer = (buffer + " " + frag).strip() if buffer else frag
        tokens = buffer.split()
        last = tokens[-1] if tokens else ""
        is_initial = bool(re.fullmatch(r"[A-Z]\.", last))
        is_abbrev = last.rstrip(".").lower() in _RAG_ABBREVS
        if is_initial or is_abbrev:
            continue  # keep buffering — this period wasn't a real sentence end
        merged.append(buffer)
        buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def _extract_relevant_sentences(text: str, query: str, max_sentences: int = 2) -> tuple[str, bool]:
    """
    RAGService's chunks sometimes bundle many unrelated facts into one
    paragraph (e.g. a whole "college facts" block containing the address,
    director, principal, admissions phone, etc. all together). Instead of
    handing the entire chunk back to the visitor for every question that
    happens to match it, pull out just the fact(s) that actually contain
    the question's keywords.

    Returns (text, matched) — matched=True means we found keyword overlap
    and text is the focused extract; matched=False means nothing in this
    chunk matched and text is the original, unmodified chunk (the caller
    decides whether an unmatched chunk is even worth including at all).
    """
    keywords = {w for w in re.findall(r"[a-z0-9]+", query.lower())
                if w not in _RAG_STOPWORDS and len(w) > 2}
    if not keywords:
        return text, False

    fragments = _split_into_facts(text)
    scored = []
    for frag in fragments:
        frag_lower = frag.lower()
        hits = sum(1 for kw in keywords if kw in frag_lower)
        if hits > 0:
            scored.append((hits, frag.strip()))

    if not scored:
        return text, False

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [frag for _, frag in scored[:max_sentences]]
    return " ".join(best), True


async def query_rag_service(query: str, k: int | None = None) -> str:
    """
    Calls the standalone RAGService microservice (RAG_SERVICE_URL, default
    http://127.0.0.1:8600) to semantically search the knowledge base and
    returns a natural-language answer built from the best-matching chunk(s).

    RAG_TOP_K controls how many candidates we ask for; RAG_SIMILARITY_THRESHOLD
    filters out weak matches (RAGService's score = 1 - vector distance, so
    higher is better — 0.35 is a reasonable "actually related" cutoff).

    NOTE: as of the Phase 3 fix, the primary voice pipeline (/ask) no longer
    calls this — it calls backend.llm.generate_rag_kiosk_response, which
    retrieves the same way but then passes the context to the LOCAL LLM
    for a real generated answer instead of returning raw retrieved text.
    This function is kept for /api/chat, a lower-level diagnostic endpoint
    useful for inspecting exactly what RAGService itself returns.
    """
    k = k or RAG_TOP_K
    try:
        # Timeout is generous (60s) because RAGService downloads/loads its
        # embedding model lazily on its very first search request ever —
        # after that first warm-up it responds in well under a second.
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{RAG_SERVICE_URL}/v1/collections/{RAG_COLLECTION}/search",
                json={"query": query, "k": k},
            )
        resp.raise_for_status()
        results = resp.json()
    except httpx.RequestError as exc:
        logger.error("[RAG] RAGService unreachable at %s: %s", RAG_SERVICE_URL, exc)
        return "I'm having trouble reaching the knowledge base right now. Please visit the Admin Block for assistance."
    except httpx.HTTPStatusError as exc:
        logger.error("[RAG] RAGService returned an error: %s", exc.response.text)
        return "I couldn't find an answer to that just now. Could you rephrase your question?"
    except Exception as exc:
        logger.error("[RAG] Unexpected error querying RAGService: %s", exc)
        return "I'm having trouble processing that right now. Please visit the Admin Block for assistance."

    # Drop weak matches so an unrelated question doesn't get a confident-sounding
    # but wrong answer stitched together from irrelevant chunks.
    strong_results = [r for r in results if r.get("score", 0) >= RAG_SIMILARITY_THRESHOLD]
    if not strong_results:
        logger.info("[RAG] No result above threshold %.2f for: '%s' (best score=%s)",
                    RAG_SIMILARITY_THRESHOLD, query,
                    results[0]["score"] if results else "n/a")
        return "I don't have information on that yet. Please check with the Admin Block or try rephrasing your question."

    # Take the single best-matching chunk instead of stitching facts from
    # several chunks together. strong_results is ordered by score, so the
    # first chunk whose text actually contains the question's keywords is
    # the best answer — stop there rather than appending more chunks that
    # happen to also mention the same topic (which was producing repetitive,
    # multi-part answers for simple one-fact questions).
    answer = ""
    fallback_text = ""
    for r in strong_results[:3]:
        raw_text = (r.get("text") or "").strip()
        if not raw_text:
            continue
        raw_text = _strip_qa_labels(raw_text)
        if not raw_text:
            continue
        extracted, matched = _extract_relevant_sentences(raw_text, query, max_sentences=1)
        if matched:
            answer = extracted
            break
        if not fallback_text:
            fallback_text = raw_text

    if not answer:
        # Nothing matched a keyword anywhere — fall back to just the single
        # best-ranked chunk's full text rather than stitching several
        # unrelated chunks together.
        answer = fallback_text

    return answer or "I don't have information on that yet. Please check with the Admin Block or try rephrasing your question."


# ==========================================
# SERVER LIFECYCLE
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[SYSTEM] Booting server resources...")
    await ensure_indexes()
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


class QueryRequest(BaseModel):
    query: str = Field(..., description="Visitor's question, answered via the RAGService knowledge base")


@app.post("/api/chat")
async def proxy_to_rag(payload: QueryRequest):
    """
    Convenience POST endpoint: answers a question straight from the
    RAGService knowledge base (port 8600 by default), then logs the
    interaction to MongoDB and broadcasts it over the websocket so the
    frontend and the /logs-dashboard admin view both see it — same as /ask.
    """
    global _last_activity_ts
    _last_activity_ts = datetime.now().timestamp()

    if not verify_input_safety(payload.query):
        raise HTTPException(status_code=400, detail="Security Exception: Request contains blocked sequences.")

    sid = active_session["session_id"] if active_session else "unknown"
    fid = active_session.get("face_id") if active_session else None

    visitor_entry = _log_message(payload.query, "visitor")
    await manager.broadcast({"type": "message", **visitor_entry})

    answer = await query_rag_service(payload.query)

    try:
        await save_interaction(sid, payload.query, answer, face_id=fid)
    except Exception as exc:
        logger.error("[DATABASE ERROR] Failed to log interaction: %s", exc)

    kiosk_entry = _log_message(answer, "kiosk")
    await manager.broadcast({"type": "message", **kiosk_entry})

    return {"query": payload.query, "answer": answer, "source": "rag_service"}


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
# ==========================================
# IDENTITY / SESSION CORE
#   person (face_id)  1 --- N  visit threads (session_id)
#   Same person back within 30 days -> RESUME the same session_id.
# ==========================================

INSTITUTE_NAME = os.getenv("INSTITUTE_NAME", "R N S Institute of Technology")

def build_greeting(name: str, is_returning: bool, resumed: bool,
                    previous_topic: str | None = None) -> str:
    """The exact spoken lines for first-time vs returning visitors.

    MEMORY-AWARE RE-ENGAGEMENT: when we resume a visitor's thread AND we
    have a high-confidence topic label for their last stored question,
    the greeting names that topic and asks (never assumes) whether they
    want to continue with it. If no topic is available, we fall back to
    the generic resumed-session line — we never guess or hallucinate a
    previous topic.
    """
    who = name if name and name not in ("Guest", "Unknown", "") else "there"
    if not is_returning:
        return (f"Welcome {who}! I am Nova, the digital receptionist of {INSTITUTE_NAME}. "
                f"I can help you with admissions, departments, placements, fees, "
                f"and finding your way around campus. How may I assist you today?")
    if resumed:
        if previous_topic:
            return (f"Welcome back, {who}!Good to see you again, Last time you were asking about "
                    f"{previous_topic} — would you like to continue with that, "
                    f"or help with something else today?")
        return (f"Welcome back, {who}! Good to see you again. "
                f"We can continue where we left off. How may I assist you today?")
    return f"Welcome back, {who}! How may I assist you today?"


async def resume_or_create_session(face_id: str, user_name: str,
                                   is_returning: bool, visit_count: int,
                                   trigger: str = "camera") -> dict:
    """
    THE fix for the session_id == face_id bug.
      * face_id  = the PERSON  (minted once, by detection, at enrolment)
      * session_id = the VISIT THREAD (its own uuid, never the face_id)
    A known face returning inside 30 days resumes its previous session_id.
    """
    resumed = False
    continued_from = None
    session_id = None
    previous_topic = None

    if face_id:
        prev = await find_recent_session_by_face(face_id, days=30)
        if prev and prev.get("session_id"):
            session_id = prev["session_id"]          # SAME thread continues
            continued_from = prev.get("continued_from") or prev["session_id"]
            resumed = True
            visit_count = max(visit_count, int(prev.get("visit_count") or 1) + 1)

    if not session_id:
        session_id = str(uuid.uuid4())               # never face_id

    # ── Memory-aware re-engagement: look up what they last asked about ──
    # Pulls the last few (not just one) stored questions from their
    # previous session so the topic summary covers everything they were
    # asking about (e.g. "hostel facilities and fees"), not only their
    # final message. Only attempted for resumed sessions; failures here
    # are non-fatal and simply fall back to the generic greeting (see
    # build_greeting).
    if resumed:
        try:
            recent = await get_recent_interactions(session_id=session_id, face_id=face_id, limit=3)
            recent_questions = [r["input_text"] for r in recent if r.get("input_text")]
            if recent_questions:
                previous_topic = await extract_topic_label(recent_questions)
                logger.info(
                    "[SESSION] Re-engagement lookup: last_qs=%r -> topic=%r",
                    recent_questions, previous_topic,
                )
            else:
                logger.info("[SESSION] Re-engagement lookup: no prior interaction on file for face=%s session=%s", face_id, session_id)
        except Exception as e:
            logger.warning(f"[SESSION] Skipping re-engagement topic lookup: {e}")
            previous_topic = None

    sess = {
        "session_id":   session_id,
        "user_name":    user_name,
        "is_returning": is_returning,
        "visit_count":  visit_count,
        "face_id":      face_id or "",
        "trigger":      trigger,
        "asking_name":  False,
        "resumed":      resumed,
        "resumed_at":   datetime.now().isoformat(),
        "previous_topic": previous_topic,
        # True only when the greeting actually named a topic and asked a
        # "continue with that, or something else?" question — the NEXT
        # visitor reply, if it's a short accept/decline like "something
        # else" or "yes", is about THAT offer, not a new RAG-worthy
        # question, and must be intercepted before RAG (see /ask).
        "awaiting_topic_choice": bool(previous_topic),
        "greeting":     build_greeting(user_name, is_returning, resumed, previous_topic),
    }

    await save_session(session_id, face_id or None, user_name,
                       is_returning, visit_count,
                       continued_from=continued_from if resumed else None)
    logger.info(f"[SESSION] {'RESUMED' if resumed else 'NEW'} {session_id[:8]} "
                f"face={face_id[:8] if face_id else '-'} name={user_name}")
    return sess


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

    # Resume this PERSON's thread if seen in the last 30 days, else mint a
    # fresh session_id. session_id is NEVER face_id (that bug overwrote the
    # previous session record on every visit).
    new_sess = await resume_or_create_session(
        face_id=(face_id or "").strip(),
        user_name=user_name,
        is_returning=is_returning,
        visit_count=visit_count,
        trigger=trigger,
    )
    final_session_id = new_sess["session_id"]

    if active_session and active_session.get("session_id") == final_session_id:
        _last_activity_ts = datetime.now().timestamp()
        active_session["resumed_at"] = new_sess["resumed_at"]
        active_session["greeting"]   = new_sess["greeting"]
        await touch_session(final_session_id)
        return {
            "status":     "already_active",
            "session_id": final_session_id,
            "session":    active_session,
        }

    active_session    = new_sess
    message_log       = []
    _last_activity_ts = datetime.now().timestamp()

    # NOTE: resume_or_create_session() already persisted this session.
    await manager.broadcast({
        "type": "session_start",
        "session": active_session,
        "tts_text": active_session.get("greeting", ""),
    })
    return {"status": "success", "session_id": final_session_id, "session": active_session}


@app.post("/session/end")
async def end_session_endpoint(session_id: str = None):
    global active_session, _last_activity_ts
    sid = session_id or (active_session["session_id"] if active_session else None)
    
    active_session    = None
    _last_activity_ts = 0.0
    await manager.broadcast({"type": "session_end", "session_id": sid})
    return {"status": "success"}


@app.post("/session/are_you_there")
async def are_you_there_endpoint():
    global active_session
    if active_session:
        user_name = active_session.get("user_name") or "there"
        sid = active_session.get("session_id") or ""
        logger.info(f"[SESSION] Triggering 3s departure prompt for '{user_name}'")
        tts_prompt = f"Are you there, {user_name}?" if user_name not in ("Guest", "there", "Unknown", "") else "Are you there?"
        await manager.broadcast({
            "type": "are_you_there",
            "user_name": user_name,
            "session_id": sid,
            "tts_text": tts_prompt,
        })
        return {"status": "ok", "user_name": user_name}
    return {"status": "no_active_session"}


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
# Fallback/apology-shaped answers are NEVER cached at the normal 1-hour
# TTL. Without this guard, a question asked during a temporary outage
# (RAGService down, LLM down) gets its "I don't know" answer cached as
# if it were correct, and keeps serving that same apology for an hour
# even after everything recovers — exactly what happened with "where is
# canteen" during a RAGService outage tonight.
# Module-level (was previously defined inline inside ask_kiosk) so both
# /ask and /ask/stream share the exact same cacheability rule.
_UNCACHEABLE_PATTERNS = (
    "i don't have that detail",
    "temporarily unavailable",
    "having trouble accessing",
    "having trouble processing",
    "having trouble formatting",
    "connectivity issues",
    "couldn't generate a conversational response",   # Tier-3 degraded answer
)


def _is_cacheable(ans: str) -> bool:
    a = (ans or "").lower()
    return not any(p in a for p in _UNCACHEABLE_PATTERNS)


async def _deterministic_route(q_normalized: str, sid: str, visitor_name: str):
    """
    All the fast, deterministic pre-RAG routes — greeting, farewell,
    memory-recall, and topic-choice reply — factored out of ask_kiosk so
    /ask and /ask/stream take EXACTLY the same fast path for these and can
    never drift apart. Performs whatever side effects each route needs
    (session-state mutation, the farewell's session_end broadcast) itself,
    since those don't depend on which endpoint is asking.

    Returns (answer, source, session_action) if one of these matched, or
    None if the question needs the real RAG/LLM pipeline.
    """
    global active_session, _last_activity_ts

    # ─── Greeting → instant, deterministic, zero RAG/LLM round-trip ─────────
    if _matches_short_phrase(q_normalized, GREETING_PHRASES):
        answer = _GREETING_RESPONSES[hash(sid) % len(_GREETING_RESPONSES)]
        logger.info("[ROUTE] GREETING (deterministic) — '%s'", q_normalized)
        return answer, "greeting", "CONTINUE"

    # ─── Easter eggs → instant, deterministic, zero RAG/LLM round-trip ──────
    # Same short-utterance safety net as GREETING_PHRASES: only fires for a
    # standalone match (<=4 words), never for a real question that happens
    # to contain one of these phrases as a fragment.
    for phrase, responses in EASTER_EGGS.items():
        if _matches_short_phrase(q_normalized, {phrase}):
            answer = responses[hash(sid + phrase) % len(responses)]
            logger.info("[ROUTE] EASTER_EGG (deterministic) — '%s' -> '%s'", q_normalized, phrase)
            return answer, "easter_egg", "CONTINUE"

    # ─── Thank you / bye / natural sign-off → end session immediately ───────
    if _is_farewell(q_normalized):
        farewell = (
            f"You're welcome{', ' + visitor_name if visitor_name != 'there' else ''}! "
            "Have a great day. Goodbye!"
        )
        logger.info("[ROUTE] FAREWELL (deterministic) — '%s'", q_normalized)
        active_session    = None
        _last_activity_ts = 0.0
        await manager.broadcast({
            "type":       "session_end",
            "session_id": sid,
            "reason":     "thank_you",
        })
        return farewell, "farewell", "END"

    # ─── "What was my last session about?" → answer from stored memory ──────
    MEMORY_RECALL_PHRASES = (
        "last session", "last time", "previous session", "previous time",
        "what did i ask", "what did we talk about", "what did i talk about",
        "earlier session", "my last visit", "last visit",
    )
    if any(phrase in q_normalized for phrase in MEMORY_RECALL_PHRASES):
        prev_topic = (active_session or {}).get("previous_topic")
        if not prev_topic and active_session and (active_session.get("face_id") or active_session.get("session_id")):
            try:
                recent = await get_recent_interactions(
                    session_id=active_session.get("session_id"),
                    face_id=active_session.get("face_id"),
                    limit=3,
                )
                recent_questions = [r["input_text"] for r in recent if r.get("input_text")]
                if recent_questions:
                    prev_topic = await extract_topic_label(recent_questions)
            except Exception as e:
                logger.warning(f"[MEMORY RECALL] Lookup failed: {e}")

        if prev_topic:
            answer = (f"Last time you were asking about {prev_topic}. "
                      f"Want me to continue with that, or help with something else?")
        else:
            answer = ("I don't have a record of an earlier session for you — "
                      "this looks like the start of a fresh conversation. "
                      "What would you like help with today?")
        logger.info("[ROUTE] MEMORY_RECALL (deterministic) — '%s' -> topic=%r", q_normalized, prev_topic)
        return answer, "memory_recall", "CONTINUE"

    # ─── Reply to "continue with X, or something else?" re-engagement ───────
    if active_session and active_session.get("awaiting_topic_choice"):
        active_session["awaiting_topic_choice"] = False  # applies to this one reply only
        prev_topic = active_session.get("previous_topic")

        if _matches_short_phrase(q_normalized, TOPIC_DECLINE_PHRASES):
            answer = "Sure! What would you like help with today?"
            logger.info("[ROUTE] TOPIC_CHOICE_DECLINE (deterministic) — '%s'", q_normalized)
            return answer, "topic_choice_decline", "CONTINUE"

        if _matches_short_phrase(q_normalized, TOPIC_CONTINUE_PHRASES):
            if prev_topic:
                answer = f"Great, let's continue with {prev_topic}. What would you like to know?"
            else:
                answer = "Sure, what would you like to know?"
            logger.info("[ROUTE] TOPIC_CHOICE_CONTINUE (deterministic) — '%s'", q_normalized)
            return answer, "topic_choice_continue", "CONTINUE"
        # else: not a short accept/decline — treat as a real question and
        # fall straight through to the normal pipeline.

    return None


@app.get("/ask")
async def ask_kiosk(question: str = Query(..., description="Visitor question")):
    global _last_activity_ts, active_session
    _last_activity_ts = datetime.now().timestamp()

    # Apply the backend security guardrail validation immediately
    if not verify_input_safety(question):
        raise HTTPException(status_code=400, detail="Security Exception: Request contains blocked sequences.")

    q_clean = question.lower().strip()
    q_clean = q_clean.translate(str.maketrans('', '', string.punctuation)).strip()

    # Multi-word mis-hearings (e.g. "r n s fit") first, then single-word
    # ones — both feed into q_normalized, which is what's actually sent
    # to retrieval below (see the "corrected_question" note further down).
    for wrong_phrase, right_phrase in PHRASE_CORRECTIONS.items():
        q_clean = re.sub(rf"(?:^|\s){re.escape(wrong_phrase)}(?:$|\s)", f" {right_phrase} ", q_clean)
    q_clean = q_clean.strip()

    words           = q_clean.split()
    corrected_words = [DOMAINS_CORRECTIONS.get(w, w) for w in words]
    q_normalized    = " ".join(corrected_words)

    sid          = active_session["session_id"] if active_session else "unknown"
    fid          = active_session.get("face_id") if active_session else None
    visitor_name = (active_session.get("user_name") or "there") if active_session else "there"

    visitor_entry = _log_message(question, "visitor")
    await manager.broadcast({"type": "message", **visitor_entry})

    async def _respond(answer: str, source: str = "", session_action: str = "CONTINUE") -> dict:
        try:
            await save_interaction(sid, question, answer, face_id=fid)
        except Exception as exc:
            logger.error("[DATABASE ERROR] Failed to log interaction: %s", exc)
        kiosk_entry = _log_message(answer, "kiosk")
        await manager.broadcast({"type": "message", **kiosk_entry})
        result = {"question": question, "answer": answer, "session_action": session_action}
        if source:
            result["source"] = source
        return result

    # ─── Deterministic fast paths (greeting/farewell/memory-recall/topic-
    # choice), shared with /ask/stream — see _deterministic_route for why
    # this is factored out instead of inlined here. ─────────────────────
    det = await _deterministic_route(q_normalized, sid, visitor_name)
    if det is not None:
        answer, source, session_action = det
        return await _respond(answer, source=source, session_action=session_action)

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

    # ─── RNSIT_RAG / GENERAL_LLM / LIVE_INFO / UNSUPPORTED_EXTERNAL ─────────
    # generate_rag_kiosk_response is the real pipeline: condense follow-up
    # questions using recent history -> semantic search against RAGService
    # -> threshold check -> ground the LOCAL LLM in the retrieved context (or
    # route off-topic questions through _handle_offtopic) -> natural answer.
    # This is the function that was previously built but never wired up to
    # any live endpoint — /ask used to call the LLM-free query_rag_service
    # instead, which is why answers kept working with the LLM disconnected.
    recent_history = [
        {"speaker": m["speaker"], "text": m["text"]}
        for m in message_log[-6:]
        if m.get("index") != visitor_entry.get("index")
    ]

    try:
        # THE FIX: q_normalized already has DOMAINS_CORRECTIONS/PHRASE_
        # CORRECTIONS applied (typos, and STT mis-hearings of "RNSIT"
        # itself) but was previously only used for greeting/farewell/
        # memory-recall matching — RAG retrieval was still getting the
        # raw, uncorrected `question`, so a misheard "RNSFIT" never got
        # normalized back to "RNSIT" before the embedding search ran.
        answer = await generate_rag_kiosk_response(q_normalized, history=recent_history)

        if redis_client and answer:
            try:
                if _is_cacheable(answer):
                    redis_client.set(cache_key, answer, ex=3600)
                else:
                    logger.info("[REDIS] Skipped caching a fallback/apology answer "
                               "(would have poisoned this question for 1 hour): %r", answer[:80])
            except Exception as e:
                logger.warning("Redis write error: %s", e)

    except Exception as exc:
        logger.error("[LLM PIPELINE] generate_rag_kiosk_response failed: %s", exc)
        answer = "I'm having trouble processing that right now. Please visit the Admin Block for assistance."

    return await _respond(answer, source="rag_llm")


@app.get("/ask/stream")
async def ask_kiosk_stream(question: str = Query(..., description="Visitor question")):
    """
    Streaming sibling of /ask — sends the answer as Server-Sent Events,
    one event per SENTENCE, as soon as each sentence is ready, instead of
    one big JSON blob after the whole answer (and its whole LLM generation)
    is done. Lets the frontend start speaking the first sentence while the
    rest is still being generated (see generate_rag_kiosk_response_stream
    and chat_completion_with_fallback_stream in llm.py for where the actual
    streaming happens).

    Event shapes sent (each a `data: {...}\\n\\n` line):
      {"sentence": "..."}   — one for each completed sentence, in order
      {"done": true, "answer": "...", "session_action": "CONTINUE"|"END"}
        — always sent last, with the full reconstructed answer (so the
        frontend can still show/log the complete text) and whatever
        session_action a matched deterministic route (e.g. farewell) needs.

    Shares ALL the same routing (deterministic fast paths, Redis cache) as
    /ask via _deterministic_route — the only thing that's actually
    different between the two endpoints is HOW the RNSIT_RAG/LLM answer is
    delivered to the client once we know it's going through the slow path.
    """
    global _last_activity_ts, active_session
    _last_activity_ts = datetime.now().timestamp()

    if not verify_input_safety(question):
        raise HTTPException(status_code=400, detail="Security Exception: Request contains blocked sequences.")

    q_clean = question.lower().strip()
    q_clean = q_clean.translate(str.maketrans('', '', string.punctuation)).strip()
    for wrong_phrase, right_phrase in PHRASE_CORRECTIONS.items():
        q_clean = re.sub(rf"(?:^|\s){re.escape(wrong_phrase)}(?:$|\s)", f" {right_phrase} ", q_clean)
    q_clean = q_clean.strip()
    words           = q_clean.split()
    corrected_words = [DOMAINS_CORRECTIONS.get(w, w) for w in words]
    q_normalized    = " ".join(corrected_words)

    sid          = active_session["session_id"] if active_session else "unknown"
    fid          = active_session.get("face_id") if active_session else None
    visitor_name = (active_session.get("user_name") or "there") if active_session else "there"

    visitor_entry = _log_message(question, "visitor")
    await manager.broadcast({"type": "message", **visitor_entry})

    async def _finish(answer: str, session_action: str = "CONTINUE"):
        try:
            await save_interaction(sid, question, answer, face_id=fid)
        except Exception as exc:
            logger.error("[DATABASE ERROR] Failed to log interaction: %s", exc)
        kiosk_entry = _log_message(answer, "kiosk")
        await manager.broadcast({"type": "message", **kiosk_entry})

    async def _sse_gen():
        # ── Deterministic fast paths — identical decision to /ask ────────
        det = await _deterministic_route(q_normalized, sid, visitor_name)
        if det is not None:
            answer, source, session_action = det
            yield f"data: {json.dumps({'sentence': answer})}\n\n"
            await _finish(answer, session_action)
            yield f"data: {json.dumps({'done': True, 'answer': answer, 'session_action': session_action})}\n\n"
            return

        # ── Redis cache — also a single instant chunk, same as /ask ──────
        cache_key = f"kiosk:cache:{hashlib.md5(q_normalized.encode()).hexdigest()}"
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    logger.info("[REDIS HIT] For normalized key: '%s'", q_normalized)
                    yield f"data: {json.dumps({'sentence': cached})}\n\n"
                    await _finish(cached, "CONTINUE")
                    yield f"data: {json.dumps({'done': True, 'answer': cached, 'session_action': 'CONTINUE'})}\n\n"
                    return
            except Exception as e:
                logger.warning("Redis read error: %s", e)

        # ── Real streaming path: RNSIT_RAG / off-topic / etc. ─────────────
        recent_history = [
            {"speaker": m["speaker"], "text": m["text"]}
            for m in message_log[-6:]
            if m.get("index") != visitor_entry.get("index")
        ]

        parts: list[str] = []
        try:
            async for sentence in generate_rag_kiosk_response_stream(q_normalized, history=recent_history):
                parts.append(sentence)
                yield f"data: {json.dumps({'sentence': sentence})}\n\n"
            answer = "".join(parts).strip()
            if not answer:
                answer = "I'm having trouble processing that right now. Please visit the Admin Block for assistance."
                yield f"data: {json.dumps({'sentence': answer})}\n\n"
        except Exception as exc:
            logger.error("[LLM PIPELINE] generate_rag_kiosk_response_stream failed: %s", exc)
            answer = "I'm having trouble processing that right now. Please visit the Admin Block for assistance."
            yield f"data: {json.dumps({'sentence': answer})}\n\n"

        if redis_client and answer:
            try:
                if _is_cacheable(answer):
                    redis_client.set(cache_key, answer, ex=3600)
                else:
                    logger.info("[REDIS] Skipped caching a fallback/apology answer "
                               "(would have poisoned this question for 1 hour): %r", answer[:80])
            except Exception as e:
                logger.warning("Redis write error: %s", e)

        await _finish(answer, "CONTINUE")
        yield f"data: {json.dumps({'done': True, 'answer': answer, 'session_action': 'CONTINUE'})}\n\n"

    return StreamingResponse(_sse_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # nginx: don't buffer the stream if this ever sits behind one
    })
# ==========================================
# BIOMETRICS / FACE REGISTRATION ENDPOINTS
# ==========================================
class RegisterFacePayload(BaseModel):
    face_id:  str         = Field(..., description="Unique face id")
    name:     str         = Field(..., description="Person's name")
    encoding: List[float] = Field(..., description="Face encoding vector")
    encodings: List[List[float]] = Field(default_factory=list,
                                         description="Multi-template encodings (preferred)")

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
    await save_face_encoding(payload.face_id, payload.name, payload.encoding,
                             payload.encodings or [payload.encoding])
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


@app.websocket("/ws/detect")
async def detect_websocket(ws: WebSocket):
    """
    Browser-camera detection pipeline (cross-platform, no native window needed).

    Browser → backend : JSON  {"frame": "<base64 JPEG>"}
    Backend → browser : JSON  {present, state, identity, verified, bbox, bystanders, blink}

    bbox format when present: {x, y, w, h}  — pixel coords in the captured frame
    `blink` is an experimental, debounced one-frame blink EVENT (true for
    exactly the frame the blink completed on) — the frontend can use it as
    an optional "yes" gesture. It never affects detection.py's own state
    machine.
    """
    await ws.accept()
    logger.info("[WS/DETECT] Browser camera connected")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                b64 = msg.get("frame", "")
                if not b64:
                    continue
                # Strip Data-URL prefix if the browser sends image/jpeg;base64,...
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                frame_bytes = base64.b64decode(b64)
                result = await asyncio.to_thread(run_pipeline, frame_bytes)
                bbox = None
                if result.bbox:
                    bbox = {
                        "x": result.bbox.x,
                        "y": result.bbox.y,
                        "w": result.bbox.w,
                        "h": result.bbox.h,
                    }
                await ws.send_json({
                    "present":    result.present,
                    "state":      result.state,
                    "identity":   result.identity or "",
                    "verified":   result.verified,
                    "bbox":       bbox,
                    "bystanders": result.bystanders,
                    "blink":      bool(getattr(result, "blink", False)),
                })
            except Exception as frame_err:
                logger.warning(f"[WS/DETECT] Frame processing error: {frame_err}")
    except WebSocketDisconnect:
        logger.info("[WS/DETECT] Browser camera disconnected")


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
    global active_session, visitor_name_response, _last_activity_ts, message_log
    _last_activity_ts = datetime.now().timestamp()
    message_log = []  # BUG FIX: this used to be a same-name local shadowing the
    # module-level `message_log`, so it never actually cleared — a returning or
    # new visitor's conversation would silently inherit the PREVIOUS visitor's
    # message history, which corrupts follow-up context resolution ("its",
    # "what about ISE", etc. could resolve against a stranger's conversation).

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
            "tts_text": "Hi! May I know your name?"
        })
        return {"status": "asking", "session_id": active_session["session_id"]}

    # Recognised visitor: resume the 30-day thread (or start one) AND persist.
    # The old code built an in-memory session that never reached MongoDB and
    # used face_id as session_id.
    active_session = await resume_or_create_session(
        face_id=(payload.face_id or "").strip(),
        user_name=payload.name,
        is_returning=payload.is_returning,
        visit_count=payload.visit_count,
        trigger="camera",
    )
    final_session_id = active_session["session_id"]
    message_log = []
    _last_activity_ts = datetime.now().timestamp()

    await manager.broadcast({
        "type": "session_start",
        "session": active_session,
        "tts_text": active_session["greeting"],
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

    await manager.broadcast({
        "type": "asking_name",
        "session": active_session,
        "tts_text": "Hi! May I know your name?",
    })
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


# ==========================================
# RAG KNOWLEDGE BASE MANAGEMENT
# ==========================================
# NOTE: these reuse RAG_SERVICE_URL / RAG_COLLECTION defined near the top of
# this file — do not re-declare them here, or you'll re-introduce the same
# kind of NameError bug that used to crash this app on startup.


@app.post("/api/rag/upload")
async def rag_upload_file(
    file: UploadFile = File(...),
    source: str = Form(""),
    username: str = Depends(authenticate_admin),
):
    """Upload a document (PDF/DOCX/PPTX/TXT/MD/CSV) to the RAG knowledge base."""
    content = await file.read()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{RAG_SERVICE_URL}/v1/collections/{RAG_COLLECTION}/index/file",
            files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
            data={"source": source or file.filename},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"RAGService error: {resp.text}")
    return resp.json()


@app.get("/api/rag/files")
async def rag_list_files(username: str = Depends(authenticate_admin)):
    """List all files indexed in the RAG knowledge base."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{RAG_SERVICE_URL}/v1/collections/{RAG_COLLECTION}/files")
    if resp.status_code != 200:
        return {"files": []}
    return resp.json()


@app.get("/api/rag/stats")
async def rag_stats(username: str = Depends(authenticate_admin)):
    """Return collection stats (chunk count, indexed files) from RAGService."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{RAG_SERVICE_URL}/v1/collections/{RAG_COLLECTION}")
    if resp.status_code != 200:
        return {"error": "RAGService unreachable"}
    return resp.json()


@app.delete("/api/rag/files/{filename}")
async def rag_delete_file(filename: str, username: str = Depends(authenticate_admin)):
    """Remove a specific file's chunks from the RAG knowledge base."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.delete(
            f"{RAG_SERVICE_URL}/v1/collections/{RAG_COLLECTION}/files/{filename}"
        )
    if resp.status_code not in (200, 404):
        raise HTTPException(status_code=502, detail=f"RAGService error: {resp.text}")
    return {"message": f"Removed '{filename}' from knowledge base."}