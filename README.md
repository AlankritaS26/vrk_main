# VRK — Voice Receptionist Kiosk (Nova)

An AI-powered digital receptionist kiosk for **RNS Institute of Technology**. Visitors walk up, are recognized (or greeted and registered) by camera, and engage in natural, synchronized voice conversations — built on a self-hosted, high-performance, cost-effective stack.

**Fully hands-free: 100% voice-driven, zero clicking or touchscreen interaction required.**

---

## ⚡ Quick Start — One Command

To start all microservices, backend engines, and the kiosk browser:

```powershell
venv\Scripts\python.exe run.py
```

### What `run.py` does automatically:
1. **Port Self-Healing**: Checks ports `8001` (Backend), `8600` (RAG Service), and `3000` (Frontend). Safely frees stale kiosk processes if lingering.
2. **Parallel Boot Orchestration**:
   - Launches **RAGService** on port `8600` (loads sentence-transformer embedding model).
   - Launches **FastAPI Backend** on port `8001` (loads faster-whisper STT + Kokoro TTS models).
   - Launches **React Frontend** on port `3000` concurrently with Webpack dev server.
3. **Health Validation**: Asynchronously awaits health checks (`/health`) on ports `8001` and `8600`.
4. **Autoplay-Enabled Browser Launch**: Opens Google Chrome or Microsoft Edge in app mode with `--autoplay-policy=no-user-gesture-required` so Nova can speak greetings without requiring a prior mouse click.
5. **Unified Logging**: Prefixes logs clearly by service: `[BACKEND]`, `[RAG]`, `[FRONTEND]`, `[RUN]`.
6. **Graceful Shutdown**: Pressing `Ctrl+C` cleans up all child processes, sockets, and subprocess trees.

---

## Core System Capabilities

### 1. Voice-First Hands-Free Experience
- **Voice-Based Visitor Onboarding**: When a new face appears, Nova welcomes them. Silero Voice Activity Detection (VAD) activates automatically.
- **Auto-Guest Default (5 Seconds)**: If no voice response or blink is given within 5 seconds, Nova automatically proceeds in Guest mode.
- **Dynamic Mid-Session Name Change**: Visitors can change their name at any point. Names are immediately synchronized across MongoDB.
- **Double-Blink & Voice Confirmation**: Confirm names using voice or a double-blink gesture detected via MediaPipe EAR geometry.
- **Letter-by-Letter Spelling Mode**: If a name is misheard, visitors can say "No / Spell it" to spell their name letter by letter (with full NATO phonetic alphabet support).
- **Known Visitor Auto-Skip**: Returning or already-named visitors are greeted directly, skipping the onboarding prompt entirely.

### 2. Vision Intelligence & ArcFace Recognition
- **ArcFace + SCRFD Face Recognition**: Uses InsightFace ArcFace with canonical 5-point alignment to extract 512-d L2-normalized embeddings.
- **MediaPipe Landmark Tracking**: Real-time facial mesh processing for presence tracking, eye aspect ratio, and head alignment.
- **Blink & Double-Blink Detection (Eye Aspect Ratio - EAR)**: Real-time geometry calculation providing hands-free affirmative gesture input.
- **Nearest-Person / Largest-Face Selection**: Bounding box geometry prioritizes the closest person standing in front of the kiosk.
- **State Machine**: Clean progression from IDLE to DWELLING to RECOGNIZING to ACTIVE.

### 3. Hybrid Answer Engine - 5-Stage Intent Router

The answer pipeline uses a **strict 5-stage intent router** (`backend/confidence_rag.py`) that resolves every query through exactly one path, in priority order.

```
User Query
    |
    v
Stage 1: WEATHER  ->  Open-Meteo Live Weather API     source: weather_api
e.g. "What is the weather today?", "Will it rain?"
    |
    v (no weather intent)
Stage 2: TRAFFIC  ->  TomTom Live Traffic API          source: traffic_api
e.g. "How is the traffic near RNSIT?"
    |
    v (no traffic intent)
Stage 3: RNSIT SPECIFIC  ->  Canonical Entity KB       source: entity_kb
e.g. "Where is the library?", "Who is the HOD?"
Uses entity_mapping.py fast O(1) pattern rules
    |
    v (no specific entity match)
Stage 4: RNSIT GENERAL  ->  RAG + LLM Generation      source: general_rag
e.g. "Tell me about RNSIT", broad campus queries
Uses ChromaDB semantic search + Qwen/Gemini LLM
    |
    v (off-topic, no campus keywords)
Stage 5: UNSUPPORTED  ->  Guardrail Refusal            source: guardrail_refusal
"I am here to answer RNSIT-related questions only."
```

**Every query logs these 6 lines to the backend console:**
```
USER QUERY: <original text>
NORMALIZED QUERY: <after STT correction>
DETECTED INTENT: WEATHER | TRAFFIC | RNSIT | RNSIT_GENERAL | UNSUPPORTED
DETECTED ENTITY: <entity_id> | NONE
ENTITY CONFIDENCE: <0.00-1.00>
RETRIEVAL RESULT: <first 120 chars of answer>
ANSWER SOURCE: weather_api | traffic_api | entity_kb | general_rag | guardrail_refusal
```

### 4. Ultra-Fast Entity Retrieval (Sub-Millisecond)
- **Canonical Entity Mapping** (`backend/entity_mapping.py`): 50+ campus entities mapped via compiled regex patterns directly to verified answers from `college_info.json`. No LLM call, no vector search.
- **Examples**: "Where is the library?" -> LIBRARY (confidence 1.00), "Who is the CSE HOD?" -> CSE_HOD (confidence 1.00).
- **Broad Queries Route to RAG**: Intentionally broad queries like "Tell me something about RNSIT" do NOT collapse into a single entity — they route to RNSIT_GENERAL -> general_rag for a rich, contextual answer.

### 5. Neural TTS & Synchronized Audio/Text Rendering
- **Kokoro-82M Neural Synthesis**: High-quality `af_bella` voice at natural conversational pacing.
- **Full-Answer Rendering First**: When the backend answer arrives, the **complete text is rendered immediately** into the message bubble, then Nova speaks it. Text is always visible on screen **before** audio begins.
- **Pre-fetch TTS in Parallel**: The first audio chunk is fetched in parallel with the DOM paint cycle (double requestAnimationFrame) — Nova's voice starts within milliseconds of the text appearing.
- **Nova Timing Fix**: TTS START only fires after TEXT DOM RENDERED — Nova never speaks before her words appear on screen.
- **Sentence Chunking & Gapless Playback**: Pre-fetches upcoming sentence chunks and streams them through the Web Audio API without pauses.
- **Instant Acknowledgment (Thinking Filler)**: A short filler phrase plays immediately while the backend is processing — preserving natural conversation pacing.
- **Barge-In Interrupt & Resume**: Visitors can interrupt Nova mid-sentence. A resume offer is presented; saying "Continue" replays the full stored answer.
- **Browser Speech Synthesis Fallback**: Ensures the kiosk remains vocal even during network disruptions.

### 6. Admin Dashboard & Telemetry (/logs-dashboard)
- **Live Interaction Logs**: Full conversation timeline with Name Changed badges highlighting visitor rename events.
- **Face Tracks & Alias History**: Visual face records displaying Guest -> [Name] badges with historical name tracking.
- **Session Telemetry**: Track active/past kiosk sessions with face enrollment indicators.

---

## System Architecture

```
                                  Client Web Browser
                                  React 18 Kiosk App (Port 3000)
                                         |
                                  WebSocket /ws & HTTP API Requests
                                         v
         FastAPI Backend (Port 8001)
         |
         +-- Camera Detection (ArcFace, MediaPipe, Blink)
         +-- STT (faster-whisper, Silero VAD)
         +-- TTS (Kokoro-82M, Sentence Prefetch, Web Audio)
         +-- main.py Router (Session State, Guardrails)
         +-- confidence_rag.py (5-Stage Intent Router)
         |     Stage 1: WEATHER API
         |     Stage 2: TRAFFIC API
         |     Stage 3: entity_mapping.py (O(1) entity KB)
         |     Stage 4: RAG + LLM (Qwen -> Gemini -> RAG-only)
         |     Stage 5: Guardrail Refusal
         +-- MongoDB (Faces, Sessions, Interactions, Unanswered)
         +-- Redis / Memurai (Query Cache)
                |
                v (Semantic Search)
         RAGService (Port 8600)
         ChromaDB + Sentence-Transformers + RNSIT Knowledge Base
```

---

## API & WebSocket Endpoint Reference

| Endpoint | Method | Functionality |
|---|---|---|
| `/health` | GET | System health check (used by run.py during boot). |
| `/ws` | WebSocket | Central kiosk event channel (session transitions, audio cues). |
| `/ws/detect` | WebSocket | Live vision stream: presence, identity, bounding box, blink flags. |
| `/ws/stt` | WebSocket | Live streaming transcription socket. |
| `/stt/pcm` | POST | Ingests 16 kHz mono PCM audio and returns transcribed text. |
| `/tts` | POST | Generates base64 WAV speech audio from text using Kokoro-82M. |
| `/ask` | GET | Primary question answering: 5-stage intent router. |
| `/ask/stream` | GET (SSE) | Server-Sent Events stream for streaming LLM tokens (same routing). |
| `/session/start` | POST | Starts a session, records visitor metadata, returns custom greeting. |
| `/session/end` | POST | Closes active session and computes interaction statistics. |
| `/session/are_you_there` | POST | Re-engagement heartbeat checking if the visitor is still present. |
| `/visitor/unknown` | POST | Triggers voice onboarding modal for newly detected visitors. |
| `/visitor/submit_name` | POST | Confirms visitor name and registers face encoding to MongoDB. |
| `/visitor/rename` | POST | Renames an existing visitor identity. |
| `/faces/all` | GET | Fetches registered face vectors for detection caching. |
| `/api/rag/upload` | POST | Uploads and indexes campus documents into the RAG vector store. |
| `/api/rag/files` | GET | Lists all indexed documents in the active RAG knowledge base. |
| `/logs-dashboard` | GET | Protected admin visual dashboard for kiosk interactions. |

---

## Repository Structure

```
vrk_main/
+-- run.py                  Master one-command multi-service orchestrator
+-- .env                    Configuration & API keys (Mongo, Qwen/Gemini, Ports)
+-- README.md               Project documentation (this file)
|
+-- backend/                FastAPI Backend Application (Port 8001)
|   +-- main.py             API routes, WebSockets, session management
|   +-- confidence_rag.py   5-Stage Intent Router + Confidence-Based RAG pipeline
|   +-- entity_mapping.py   50+ canonical entity fast-lookup (sub-ms, no LLM call)
|   +-- query_correction.py STT normalization ("rns it" -> "rnsit", "ai ml" -> "aiml")
|   +-- detection.py        Camera pipeline, MediaPipe mesh, presence states
|   +-- recognition.py      ArcFace + SCRFD face embedding extraction
|   +-- stt.py              faster-whisper speech-to-text pipeline
|   +-- tts.py              Kokoro-82M neural TTS synthesizer
|   +-- llm.py & gemini.py  Qwen LLM engine, Gemini fallback, live weather/traffic APIs
|   +-- database.py         Motor async MongoDB connector & schemas
|   +-- audio_processing.py Bandpass DSP filters and noise gates
|   +-- requirements.txt    Python backend dependencies
|   +-- README.md           Backend-specific architecture guide
|
+-- RAGService/             Standalone Vector RAG Microservice (Port 8600)
|   +-- app.py              FastAPI RAG search API
|   +-- rag_store.py        ChromaDB vector collection manager
|   +-- embeddings.py       Sentence-transformers embedding engine
|   +-- file_parser.py      Document ingest (PDF, DOCX, TXT)
|   +-- ui.py               Streamlit management interface
|   +-- requirements.txt    RAG service dependencies
|
+-- frontend/               React 18 Kiosk Single Page Application (Port 3000)
|   +-- package.json        Node.js dependencies and scripts
|   +-- public/             Static assets (rnslogo.png, favicons)
|   +-- src/
|       +-- App.js              Top-level state coordinator & detection listener
|       +-- WelcomeScreen.js    Active voice screen, full-answer render, audio sync
|       +-- IdleScreen.js       Attract mode, vision state indicators, carousel
|       +-- GoodbyeScreen.js    Farewell transition card with RNSIT branding
|       +-- AriaAvatar.js       Animated vector avatar character engine (Nova)
|       +-- avatarReactions.js  Intent recognition & personality prefixes
|       +-- kioskMic.js         Browser VAD 16kHz audio capture
|       +-- index.css           Glassmorphic kiosk design system
|       +-- index.js            React root mount
|
+-- data/
|   +-- college_info.json   Initial structured campus reference corpus
+-- docs/
    +-- SETUP.md            First-time machine installation manual
    +-- ARCHITECTURE.md     Detailed system dataflow & diagrams
    +-- OPERATIONS.md       Production kiosk maintenance guide
```

---

## Testing & Verification Guide

### 1. Verifying the 5-Stage Intent Router

Start the backend and check the console output for these log lines on every query:

```
USER QUERY: What is the weather today?
NORMALIZED QUERY: what is the weather today?
DETECTED INTENT: WEATHER
DETECTED ENTITY: NONE
ENTITY CONFIDENCE: 1.00
RETRIEVAL RESULT: It is currently 26 degrees C in Bengaluru...
ANSWER SOURCE: weather_api
```

| Test Query | Expected DETECTED INTENT | Expected ANSWER SOURCE |
|---|---|---|
| "What is the weather today?" | WEATHER | weather_api |
| "How is the traffic near RNSIT?" | TRAFFIC | traffic_api |
| "Where is the library?" | RNSIT | entity_kb |
| "Who is the CSE HOD?" | RNSIT | entity_kb |
| "Tell me about RNSIT" | RNSIT_GENERAL | general_rag |
| "Who is the president of USA?" | UNSUPPORTED | guardrail_refusal |

### 2. Hands-Free Voice Onboarding & Auto-Guest
1. Run `python run.py`.
2. Stand in front of the camera as a new visitor.
3. Listen for Nova's onboarding prompt.
4. **Option A (Give Name)**: Say "Yes" or blink twice, then say your name. Nova confirms and saves your face identity.
5. **Option B (Guest Mode)**: Say "Guest" or wait 5 seconds without speaking — Nova auto-proceeds as Guest.

### 3. Mid-Conversation Name Change
1. Start as a Guest and ask campus questions.
2. Say: "Change my name to Akshata" or "Actually my name is Rahul".
3. Nova immediately updates MongoDB and confirms.
4. Check `/logs-dashboard` for the "Name Changed" badge.

### 4. Double-Blink Affirmation & Spelling Fallback
1. Say: "Change my name". Nova prompts for the new name.
2. Say your name. Nova asks for confirmation.
3. **Double-Blink**: Blink twice — confirmed!
4. **Spelling Mode**: Say "No / Spell it" — Nova enters letter-by-letter spelling mode.

### 5. Vision State & Recognition
1. Stand away: Attract screen shows "Walk up — I will recognise you".
2. Approach the camera: State changes through DWELLING -> RECOGNIZING -> ACTIVE.
3. If previously registered: Greeted with personalized greeting, skipping the onboarding prompt.

### 6. Personality & Easter Eggs
- Ask: "Are you a robot?" -> Nova gives a witty answer.
- Ask: "Tell me a joke" -> Nova tells a campus joke.
- Say: "Thank you, bye!" -> Nova delivers a personalized farewell.

---

## Recent Fixes (Session: Sep 2026)

The following targeted fixes were applied without redesigning the UI or removing working functionality:

| Issue | Fix Applied |
|---|---|
| Nova spoke before text appeared | TTS START now fires only after TEXT DOM RENDERED (double requestAnimationFrame DOM paint cycle). |
| Answer appeared sentence-by-sentence | sendToBackend now renders the full answer text to the message bubble at once, then calls speakStream. |
| Broad RNSIT queries returned "I do not have that detail" | COLLEGE_OVERVIEW removed from _ENTITY_RULES; broad queries now route to RNSIT_GENERAL -> general_rag with overview context injection. |
| Weather & traffic used hardcoded/easter-egg answers | Intent router in confidence_rag.py now checks for weather/traffic BEFORE entity matching, calling live Open-Meteo and TomTom APIs respectively. |

---

## Project Team

**RNS Institute of Technology - VRK Project Team**
- **Alankrita Singh**
- **Akshatha A**
- **B Sneha**
