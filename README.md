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

## 🌟 Core System Capabilities

### 1. 🎙️ Voice-First Hands-Free Experience
- **Voice-Based Visitor Onboarding**: When a new face appears, Nova welcomes them: *"Hi! I'm Nova. What's your name?"* Silero Voice Activity Detection (VAD) activates automatically.
- **Natural Spoken Name Recognition**: Visitors can say *"Akshatha"*, *"I am Sneha"*, or *"My name is Alankrita"*. The system transcribes the name, displays a confirmation badge, and **auto-advances after 3 seconds** without touching the screen.
- **Zero-Touch Interaction**: From arrival to departure, all operations—including inquiries, topic selection, re-engagement, and farewells—are handled entirely by voice.

### 2. 👁️ Vision Intelligence & ArcFace Recognition
- **ArcFace + SCRFD Face Recognition**: Uses InsightFace ArcFace (`w600k_r50.onnx`) with canonical 5-point alignment to extract 512-d L2-normalized embeddings for fast cosine similarity matching against MongoDB face records.
- **MediaPipe Landmark Tracking**: Real-time facial mesh processing for presence tracking and eye aspect ratio.
- **Blink Detection (Eye Aspect Ratio - EAR)**: Real-time geometry calculation ($EAR < 0.15$ closed, $EAR \ge 0.20$ open) for natural blink detection.
- **Nearest-Person / Largest-Face Selection**: Bounding box geometry prioritizes the closest person standing in front of the kiosk.
- **State Machine**: Clean progression from `IDLE` ("Walk up — I'll recognise you") $\to$ `DWELLING` ("I see you — hold still…") $\to$ `RECOGNIZING` ("Identifying…") $\to$ `ACTIVE` ("Welcome, {Name}!").

### 3. 🧠 Hybrid RAG Answer Engine & Multi-Tier LLM Architecture
- **Primary LLM (Qwen)**: Uses Qwen (`Qwen/Qwen2.5-7B-Instruct` or local OpenAI-compatible inference) as the primary generation engine.
- **Google Gemini Fallback**: Seamless fallback to Gemini Flash if the primary local LLM is unreachable.
- **RAGService Microservice (Port 8600)**: Standalone ChromaDB vector store with `sentence-transformers` embeddings delivering high-precision semantic retrieval over campus data.
- **Grounded Campus Knowledge**: College syllabus, courses, departments, fee structures, faculty contacts, placement statistics, and FAQs.
- **Redis / Memurai Caching**: Low-latency caching layer for recurring queries and pre-computed embeddings.

### 4. 🎭 Conversational Personality & Easter Eggs
- **Intent Analysis**: Classifies questions into intents (`curious`, `happy`, `thanks`, `greeting`, `confused`) to trigger contextual avatar expressions and prefixes.
- **Interactive Small Talk**: Built-in deterministic responses for conversational moments (*"Are you a robot?"*, *"Tell me a joke"*, *"How are you?"*, *"You are smart"*).
- **Time-Aware Salutations**: Contextual greetings dynamically adapted to the time of day.

### 5. 🔊 Neural Text-To-Speech & Audio Sync
- **Kokoro-82M Neural Synthesis**: High-quality `af_bella` voice at natural conversational pacing.
- **Sentence Chunking & Gapless Playback**: Pre-fetches upcoming sentence chunks and streams them through the Web Audio API without pauses.
- **Synchronized Text Bubbles & Auto-Scroll**: Spoken speech is synchronized with animated message bubbles, automatically scrolling conversation threads into focus.
- **Browser Speech Synthesis Fallback**: Ensures the kiosk remains vocal even during network disruptions.

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────────┐
                                  │      Client Web Browser       │
                                  │  React 18 Kiosk App (Port 3000)│
                                  └───────────────┬───────────────┘
                                  WebSocket /ws   │ HTTP API Requests
                                 & /ws/detect     │ (/ask, /tts, /stt/pcm)
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FastAPI Backend (Port 8001)                                   │
│                                                                                                 │
│  ┌───────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────────────────┐  │
│  │   Camera Detection    │  │   Speech-To-Text (STT)  │  │      Text-To-Speech (TTS)         │  │
│  │ • ArcFace + SCRFD     │  │ • faster-whisper        │  │ • Kokoro-82M ('af_bella')         │  │
│  │ • MediaPipe Mesh      │  │ • Bandpass DSP Filters  │  │ • Silence Trimmer                 │  │
│  │ • Blink Detection     │  │ • Silero VAD Processor  │  │ • Sentence Chunk Prefetch Cache   │  │
│  └───────────┬───────────┘  └────────────┬────────────┘  └─────────────────▲─────────────────┘  │
│              │                           │                                 │                    │
│  ┌───────────▼───────────────────────────▼─────────────────────────────────┴─────────────────┐  │
│  │                                     main.py Router                                        │  │
│  │       Session State Machine • Safety Guardrails • Intent Router • Easter Eggs Intercept   │  │
│  └───────────┬───────────────────────────┬─────────────────────────────────┬─────────────────┘  │
│              │                           │                                 │                    │
│  ┌───────────▼───────────┐  ┌────────────▼────────────┐  ┌─────────────────▼─────────────────┐  │
│  │     LLM Router        │  │   MongoDB (Motor Async) │  │      Redis / Memurai Caching      │  │
│  │ • Primary: Qwen LLM   │  │ • Visitor Face Vectors  │  │ • Query Cache                     │  │
│  │ • Fallback: Gemini    │  │ • Session Logs          │  │ • Hot Chunk Retrieval             │  │
│  └───────────┬───────────┘  └─────────────────────────┘  └───────────────────────────────────┘  │
└──────────────┼──────────────────────────────────────────────────────────────────────────────────┘
               │ Semantic Search Query
               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                RAGService Microservice (Port 8600)                              │
│         ChromaDB Vector Database • Sentence-Transformers • RNSIT College Knowledge Base         │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📡 API & WebSocket Endpoint Reference

| Endpoint | Method / Protocol | Functionality |
|---|---|---|
| `/health` | `GET` | System health check (used by `run.py` during boot). |
| `/ws` | `WebSocket` | Central kiosk event channel (session transitions, audio cues). |
| `/ws/detect` | `WebSocket` | Live vision stream: presence, identity, bounding box, blink flags, distance. |
| `/ws/stt` | `WebSocket` | Live streaming transcription socket. |
| `/stt/pcm` | `POST` | Ingests 16 kHz mono PCM audio and returns transcribed text. |
| `/tts` | `POST` | Generates base64 WAV speech audio from text using Kokoro-82M. |
| `/ask` | `GET` | Primary question answering: safety $\to$ RAG retrieval $\to$ Qwen/Gemini synthesis. |
| `/ask/stream` | `GET` (SSE) | Server-Sent Events stream for streaming LLM tokens. |
| `/session/start` | `POST` | Starts a session, records visitor metadata, and returns custom greeting. |
| `/session/end` | `POST` | Closes active session and computes interaction statistics. |
| `/session/are_you_there`| `POST` | Re-engagement heartbeat checking if the visitor is still present. |
| `/visitor/unknown` | `POST` | Triggers voice onboarding modal for newly detected visitors. |
| `/visitor/submit_name` | `POST` | Confirms visitor name and registers face encoding to MongoDB. |
| `/visitor/rename` | `POST` | Renames an existing visitor identity. |
| `/faces/all` | `GET` | Fetches registered face vectors for detection caching. |
| `/api/rag/upload` | `POST` | Uploads and indexes campus documents into the RAG vector store. |
| `/api/rag/files` | `GET` | Lists all indexed documents in the active RAG knowledge base. |
| `/logs-dashboard` | `GET` | Protected administrative visual dashboard for kiosk interactions. |

---

## 📁 Repository Structure

```
vrk_main/
├── run.py                 # Master one-command multi-service orchestrator
├── .env                   # Configuration & API keys (Mongo, Qwen/Gemini, Ports)
├── README.md              # Project documentation (this file)
│
├── backend/               # FastAPI Backend Application (Port 8001)
│   ├── main.py            # API routes, WebSockets, session management
│   ├── detection.py       # Camera pipeline, MediaPipe mesh, presence states
│   ├── recognition.py     # ArcFace + SCRFD face embedding extraction
│   ├── stt.py             # faster-whisper speech-to-text pipeline
│   ├── tts.py             # Kokoro-82M neural TTS synthesizer
│   ├── llm.py & gemini.py # Qwen LLM engine, Gemini fallback & prompt safety
│   ├── database.py        # Motor async MongoDB connector & schemas
│   ├── audio_processing.py# Bandpass DSP filters and noise gates
│   ├── face_landmarker.task# MediaPipe landmark model binary
│   ├── requirements.txt   # Python backend dependencies
│   └── README.md          # Backend-specific architecture guide
│
├── RAGService/            # Standalone Vector RAG Microservice (Port 8600)
│   ├── app.py             # FastAPI RAG search API
│   ├── rag_store.py       # ChromaDB vector collection manager
│   ├── embeddings.py      # Sentence-transformers embedding engine
│   ├── file_parser.py     # Document ingest (PDF, DOCX, TXT)
│   ├── ui.py              # Streamlit management interface
│   └── requirements.txt   # RAG service dependencies
│
├── frontend/              # React 18 Kiosk Single Page Application (Port 3000)
│   ├── package.json       # Node.js dependencies and scripts
│   ├── public/            # Static assets (rnslogo.png, favicons)
│   ├── scripts/
│   │   └── copyVadAssets.js# Script bundling Silero VAD WASM/ONNX assets
│   ├── src/
│   │   ├── App.js         # Top-level state coordinator & detection listener
│   │   ├── WelcomeScreen.js# Active voice screen, name capture, audio sync
│   │   ├── IdleScreen.js  # Attract mode, vision state indicators, carousel
│   │   ├── GoodbyeScreen.js# Farewell transition card with RNSIT branding
│   │   ├── AriaAvatar.js  # Animated vector avatar character engine (Nova)
│   │   ├── avatarReactions.js# Intent recognition & personality prefixes
│   │   ├── kioskMic.js    # Browser VAD 16kHz audio capture
│   │   ├── index.css      # Glassmorphic kiosk design system
│   │   └── index.js       # React root mount
│   └── README.md          # Frontend-specific architecture guide
│
├── data/
│   └── college_info.json  # Initial structured campus reference corpus
└── docs/
    ├── SETUP.md           # First-time machine installation manual
    ├── ARCHITECTURE.md    # Detailed system dataflow & diagrams
    └── OPERATIONS.md      # Production kiosk maintenance guide
```

---

## 🧪 Testing & Verification Guide

### 1. Hands-Free Voice Onboarding
1. Run `python run.py`.
2. Stand in front of the camera as a new visitor.
3. Listen for Nova: *"Hi! I'm Nova. What's your name?"*
4. Speak naturally: *"Akshatha"* or *"I am Sneha"*.
5. Observe the live audio waveform. The screen updates to: `"✓ Got it! Akshatha"`.
6. Without touching anything, the system auto-proceeds after 3 seconds and welcomes you.

### 2. Vision State & Recognition
1. Stand away: Attract screen shows `Walk up — I'll recognise you`.
2. Approach the camera: State changes to `I see you — hold still…` $\to$ `Identifying…`.
3. If registered: Greeted with personalized greeting (*"Welcome back, {Name}!"*).

### 3. Blink Detection (Yes/No Confirmation)
1. Open Developer Tools (`F12`) $\to$ **Network** $\to$ `ws/detect`.
2. Blink naturally in front of the camera.
3. Observe WebSocket payload stream:
   ```json
   {
     "blink_detected": true,
     "eyes_closed": true,
     "ear_left": 0.09,
     "ear_right": 0.08
   }
   ```

### 4. Personality & Easter Eggs
- Ask: *"Are you a robot?"* $\to$ *"I'm a digital receptionist, so yes and no — no body, but I do the job!"*
- Ask: *"Tell me a joke"* $\to$ *"Why did the student bring a ladder to class? To reach the higher studies!"*
- Say: *"Thank you so much, bye!"* $\to$ Nova delivers a personalized farewell and transitions cleanly to the goodbye screen.

---

## 👥 Project Team

**RNS Institute of Technology — VRK Project Team**
- **Alankrita Singh**
- **Akshatha A**
- **B Sneha**
