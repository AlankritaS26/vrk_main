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

## 🔐 Access Credentials & Portals

| Portal / Feature | URL / Endpoint | Default Credentials | Supported Device | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Interactive Kiosk** | `http://localhost:3000` | *None (Public Voice Terminal)* | **Laptop / Desktop Terminal Only** | Main digital receptionist (Nova). 100% voice-driven, vision recognition, no touch buttons. Blocked on mobile phones. |
| **Front Desk Staff Dashboard** | `http://127.0.0.1:8001/staff`<br>*(or `http://<LAN-IP>:8001/staff`)* | Username: `staff`<br>Password: `rnsit2024`<br>*(Admin: `admin` / `111111`)* | **Staff Laptop / Tablet** | Real-time human handover queue, incoming visitor alerts with audio chime, live transcripts, Accept & Resolve actions. |
| **Admin Control Plane** | `http://127.0.0.1:8001/logs-dashboard` | Username: `admin`<br>Password: `111111` | **Admin Laptop / PC** | Live conversation logs, rename history audit, biometric face encodings, and system wipe controls. |
| **Mobile Session Summary & Brochure Portal** | `http://<LAN-IP>:8001/companion/<token>` | *None (Secured by 20-min session token)* | **Mobile Phone (Scan QR on Kiosk)** | Visitor takes away their session summary, personalized recap, official PDF brochures, and quick links on smartphone. |

### 🛑 Platform & Device Rules:
1. **Interactive Kiosk (Port 3000)**:
   - Built strictly for laptops, desktops, or kiosk terminals equipped with a webcam and microphone.
   - **Mobile Access Blocked**: If opened on a smartphone browser, a dedicated screen displays: *"Kiosk Not Available on Mobile. The interactive voice kiosk is designed for the terminal screen. Scan the QR code on the kiosk screen to view your session summary."*
2. **Mobile Companion (Port 8001)**:
   - Designed specifically for smartphone screens.
   - Visitors scan the dynamic QR code on the kiosk screen to view their personalized visit summary and download official college brochures.

### 🎙️ Pure Voice-Only Interaction (No Clicking):
- **Zero Touch Required**: All clickable buttons ("Talk to Staff", "Phone Companion") have been removed from the kiosk header.
- **Escalation Trigger**: Say *"I want to talk to a person"*, *"Connect me to human staff"*, *"I need a real person"*, or report an emergency (*"There is a medical emergency"*). Nova automatically alerts the staff dashboard.

---

## 🌟 Core System Capabilities

### 1. 🎙️ Voice-First Hands-Free Experience
- **Voice-Based Visitor Onboarding**: When a new face appears, Nova welcomes them: *"Would you like to give your name or continue as guest? Say 'Yes' or blink twice to give your name, or say 'Guest' to continue as guest."* Silero Voice Activity Detection (VAD) activates automatically.
- **Auto-Guest Default (5 Seconds)**: If no voice response or blink is given within 5 seconds, Nova automatically proceeds in Guest mode (*"Continuing as Guest! How may I assist you today?"*).
- **Dynamic Mid-Session Name Change**: Visitors can change their name at any point (e.g., *"Change my name to Akshata"*, *"Call me Rahul"*, or bare *"Change my name"*). Names are immediately synchronized across MongoDB (`faces`, `sessions`, `interactions`) and broadcast via WebSocket.
- **Double-Blink & Voice Confirmation**: Confirm names using voice (*"Yes"*) or a double-blink gesture detected via MediaPipe EAR geometry.
- **Letter-by-Letter Spelling Mode**: If a name is misheard, visitors can say *"No / Spell it"* to spell their name letter by letter (with full NATO phonetic alphabet support: *Alpha, Bravo, Charlie...*).
- **Known Visitor Auto-Skip**: Returning or already-named visitors are greeted directly (*"Welcome back, {Name}!"*), skipping the onboarding prompt entirely.

### 2. 👁️ Vision Intelligence & ArcFace Recognition
- **ArcFace + SCRFD Face Recognition**: Uses InsightFace ArcFace (`w600k_r50.onnx`) with canonical 5-point alignment to extract 512-d L2-normalized embeddings for fast cosine similarity matching against MongoDB face records.
- **MediaPipe Landmark Tracking**: Real-time facial mesh processing for presence tracking, eye aspect ratio, and head alignment.
- **Blink & Double-Blink Detection (Eye Aspect Ratio - EAR)**: Real-time geometry calculation ($EAR < 0.15$ closed, $EAR \ge 0.20$ open) providing hands-free affirmative gesture input.
- **Nearest-Person / Largest-Face Selection**: Bounding box geometry prioritizes the closest person standing in front of the kiosk.
- **State Machine**: Clean progression from `IDLE` ("Walk up — I'll recognise you") $\to$ `DWELLING` ("I see you — hold still…") $\to$ `RECOGNIZING` ("Identifying…") $\to$ `ACTIVE` ("Welcome, {Name}!").

### 3. 🧠 Hybrid RAG Answer Engine & Multi-Tier LLM Architecture
- **Primary LLM (Qwen)**: Uses Qwen (`Qwen/Qwen2.5-7B-Instruct` or local OpenAI-compatible inference) as the primary generation engine.
- **Google Gemini Fallback**: Seamless fallback to Gemini Flash (`gemini-2.0-flash` / `gemini-1.5-flash`) if the primary local LLM is unreachable.
- **RAGService Microservice (Port 8600)**: Standalone ChromaDB vector store with `sentence-transformers` embeddings delivering high-precision semantic retrieval over campus data.
- **Direct-Answer Persona Tuning & Zero-Boilerplate Pipeline**: System prompts and multi-stage regex scrubbing (`_clean_repetitive_greeting`) strip redundant greetings (*"Hello! I am Nova"*, *"As Nova..."*, *"Nova here"*, *"Certainly!"*) from answer streams so Nova delivers direct, punchy, and conversational campus facts immediately without repetitive self-introductions on every question turn.
- **Strict Receptionist Persona Boundaries**: Enforced system prompts guarantee the assistant always identifies as **Nova** and never confuses visitor names with its own identity.
- **Persistent Thinking Action**: Visual thinking indicators (avatar thinking arm pose, `💭 Thinking…` status badge, thinking bubble) stay seamlessly active while RAG answers generate.
- **Grounded Campus Knowledge**: College syllabus, courses, departments, fee structures, faculty contacts, placement statistics, and FAQs.
- **Redis / Memurai Caching**: Low-latency caching layer for recurring queries and pre-computed embeddings.

### 4. 📊 Admin Dashboard & Telemetry (`/logs-dashboard`)
- **Live Interaction Logs**: Full conversation timeline with `🏷️ Name Changed` badges highlighting visitor rename events.
- **Face Tracks & Alias History**: Visual face records displaying `✏️ Guest → [Name]` badges with historical name tracking and timestamp audit trails.
- **Session Telemetry**: Track active/past kiosk sessions with face enrollment indicators.

### 5. 🔊 Neural Text-To-Speech & Audio Sync
- **Kokoro-82M Neural Synthesis**: High-quality `af_bella` voice at natural conversational pacing.
- **Sentence Chunking & Gapless Playback**: Pre-fetches upcoming sentence chunks and streams them through the Web Audio API without pauses.
- **Synchronized Text Bubbles & Auto-Scroll**: Spoken speech is synchronized with animated message bubbles, automatically scrolling conversation threads into focus.
- **Browser Speech Synthesis Fallback**: Ensures the kiosk remains vocal even during network disruptions.

### 6. 📱 QR Companion & Mobile Brochure Portal
- **Always-Visible Hands-Free QR Card**:
  - Automatically rendered in the **top-left corner of the active kiosk screen** (next to Nova) as soon as the session starts.
  - **Zero Touching or Clicking Required**: Visitors simply point their smartphone camera at the screen at any point during their conversation.
  - Features clear visual badging: *"📱 Phone Companion — Scan to take Session Summary & PDF Brochures with you"*.
- **Voice-Activated Guidance**:
  - If a visitor asks: *"Where is the QR code?"*, *"How to get the brochure?"*, or *"Can I get this on my phone?"*, Nova vocally guides them: *"The QR code is displayed right on the top left of the screen! Scan it with your phone's camera to take this conversation summary and college brochures with you."*
- **Seamless Phone Handover**: Visitors scan the dynamic QR code on the kiosk screen to continue their interaction on their personal mobile device without installing any app.
- **Short-Lived Secure Tokens**: Backend generates a unique, cryptographically random 32-byte session token with a 20-minute TTL (stored in Redis and in-memory). Tokens are invalidated immediately upon session termination.
- **Accurate & Personalized Session Summary**: The mobile landing page (`/companion/{token}`) pulls conversation history from MongoDB (by `session_id` and `face_id`) and uses the local LLM to generate an accurate 2–4 sentence recap strictly grounded in the conversation, addressing the visitor by their recognized or updated name.
- **Executive-Grade Official PDF Brochures**: The backend automatically maps conversation topics (e.g. Computer Science, Electronics, Admissions, Hostel, Fees, Placements) to official campus brochures (`/companion/brochure/{token}`) stored in `data/brochures/`. Brochures feature full-width RNSIT Navy (`#0B192C`) and Gold (`#D97706`) branding, official autonomous VTU accreditation badges, structured statistics tables, and verified campus contact info.
- **Available Brochures**:
  - `cse.pdf` — Computer Science & Engineering (720 seats, ₹50 LPA package, AI & Cloud tracks)
  - `ece.pdf` — Electronics & Communication (VLSI, Texas Instruments Lab, Qualcomm)
  - `ise.pdf` — Information Science & Engineering (Data Science, Cybersecurity)
  - `me.pdf` — Mechanical Engineering (Toyota Center of Excellence, EV racing)
  - `civil.pdf` — Civil Engineering (Smart Infrastructure, NABL Soil & Concrete lab)
  - `eee.pdf` — Electrical & Electronics (Smart Grids, EV Powertrains)
  - `admissions.pdf` — Official Admissions Guide (KCET E118, COMEDK E104, steps & eligibility)
  - `placements.pdf` — Placement Highlights (200+ recruiters, ₹50 LPA, 1,060+ offers)
  - `hostel.pdf` — On-Campus Hostels (Boys 530 cap, Girls 300 cap, vegetarian mess, 24/7 security)
  - `fees.pdf` — Fee Structure & Scholarships (Govt KCET, COMEDK, SSP & NSP portals)
  - `mba.pdf` — Management Studies (VTU Autonomous MBA, dual specializations)
  - `general.pdf` — General Campus Prospectus & Overview
- **Graceful Token Degradation**: Expired or invalid tokens display a clear, helpful error card with campus contact details and office hours.

### 7. 🛎️ Human Handover & Front Desk Escalation System
- **State Machine Architecture**: `BOT_HANDLING` $\to$ `ESCALATION_REQUESTED` $\to$ `STAFF_NOTIFIED` $\to$ `STAFF_CONNECTED` $\to$ `RESOLVED` (with a 60-second auto-timeout reverting to bot if staff is unavailable).
- **Purely Voice-Triggered Escalation — No Buttons**:
  - **Explicit Voice Intent**: Say *"I want to talk to a person"*, *"Speak to human staff"*, *"Connect me to front desk"*, *"I need a real person"*, *"Talk to someone"*, or *"Transfer me"*.
  - **Low RAG Confidence**: 3 consecutive low-confidence / uncacheable fallback answers automatically trigger escalation.
  - **Sensitive Keywords**: Automatic priority routing on queries containing *"emergency"*, *"medical"*, *"accident"*, *"harassment"*, or *"police"*.
- **Live Staff Dashboard** — accessed by staff (not visitors) at `http://127.0.0.1:8001/staff`:
  - Protected with HTTP Basic authentication:
    - **Staff**: Username `staff` / Password `rnsit2024`
    - **Admin**: Username `admin` / Password `111111`
  - Real-time WebSocket alerts, live pending queue, conversation transcripts, and **Accept** / **Resolve** controls.
  - Staff open this URL on a separate laptop/tablet at the front desk and log in once.
- **Kiosk Wait UX**: Smooth glassmorphic modal with pulsing indicator, connection countdown, and auto-timeout after 60 seconds.

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────────┐
                                  │      Client Web Browser       │
                                  │ React 18 Kiosk App (Port 3000)│
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
| `/companion/token` | `POST` | Issues short-lived companion token and URL for current session. |
| `/companion/{token}` | `GET` (HTML) | Responsive mobile companion landing page with session summary. |
| `/companion/qr/{token}` | `GET` (PNG) | Generates and streams PNG QR code image for mobile scanning. |
| `/companion/validate/{token}` | `GET` | Validates companion token and returns session transcript summary. |
| `/companion/brochure/{token}` | `GET` (PDF) | Streams contextual department or topic brochure PDF. |
| `/escalation/request` | `POST` | Triggers a human handover request and notifies staff. |
| `/escalation/active` | `GET` | Lists all active escalation requests for staff queue. |
| `/escalation/accept/{session_id}` | `POST` | Staff member accepts an escalation request. |
| `/escalation/resolve/{session_id}` | `POST` | Staff marks escalation resolved, returning kiosk to bot. |
| `/staff` | `GET` (HTML) | Protected real-time staff escalation dashboard. |

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
│   ├── companion.py       # QR companion endpoints, mobile page, brochure matching
│   ├── escalation.py      # Human handover state machine & staff dashboard
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

### 1. Hands-Free Voice Onboarding & Auto-Guest
1. Run `python run.py`.
2. Stand in front of the camera as a new visitor.
3. Listen for Nova: *"Would you like to give your name or continue as guest? Say 'Yes' or blink twice to give your name, or say 'Guest' to continue as guest."*
4. **Option A (Give Name)**: Say *"Yes"* or blink twice $\to$ say your name (*"Akshatha"*). Nova confirms and saves your face identity.
5. **Option B (Guest Mode)**: Say *"Guest"* or wait 5 seconds without speaking $\to$ Nova auto-proceeds as Guest.

### 2. Mid-Conversation Name Change
1. Start as a Guest and ask a few campus questions (e.g. *"Where is the CSE department?"*).
2. Say: *"Change my name to Akshata"* or *"Actually my name is Rahul"*.
3. Nova immediately updates MongoDB (`faces`, `sessions`, `interactions`) and confirms:
   > *"Done! I have changed your name to Akshata. How may I assist you today?"*
4. Check `/logs-dashboard`:
   - **Interactions Tab**: Displays the conversation with a `🏷️ Name Changed` tag.
   - **Face Tracks Tab**: Displays your profile with `✏️ Guest → Akshata`.

### 3. Double-Blink Affirmation & Spelling Fallback
1. Say: *"Change my name"*.
2. Nova prompts: *"Sure! What should I change your name to?"*.
3. Speak your name or name with spelling: *"Akshata, AKSHA, THA"*.
4. Nova asks: *"Got it — should I call you Akshata? Say yes or blink twice to confirm, or say no to spell it out."*
5. **Double-Blink**: Blink twice in front of the camera $\to$ confirmed!
6. **Spelling Mode**: Say *"No / Spell it"* $\to$ Nova enters letter-by-letter spelling mode (*"A"*, *"K"*, *"S"*, *"H"*, *"A"*, *"T"*, *"A"*, *"Done"*).

### 4. Vision State & Recognition
1. Stand away: Attract screen shows `Walk up — I'll recognise you`.
2. Approach the camera: State changes to `I see you — hold still…` $\to$ `Identifying…`.
3. If previously registered: Greeted with personalized greeting (*"Welcome back, Akshata!"*), skipping the guest onboarding prompt.

### 5. ⚡ Direct-Answer Flow & Zero Repetitive Greetings
1. Ask multiple campus questions in sequence (e.g., *"Where is the CSE department?"*, followed by *"What are the hostel fees?"*, then *"Tell me about placements"*).
2. **Observe Answers**: Nova dives straight into the verified campus facts without prefixing answers with repetitive introductions (*"Hello! I am Nova"*, *"As Nova..."*, *"Nova here"*), keeping the interaction fast, natural, and conversational.

### 6. 🎭 Personality & Easter Eggs
- Ask: *"Are you real?"* $\to$ *"Real enough to help you find your way around campus! What can I help you with?"*
- Ask: *"Are you human?"* $\to$ *"Not quite — I'm a digital receptionist. But I'll do my best to help you!"*
- Ask: *"Are you a robot?"* $\to$ *"Guilty as charged! But I promise I'm a friendly one."*
- Ask: *"Tell me a joke"* $\to$ *"Why did the student bring a ladder to class? To reach the higher studies!"*
- Say: *"Thank you so much, bye!"* $\to$ Nova delivers a personalized farewell and transitions cleanly to the goodbye screen.

### 7. 📱 Testing the QR Companion & Mobile Brochure
1. **Start a Session**: Approach the kiosk camera. Nova greets you and automatically renders the **📱 Phone Companion QR Card** in the **top-left corner of the screen** next to Nova.
2. **Scan with Phone (Hands-Free)**:
   - Point your smartphone camera directly at the QR code card in the top-left corner to open the link (e.g. `http://<your-ip>:8001/companion/<token>`).
   - Or speak naturally to Nova: *"Where is the QR code?"* or *"How do I get the brochure on my phone?"* $\to$ Nova vocally responds and directs you to the QR code on the screen.
   - Note: The companion portal is designed for mobile — the kiosk app itself is **blocked on phones**.
3. **Inspect Mobile Portal**:
   - The page displays your name, an AI-generated session recap summarizing the questions you asked Nova, and useful campus quick links.
   - Click **"Download Official Brochure (PDF)"**: It downloads the relevant official brochure (e.g., Computer Science, Admissions, Hostel, or General RNSIT brochure).
4. **Mobile Blocking Test**:
   - Opening `http://localhost:3000` on a phone shows a blocking page: *"Kiosk Not Available on Mobile"*.
   - The companion link `http://<ip>:8001/companion/<token>` works perfectly on phones.
5. **Session Revocation Test**:
   - Say *"Goodbye"* at the kiosk to conclude the session.
   - Refresh the companion page on your phone → the token is immediately invalidated, displaying the clean **"Session Expired"** card.

### 8. 🛎️ Testing Human Handover & Front Desk Escalation
1. **Open Staff Dashboard** (staff only — separate device from the kiosk):
   - On a front-desk laptop or tablet, open a browser and navigate to:
     ```
     http://127.0.0.1:8001/staff
     ```
     (If on a different machine on the same network: `http://<kiosk-LAN-IP>:8001/staff`)
   - Log in using staff credentials:
     - Username: `staff` · Password: `rnsit2024`
     - *or Admin:* Username: `admin` · Password: `111111`
   - Note the top status bar: *"Live · WebSocket connected"*.
2. **Trigger Escalation from Kiosk — Voice Only**:
   - **Voice (primary)**: Say *"I need to speak to a real person"* or *"Connect me to human staff"* or *"Talk to someone"*.
   - **Emergency**: Say *"There is a medical emergency"* — this auto-escalates immediately.
3. **Observe Kiosk Feedback**:
   - The kiosk displays the glassmorphic overlay: *"Connecting to Front Desk — A front desk reception team member has been alerted..."* with an animated connection spinner.
4. **Accept Request on Staff Dashboard**:
   - The staff dashboard flashes a red banner with the visitor's name, reason, and recent chat history.
   - Click the green **"Accept"** button.
   - The kiosk screen immediately transitions to: *"🤝 Staff Member Connected!"*.
5. **Resolve the Escalation**:
   - On the staff dashboard, click **"Resolve"**.
   - The kiosk overlay dismisses, and Nova resumes: *"Your query has been addressed. Nova is ready to help you further."*
6. **Timeout Auto-Revert Test**:
   - Trigger an escalation and do not accept it on the staff dashboard.
   - After 60 seconds, the kiosk displays: *"⏳ Staff Currently Occupied — Nova will continue helping you"* and automatically reverts to bot mode.

---

## 👥 Project Team

**RNS Institute of Technology — VRK Project Team**
- **Alankrita Singh**
- **Akshatha A**
- **B Sneha**
