# VRK - Voice Receptionist Kiosk (Nova)

An AI-powered digital receptionist kiosk for **RNS Institute of Technology**.
Visitors walk up to the kiosk, are recognized or welcomed as guests, and have
hands-free voice conversations about the campus.

## Quick start

Use Python 3.12 for the backend environment:

```powershell
venv\Scripts\python.exe run.py
```

`run.py` starts and health-checks:

- **RAGService** on port `8600`
- **FastAPI backend** on port `8001`
- **React kiosk** on port `3000`

It also frees stale development processes on those ports, waits for the
backend and RAG service, opens the kiosk browser with autoplay enabled, and
stops child processes cleanly on `Ctrl+C`.

### Manual service startup

```powershell
# Backend
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001

# RAG service
Set-Location RAGService
venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8600

# Frontend
Set-Location frontend
npm install
npm start
```

Configure MongoDB, Redis/Memurai, Qwen or another OpenAI-compatible local
endpoint, and optional Gemini/TomTom credentials in `.env` as required by the
local deployment. Do not commit credentials.

## Current portals

| Portal | URL | Purpose |
|---|---|---|
| Interactive kiosk | `http://localhost:3000` | Voice-first visitor experience |
| Admin dashboard | `http://127.0.0.1:8001/logs-dashboard` | Protected interaction logs, face records, and administration |
| Mobile companion | `http://<LAN-IP>:8001/companion/<token>` | Session summary and official brochure downloads |

The kiosk requires a webcam and microphone and is intended for a laptop,
desktop, or kiosk terminal. The companion portal is designed for a phone and is
opened by scanning the QR card shown on the kiosk.

## Features

### Voice-first visitor experience

- Automatic visitor detection and voice onboarding
- Guest mode when the visitor does not provide a name
- Name confirmation by voice or double blink
- Letter-by-letter spelling fallback
- Mid-session name changes synchronized with MongoDB
- Barge-in interruption and answer-resume flow
- Farewell and inactivity handling
- Automatic response captions displayed over the kiosk screen
- Say "off captions" or "stop the captions" to hide captions and continue with voice only

### Vision and recognition

- ArcFace/SCRFD face recognition
- MediaPipe landmarks and presence tracking
- Blink and double-blink detection
- IDLE -> DWELLING -> RECOGNIZING -> ACTIVE kiosk state flow

### Five-stage answer routing

`backend/confidence_rag.py` routes each query through the appropriate path:

1. **Weather** - live Open-Meteo weather data
2. **Traffic** - live TomTom traffic data when configured
3. **RNSIT-specific entity** - verified `entity_mapping.py` answer
4. **RNSIT general** - ChromaDB retrieval plus Qwen/Gemini generation
5. **Unsupported query** - receptionist guardrail refusal

The router also corrects common speech-to-text variants, tracks confidence,
records unanswered in-domain questions for the admin workflow, and preserves
ambiguity handling for broad or unclear questions.

### Natural response handling

- Emotional and wellbeing replies receive a short, warm re-engagement response.
- Generic acknowledgement fillers are suppressed for emotional replies.
- Generated answers are cleaned of greetings, self-introductions, and filler
  such as "Let me think" or "Sure, I will find".
- Streaming answers skip empty fragments after response cleanup.

### Audio

- faster-whisper speech-to-text
- Silero VAD in the browser
- Kokoro-82M neural text-to-speech
- Sentence chunking and prefetch for gapless playback
- Browser speech-synthesis fallback

### QR companion portal

When a session starts, the backend issues a short-lived companion token and the
kiosk displays its QR code. The companion page provides:

- A grounded summary of the current visit
- Personalized visitor name when available
- Official RNSIT brochure recommendations
- PDF downloads from `data/brochures/`
- Token expiration and session revocation after the kiosk session ends

No mobile application is required.

## Architecture

```text
Visitor
  |
  v
React kiosk (port 3000)
  |  WebSocket /ws, /ws/detect, /ws/stt
  |  HTTP /stt/pcm, /tts, /ask
  v
FastAPI backend (port 8001)
  |-- Camera detection and face recognition
  |-- STT, TTS, session state, and guardrails
  |-- Five-stage confidence/RAG router
  |-- QR companion and brochure portal
  |-- MongoDB sessions, faces, and interactions
  |-- Redis/Memurai cache and companion tokens
  |
  v
RAGService (port 8600)
  |-- ChromaDB vector store
  |-- Sentence-transformers embeddings
  `-- Campus document ingestion
```

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Backend health and warmup status |
| `/ws` | WebSocket | Kiosk session and event channel |
| `/ws/detect` | WebSocket | Vision stream and blink events |
| `/ws/stt` | WebSocket | Streaming transcription |
| `/stt/pcm` | POST | Transcribe 16 kHz mono PCM |
| `/tts` | POST | Generate speech audio |
| `/ask` | GET | Non-streaming routed answer |
| `/ask/stream` | GET (SSE) | Streaming routed answer |
| `/session/start` | POST | Start a visitor session |
| `/session/end` | POST | End a visitor session |
| `/session/are_you_there` | POST | Check visitor presence |
| `/visitor/unknown` | POST | Start unknown-visitor onboarding |
| `/visitor/submit_name` | POST | Save a visitor name and face encoding |
| `/visitor/rename` | POST | Rename a known visitor |
| `/faces/all` | GET | Load registered face vectors |
| `/api/rag/upload` | POST | Upload and index campus documents |
| `/api/rag/files` | GET | List indexed RAG documents |
| `/logs-dashboard` | GET | Protected admin dashboard |
| `/companion/token` | POST | Issue a session companion token |
| `/companion/{token}` | GET | Render the mobile companion page |
| `/companion/qr/{token}` | GET | Render a companion QR image |
| `/companion/validate/{token}` | GET | Validate a companion token |
| `/companion/brochure/{token}` | GET | Stream a recommended brochure |

## Repository structure

```text
vrk_main/
├── run.py
├── backend/
│   ├── main.py
│   ├── companion.py
│   ├── confidence_rag.py
│   ├── entity_mapping.py
│   ├── query_correction.py
│   ├── detection.py
│   ├── recognition.py
│   ├── stt.py
│   ├── tts.py
│   ├── llm.py
│   ├── gemini.py
│   ├── database.py
│   └── requirements.txt
├── RAGService/
├── frontend/
│   ├── src/App.js
│   ├── src/WelcomeScreen.js
│   ├── src/IdleScreen.js
│   ├── src/GoodbyeScreen.js
│   ├── src/kioskMic.js
│   └── scripts/copyVadAssets.js
├── data/
│   ├── college_info.json
│   └── brochures/
└── docs/
    ├── SETUP.md
    ├── ARCHITECTURE.md
    └── OPERATIONS.md
```

## Verification

### Backend syntax

```powershell
venv\Scripts\python.exe -m py_compile `
  backend\main.py backend\companion.py backend\confidence_rag.py `
  backend\database.py backend\llm.py
```

### Frontend build

```powershell
Set-Location frontend
npm run build
```

### Functional smoke checks

1. Start the stack with `venv\Scripts\python.exe run.py`.
2. Confirm `http://127.0.0.1:8001/health` reports a healthy backend.
3. Confirm the RAG service health endpoint on port `8600`.
4. Ask a weather, traffic, entity, broad RNSIT, and unsupported question.
5. Confirm the QR companion appears during an active session.
6. End the session and confirm the companion token is invalidated.

## Related documentation

- [Backend architecture](backend/README.md)
- [Frontend architecture](frontend/README.md)
- [Setup guide](docs/SETUP.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Operations guide](docs/OPERATIONS.md)
