# VRK Kiosk — Operations Handbook
### Voice Receptionist Kiosk @ RNSIT · STT/TTS & Vision Operations

This handbook covers running the system across development, latency profiling, and production deployment.

---

## 1. Architecture

```
KIOSK MACHINE (frontend)                     SERVER (backend + RAG)
┌────────────────────────────┐              ┌──────────────────────────────────────────┐
│ React app on localhost:3000│              │ FastAPI Backend on Port 8001             │
│                            │              │ RAGService on Port 8600                  │
│ mic → getUserMedia         │  POST        │                                          │
│ → Silero VAD (in browser)  │ ───────────► │ /stt/pcm                                 │
│ → one utterance as         │  raw PCM     │ int16 → float32 → bandpass               │
│   Int16 PCM @ 16 kHz       │  ~96 KB/3 s  │ → energy gate → Whisper                  │
│                            │ ◄─────────── │   large-v3-turbo (CUDA) / small.en (CPU) │
│ → /ask → answer text       │              │                                          │
│ → /tts → Kokoro WAV        │              │ /tts → Kokoro-82M ('af_bella')           │
│ → speaker plays audio      │              │                                          │
└────────────────────────────┘              └──────────────────────────────────────────┘
```

---

## 2. Prerequisites

**Backend Machine**:
- Python 3.12 (`py -3.12`)
- Dependencies: `pip install -r backend/requirements.txt -r RAGService/requirements.txt`
- Production: NVIDIA GPU + CUDA drivers (`nvidia-smi` must work)
- Redis / Memurai running on `localhost:6379`
- MongoDB Atlas cluster URI in `.env`

**Kiosk Machine**:
- Node.js 18+, Google Chrome or Microsoft Edge
- USB Microphone and HD Webcam

---

## 3. Running Modes

### Mode A — Master Orchestration (Recommended)

```powershell
venv\Scripts\python.exe run.py
```
Automatically launches RAGService (port 8600), Backend (port 8001), and Frontend (port 3000) with autoplay permissions.

### Mode B — Manual Service Launch (Debugging)

```powershell
# Terminal 1 — RAG Microservice
cd RAGService
..\venv\Scripts\python.exe -m uvicorn app:app --port 8600

# Terminal 2 — Backend
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload

# Terminal 3 — Frontend
cd frontend
npm start
```

---

## 4. Production Server Deployment

```powershell
# Verify GPU availability:
venv\Scripts\python.exe -c "import ctranslate2; print('GPUs:', ctranslate2.get_cuda_device_count())"

# Launch Backend bound to all interfaces:
venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

On the kiosk terminal:
```powershell
cd frontend
echo REACT_APP_BACKEND_URL=http://<SERVER_IP>:8001 > .env
npm run build
```
Serve the built frontend bundle using any static web server (e.g. `npx serve -s build -l 3000`).
