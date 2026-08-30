# VRK Kiosk — Complete Setup Guide

This document takes a fresh machine to a fully running kiosk. It exists because environment problems (wrong Python, copied venvs, split pip/python) have historically cost this project days — follow it exactly and none of them can recur.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.12.x** (not 3.13/3.14) | Kokoro requires <3.13. Check: `py -0` on Windows |
| Node.js | 18+ | for the React frontend |
| MongoDB | Atlas cluster or local | connection string goes in `.env` |
| Redis | Memurai (Windows) / redis-server | answer caching; kiosk runs without it, degraded |
| Webcam + mic | any | camera for ArcFace/MediaPipe detection, mic for STT |
| GPU (production only) | NVIDIA + CUDA | dev machines run CPU automatically |

---

## 2. Backend & Microservices Setup

**Rule zero: never copy a venv between folders or machines.** Venvs embed absolute paths at creation. The manifests (`backend/requirements.txt` and `RAGService/requirements.txt`) travel; the venv is rebuilt everywhere in one minute.

From the repository root:

```powershell
# Windows — force Python 3.12 explicitly
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r backend\requirements.txt
venv\Scripts\python.exe -m pip install -r RAGService\requirements.txt
```

```bash
# Linux / macOS
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r backend/requirements.txt
venv/bin/python -m pip install -r RAGService/requirements.txt
```

Always invoke tools through the venv's interpreter —
`venv\Scripts\python.exe -m pip ...`, `venv\Scripts\python.exe -m uvicorn ...` — so pip and python can never disagree about which environment is in use.

---

## 3. Configuration

Create `.env` at the repository root:

```powershell
copy .env.example .env    # then edit
```

| Variable | Purpose | Example |
|---|---|---|
| `MONGO_URI` | MongoDB connection string | `mongodb+srv://user:pass@cluster.../` |
| `LLM_API_KEY` / `GEMINI_API_KEY` | LLM API key | `your-gemini-or-qwen-key` |
| `ALLOWED_ORIGINS` | CORS for the kiosk frontend | `http://localhost:3000` |
| `BACKEND_URL` | FastAPI backend URL | `http://127.0.0.1:8001` |
| `RAG_SERVICE_URL` | RAG microservice URL | `http://127.0.0.1:8600` |
| `STT_DEVICE` | `auto` (default) / `cuda` / `cpu` | leave `auto` |
| `STT_MODEL` | override Whisper model | `large-v3-turbo` (GPU) / `small.en` (CPU) |
| `TTS_VOICE` | Kokoro voice | `af_bella`, `am_michael`, `bf_emma`… |
| `TTS_SPEED` | speaking pace | `1.05` (1.0–1.15 natural) |
| `REACT_APP_BACKEND_URL` | frontend → backend URL (frontend/.env) | `http://127.0.0.1:8001` |

---

## 4. Run — One Command

Normal operation: `venv\Scripts\python.exe run.py` starts all three services in parallel:

1. **RAGService** on Port `8600`
2. **FastAPI Backend** on Port `8001`
3. **React Frontend** on Port `3000` with automated autoplay browser launch

### Individual Debugging Commands:

```powershell
# Terminal 1 — RAG Microservice (from RAGService folder)
cd RAGService
..\venv\Scripts\python.exe -m uvicorn app:app --port 8600

# Terminal 2 — Backend (from repo root)
venv\Scripts\python.exe -m uvicorn backend.main:app --port 8001 --reload

# Terminal 3 — Frontend (from frontend folder)
cd frontend
npm install       # first time only
npm start         # prestart hook copies VAD assets into public/ automatically
```

**Mic rule:** browsers only grant microphone access on `https://` or `localhost`. Always serve the frontend on the kiosk machine itself (`localhost:3000`); point `REACT_APP_BACKEND_URL` at the backend machine's IP when they differ.

---

## 5. Production (GPU server + kiosk machine)

```powershell
# GPU server — verify CUDA is visible, then bind to the LAN
venv\Scripts\python.exe -c "import ctranslate2; print('GPUs:', ctranslate2.get_cuda_device_count())"
venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8001
```
Boot log must read `Loading large-v3-turbo on cuda (float16)`. Open port 8001 in the firewall. On the kiosk machine set `frontend/.env → REACT_APP_BACKEND_URL=http://<GPU_SERVER_IP>:8001`, build (`npm run build`) and serve the build locally.

---

## 6. Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| `pip show` finds a package but `import` fails | pip and python point at different interpreters → always use `venv\Scripts\python.exe -m pip` |
| `Could not find a version ... kokoro` + `Requires-Python` wall | venv built on Python 3.13/3.14 → rebuild with `py -3.12 -m venv venv` |
| Can't delete venv: `.pyd Access denied` | a Python process still holds it → stop uvicorn / `taskkill /F /IM python.exe`, then delete |
| `[TTS] Kokoro import failed: <reason>` | read the reason; usually wrong env or missing `misaki[en]` |
| 404 on `silero_vad*.onnx` / `*.wasm` | VAD assets missing → `npm run copy-vad` (auto on `npm start`) |
| `react-scripts not recognized` | no node_modules → `npm install` (node_modules, like venvs, is never copied) |
| Mic permission never appears | frontend not on localhost/HTTPS |
| Kiosk transcribes its own voice | use Chrome; echo-cancellation + VAD pause are built in |
| Boot shows `on cpu` on the GPU server | CUDA not visible → `nvidia-smi`, reinstall drivers |
| Everything rejected as `too_quiet` | energy gate too strict → see docs/OPERATIONS.md tuning |
