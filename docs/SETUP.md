# VRK Kiosk — Complete Setup Guide

This document takes a fresh machine to a fully running kiosk. It exists
because environment problems (wrong Python, copied venvs, split pip/python)
have historically cost this project days — follow it exactly and none of
them can recur.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.12.x** (not 3.13/3.14) | kokoro requires <3.13. Check: `py -0` on Windows |
| Node.js | 18+ | for the React frontend |
| MongoDB | Atlas cluster or local | connection string goes in `.env` |
| Redis | Memurai (Windows) / redis-server | answer caching; kiosk runs without it, degraded |
| Webcam + mic | any | camera for detection, mic for STT |
| GPU (production only) | NVIDIA + CUDA | dev machines run CPU automatically |

## 2. Backend setup

**Rule zero: never copy a venv between folders or machines.** Venvs embed
absolute paths at creation. The manifest (`backend/requirements.txt`)
travels; the venv is rebuilt everywhere in one minute.

From the repository root:

```powershell
# Windows — force Python 3.12 explicitly
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

```bash
# Linux / macOS
python3.12 -m venv venv
venv/bin/python -m pip install -r backend/requirements.txt
```

Always invoke tools through the venv's interpreter —
`venv\Scripts\python.exe -m pip ...`, `venv\Scripts\python.exe -m uvicorn ...` —
so pip and python can never disagree about which environment is in use.

The install is large (~2 GB: tensorflow via deepface, torch via kokoro).
The final "Installing collected packages" step shows no progress for
10–25 minutes on a laptop — it is working; adding a Windows Defender
exclusion for the project folder roughly halves this.

## 3. Configuration

```powershell
copy .env.example .env    # then edit
```

| Variable | Purpose | Example |
|---|---|---|
| `MONGO_URI` | MongoDB connection string | `mongodb+srv://user:pass@cluster.../` |
| `ALLOWED_ORIGINS` | CORS for the kiosk frontend | `http://localhost:3000` |
| `STT_DEVICE` | `auto` (default) / `cuda` / `cpu` | leave `auto` |
| `STT_MODEL` | override Whisper model | `large-v3-turbo` (GPU) / `small.en` (CPU) |
| `TTS_VOICE` | Kokoro voice | `af_bella`, `am_michael`, `bf_emma`… |
| `TTS_SPEED` | speaking pace | `1.05` (1.0–1.15 natural) |
| `REACT_APP_BACKEND_URL` | frontend → backend URL (frontend/.env) | `http://127.0.0.1:8000` |

## 4. Run

Normal operation: `venv\Scripts\python.exe run.py` starts everything — the steps below are for running services individually while debugging.

```powershell
# Terminal 1 — backend (from repo root)
venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Healthy boot log, in order:
```
[STT] Loading <model> on <device> ...
[STT] Warmup complete.
[TTS] Kokoro available.
[REDIS] Connected ...            (warning here = Redis not running, non-fatal)
[SYSTEM] RAG vector cache loaded successfully.
Application startup complete.
[TTS] Kokoro warmed up.
```

First-ever boot downloads models once (Whisper ~500MB–1.6GB, Kokoro ~330MB
+ voice file); all cached afterwards.

```powershell
# Terminal 2 — frontend
cd frontend
npm install       # first time only
npm start         # prestart hook copies VAD assets into public/ automatically
```

**Mic rule:** browsers only grant microphone access on `https://` or
`localhost`. Always serve the frontend on the kiosk machine itself
(`localhost:3000`); point `REACT_APP_BACKEND_URL` at the backend machine's
IP when they differ.

## 5. Production (GPU server + kiosk machine)

```powershell
# GPU server — verify CUDA is visible, then bind to the LAN
venv\Scripts\python.exe -c "import ctranslate2; print('GPUs:', ctranslate2.get_cuda_device_count())"
venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Boot log must read `Loading large-v3-turbo on cuda (float16)`. Open port
8000 in the firewall. On the kiosk machine set
`frontend/.env → REACT_APP_BACKEND_URL=http://<GPU_SERVER_IP>:8000`,
build (`npm run build`) and serve the build locally.

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

## 7. Dependency policy

`backend/requirements.txt` lists **only what the code imports** — pip
resolves the ~150 sub-dependencies automatically. When you add an import,
add its package here in the same commit. To snapshot exact versions for a
new deployment target: `venv\Scripts\python.exe -m pip freeze > requirements.lock.txt`.
