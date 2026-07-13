# VRK Kiosk — Backend

FastAPI server: speech, answers, sessions, faces. Started automatically by
`run.py` from the repository root — run individually only when debugging:

    venv\Scripts\python.exe -m uvicorn backend.main:app --reload
    venv\Scripts\python.exe -m backend.detection

## Module map (and owners)

| File | Does |
|---|---|
| `main.py` | All API routes, session lifecycle, WebSocket broadcast | shared — coordinate edits |
| `stt.py` | faster-whisper STT, GPU/CPU auto-detect |
| `audio_processing.py` | bandpass + energy gate DSP (noise handling) |
| `tts.py` | Kokoro TTS: cache, silence trim, punctuation normalize |
| `llm.py` | RAG answer engine (Gemini API + Mongo KB + Redis cache) |
| `detection.py` | Camera loop: face recognition → session triggers |
| `database.py` | Async MongoDB layer (motor) |
| `session.py` | DEAD CODE — PostgreSQL era, imports nothing that exists; safe to delete |

## Environment

Python **3.12 only** (kokoro constraint). Dependencies: `requirements.txt`
in this folder — read its header comments before adding anything; it
documents why sub-dependencies must not be pinned.

Config via `.env` at the repository root (`MONGO_URI`, `LLM_API_KEY`,
`TTS_VOICE`, `STT_DEVICE`, ...) — see `.env.example`.