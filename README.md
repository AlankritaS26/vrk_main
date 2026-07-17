# VRK — Voice Receptionist Kiosk

An AI-powered digital receptionist kiosk for RNS Institute of Technology.
Visitors walk up, are recognized (or greeted and registered) by camera, and
hold a natural voice conversation — self-hosted, zero-cost stack
(free-tier APIs only).

## Quick start — one command

```powershell
venv\Scripts\python.exe run.py
```

That single command, in one terminal:
1. Frees ports 8000/3000 if a stale kiosk process is squatting on them
2. Starts the **backend** (FastAPI) and waits for `/health` to pass
3. Starts **camera detection** (only after the backend is ready)
4. Starts the **frontend** (npm) and opens the kiosk browser window with
   autoplay enabled — required for the greeting to speak unprompted
5. Prefixes all logs by service: `[BACKEND]` `[DETECT]` `[FRONTEND]`
6. `Ctrl+C` stops everything, including npm's child processes

Do **not** open localhost:3000 in your own tab for kiosk testing — use the
window `run.py` opens; it carries the `--autoplay-policy` flag that lets the
kiosk speak before any user gesture.

First-time setup (fresh machine): see **docs/SETUP.md**. Summary: Python
3.12 venv → `pip install -r backend/requirements.txt` → `.env` from
`.env.example` (Mongo URI, API key) → `cd frontend && npm install`.

## What it does

- **Kiosk-initiated conversation** — detection spots a visitor and the kiosk
  greets them by voice first (greeting audio is pre-cached per visitor for a
  sub-second start). Returning visitors are greeted by name.
- **Speech-to-text** — Silero VAD in the browser segments speech;
  raw 16 kHz PCM streams to faster-whisper (`small.en` int8 on CPU dev,
  `large-v3-turbo` CUDA in production) with a bandpass + energy-gate DSP
  chain for lobby noise.
- **Answer engine** — guardrails → RAG over the college knowledge base
  (Gemini embeddings + generation, MongoDB source, Redis hot cache). The
  query-condense step is skipped on early turns to save a network round-trip.
- **Text-to-speech** — Kokoro-82M (`af_bella`, 1.05× pace) with:
  silence-trimmed clips, ~2-sentence chunking, two-chunk prefetch, gapless
  Web Audio playback, per-sentence response cache, punctuation
  normalization (curly quotes/dashes), and browser-voice fallback so the
  kiosk never goes mute.
- **Face recognition** — MediaPipe + DeepFace with consent flow, guest mode,
  and a GDPR-style **Delete My Data** flow (see Known Issues).
- **UI** — idle attract screen (live clock, capability carousel, watching
  radar), conversation screen with a persistent voice dock (waveform while
  listening, breathing status dot otherwise, thinking dots while the LLM
  works), synced text+voice bubbles, instant goodbye transition with the
  farewell voice playing over the goodbye screen.

## Repository layout

```
run.py       ← start everything (the only command you normally need)
backend/     FastAPI: STT, TTS, RAG/LLM, sessions, faces, detection
frontend/    React kiosk UI (CRA)
data/        Knowledge base source files
docs/        SETUP.md · ARCHITECTURE.md · OPERATIONS.md
```

## Key endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness (used by run.py) |
| `POST /stt/pcm` | raw PCM → transcript (primary STT path) |
| `POST /tts` | text → base64 WAV (Kokoro, cached) |
| `GET /ask` | question → grounded answer |
| `POST /session/start` · `/session/end` · `GET /session/current` | session lifecycle |
| `POST /visitor/unknown` · `/visitor/submit_name` · `/visitor/delete_my_data` | consent + privacy |

## Configuration (.env)

`MONGO_URI`, `LLM_API_KEY` (Gemini free tier), `ALLOWED_ORIGINS`,
`STT_DEVICE=auto`, `TTS_VOICE=af_bella`, `TTS_SPEED=1.05`.
Frontend: `frontend/.env → REACT_APP_BACKEND_URL`.

## Known issues (tracked)

- **[HIGH — privacy] detection.py stale face cache**: after Delete My Data,
  the visitor is still recognized until detection restarts. Cause: encodings
  load once at startup; the backend's `cache_reload` broadcast isn't
  consumed. Fix assigned to the detection owner (re-fetch `/faces/all` on
  the event, or every 30 s).
- **Detection-to-greeting latency (~2–3 s)**: frontend polling is now 750 ms;
  the remainder is detection.py's recognition cadence (assigned).
- **Conversation latency floor (~2–4 s/turn)**: bounded by Gemini API
  round-trips; next levers are the model tier and embedding cache (LLM owner).

## Team

RNSIT · VRK Kiosk - Alankrita Singh, Akshatha A and B Sneha