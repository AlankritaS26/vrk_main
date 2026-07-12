# VRK Kiosk — Architecture

## System overview

```
KIOSK MACHINE                                   BACKEND SERVER (GPU in prod)
┌───────────────────────────────┐              ┌────────────────────────────────┐
│ React app (localhost:3000)    │              │ FastAPI (uvicorn, async)       │
│                               │              │                                │
│ camera ─► detection.py loop ──┼── HTTP ────► │ /session/start /visitor/*      │
│ (face recognize / register)   │              │                                │
│                               │              │ /stt/pcm                       │
│ mic ► Silero VAD (in browser) ┼── PCM ─────► │  int16→float32 ► bandpass      │
│  one utterance = one POST     │              │  ► energy gate ► faster-whisper│
│                               │ ◄── JSON ─── │  {text, confidence, latency}   │
│ chat UI + waveform            │              │                                │
│                               │              │ /ask ► guardrails ► FAQ (Mongo)│
│ speaker ◄ sentence-pipelined ─┼── WAV ◄───── │  ► Redis cache ► local LLM RAG │
│  Kokoro audio (/tts)          │              │                                │
└───────────────────────────────┘              │ MongoDB (motor) · Redis        │
                                               └────────────────────────────────┘
```

Both machines share a LAN; only bytes travel. The mic and camera live on
the kiosk; all inference lives on the server. In development, both halves
run on one laptop unchanged (STT auto-falls back to a CPU model).

## Conversation flow

1. **Detection** (`backend/detection.py`): camera loop recognizes a face
   → `/session/start` (returning visitor, greeted by name) or
   `/visitor/unknown` (name/consent modal, then registration).
2. **Kiosk speaks first**: on session start the frontend speaks the
   greeting via TTS — the visitor never initiates.
3. **Listening**: browser VAD (threshold 0.8, ~800 ms end-of-speech)
   captures one utterance as 16 kHz PCM → `POST /stt/pcm`.
4. **STT** (`backend/stt.py`): DSP chain (bandpass 80 Hz–7.5 kHz → RMS
   energy gate) → faster-whisper with campus-vocabulary prompt bias.
   Runs in a worker thread; the event loop is never blocked.
5. **Answering** (`/ask`): input-safety guardrail → typo normalization →
   MongoDB FAQ match → Redis cached answer → local LLM RAG fallback.
6. **TTS** (`backend/tts.py`): Kokoro-82M; per-sentence synthesis with an
   in-memory cache. The frontend pipelines playback (sentence N plays
   while N+1 synthesizes) and prints each answer in sync with the voice.
7. **Turn-taking**: the VAD pauses while the kiosk speaks (plus browser
   echo cancellation); speech captured during processing is queued and
   automatically handled as the next turn.

## Latency budget (speech end → first audio of reply)

| Stage | CPU dev | GPU prod |
|---|---|---|
| VAD end-of-speech | ~300 ms | ~300 ms |
| STT | 1–2 s (small.en int8) | ~300 ms (large-v3-turbo fp16) |
| Answer (FAQ/cache hit) | <100 ms | <100 ms |
| TTS first sentence | ~1 s (cached: ~0) | <300 ms |
| **Total** | **~2.5–3.5 s** | **~1 s** |

## Noise & single-speaker handling (EP-06)

Six layers: browser constraints (noise suppression, echo cancellation,
AGC off) → neural VAD gate (0.8) → bandpass filter → RMS energy gate
(rejects voices not at the kiosk) → Whisper-level VAD + vocabulary bias →
[hardware] beamforming mic array. Layers 1–5 ship in this codebase; the
energy-gate threshold requires one 10-minute on-site tuning pass
(docs/OPERATIONS.md §5). Two equally close, equally loud simultaneous
speakers are out of software scope by design — handled by kiosk placement
and the beamforming mic.

## Key design decisions

- **Raw PCM over WebM**: eliminates ffmpeg, temp files, and ~500 ms per
  turn; the browser VAD already produces clean 16 kHz float PCM.
- **CPU/GPU auto-detection**: one codebase; `STT_DEVICE=auto` picks CUDA
  when present, `small.en` int8 otherwise — dev laptops need no config.
- **Graceful TTS degradation**: any Kokoro failure returns empty audio and
  the frontend falls back to the browser voice; the kiosk never goes mute.
- **Async-first backend**: Mongo via motor, blocking inference (Whisper,
  Kokoro, DeepFace) confined to `asyncio.to_thread`.
- **In-memory session + Mongo persistence**: the active session is
  in-process for speed; sessions/interactions/faces persist to MongoDB.
