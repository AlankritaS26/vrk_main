# VRK Kiosk — Architecture

## System Overview

```
KIOSK CLIENT (localhost:3000)                   BACKEND SERVER (GPU / CPU) (Port 8001)
┌────────────────────────────────┐              ┌────────────────────────────────────────┐
│ React 18 App                   │              │ FastAPI Backend (uvicorn, async)       │
│                                │              │                                        │
│ camera ─► MediaPipe/ArcFace ───┼── WebSocket ─► /ws/detect (presence, blink, identity) │
│ (SCRFD + ArcFace 512-d embed)  │   /ws        │                                        │
│                                │              │ /stt/pcm                               │
│ mic ► Silero VAD (in browser) ─┼── PCM ──────►│  int16→float32 ► bandpass filter       │
│  one utterance = one POST      │              │  ► energy gate ► faster-whisper (STT)  │
│                                │ ◄── JSON ────│  {text, confidence, latency}           │
│ chat UI + Avatar + waveform    │              │                                        │
│                                │              │ /ask ► safety ► Qwen LLM / Gemini      │
│ speaker ◄ sentence-pipelined ──┼── WAV ◄──────│  ► RAGService microservice (Port 8600) │
│  Kokoro audio (/tts)           │              │  ► Redis cache ► MongoDB               │
└────────────────────────────────┘              └────────────────────────────────────────┘
```

Both machines share a LAN; only bytes travel. The mic and camera live on
the kiosk; all inference lives on the server. In development, all services
run together via `python run.py`.

---

## Conversation & Vision Flow

1. **Detection & Recognition** (`backend/detection.py` + `backend/recognition.py`):
   - MediaPipe face mesh tracks landmarks and calculates Eye Aspect Ratio (EAR) for blink detection.
   - SCRFD + ArcFace (`w600k_r50.onnx`) extracts 512-d facial embeddings.
   - Cosine similarity matching against MongoDB registers new visitors or identifies returning visitors by name.
2. **Kiosk Speaks First**: On session start, the frontend speaks the personalized greeting via Kokoro TTS — the visitor never needs to click.
3. **Hands-Free Name Onboarding**: For new visitors, Nova asks *"What's your name?"*. Spoken names are transcribed, confirmed, and automatically registered after 3 seconds.
4. **Listening**: Browser VAD (Silero ONNX runtime) captures speech as 16 kHz raw PCM $\to$ `POST /stt/pcm`.
5. **STT** (`backend/stt.py`): Bandpass DSP chain (80 Hz–7.5 kHz) + energy gate $\to$ `faster-whisper` with campus vocabulary prompting.
6. **Answering** (`/ask`): Safety guardrail $\to$ Easter egg checks $\to$ RAGService semantic retrieval on Port 8600 $\to$ Qwen primary generation (with Gemini fallback).
7. **TTS** (`backend/tts.py`): Kokoro-82M synthesis (`af_bella`), sentence chunk prefetching, Web Audio gapless playback, and synchronized text animation.

---

## Latency Budget (Speech End → First Audio of Reply)

| Stage | CPU Dev | GPU Prod |
|---|---|---|
| VAD end-of-speech | ~300 ms | ~300 ms |
| STT (faster-whisper) | 1–2 s (small.en) | ~300 ms (large-v3-turbo) |
| Answer (RAG + Qwen LLM) | 0.5–1.5 s | 0.2–0.8 s |
| TTS First Chunk (Kokoro) | ~400 ms (cached: ~0) | <200 ms |
| **Total Turn Latency** | **~2.0–3.5 s** | **~1.0–1.6 s** |

---

## Noise & Single-Speaker Handling

1. **Browser Layer**: WebRTC constraints (noise suppression, echo cancellation, automatic gain control).
2. **Neural VAD Gate**: Silero VAD (0.8 threshold) isolates real speech from background chatter.
3. **DSP Bandpass Filter**: Restricts frequency spectrum between 80 Hz and 7.5 kHz.
4. **RMS Energy Gate**: Rejects distant or quiet ambient voices.
5. **Whisper Prompt Biasing**: Domain vocabulary suppresses hallucinations on Indian campus names and terms.
