# VRK Kiosk — Backend Microservice & Vision Pipeline

FastAPI server powering the **Voice Receptionist Kiosk (VRK)** for RNS Institute of Technology. Provides real-time speech-to-text, LLM & RAG retrieval, ArcFace face recognition, blink detection, text-to-speech generation, session management, and admin data telemetry.

Started automatically by `python run.py` from the repository root (runs on **port 8001** in parallel with the RAG microservice on **port 8600** and React frontend on **port 3000**).

---

## 🚀 Running Standalone (Development / Debugging)

To run the backend independently:

```powershell
# From the repository root with venv activated:
venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

To test the standalone camera detection pipeline:

```powershell
venv\Scripts\python.exe -m backend.detection
```

---

## 🏗️ Architecture & Component Overview

```
                          ┌────────────────────────────┐
                          │     React Frontend (3000)  │
                          └──────────┬──────▲──────────┘
                         WebSocket   │      │ HTTP REST
                        /ws /ws/detect      │ /ask, /tts, /stt/pcm, etc.
                                     ▼      │
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend (Port 8001)                     │
│                                                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │   detection.py   │  │ recognition.py   │  │      stt.py          │  │
│  │ MediaPipe Vision │  │ ArcFace + SCRFD  │  │ faster-whisper (STT) │  │
│  │ (Blink, Presence)│  │ 512-d Embeddings │  │ Bandpass DSP Gate    │  │
│  └─────────┬────────┘  └────────┬─────────┘  └──────────┬───────────┘  │
│            │                    │                       │              │
│  ┌─────────▼────────────────────▼───────────────────────▼───────────┐  │
│  │                        main.py (Core Router)                     │  │
│  │    Session State Machine • Safety Guardrails • Intent Router     │  │
│  └─────────┬────────────────────┬───────────────────────┬───────────┘  │
│            │                    │                       │              │
│  ┌─────────▼────────┐  ┌────────▼─────────┐  ┌──────────▼───────────┐  │
│  │      llm.py      │  │      tts.py      │  │     database.py      │  │
│  │ Primary: Qwen    │  │ Kokoro-82M TTS   │  │ Motor (Async MongoDB)│  │
│  │ Fallback: Gemini │  │ Audio Normalizer │  │ Sessions & Encodes   │  │
│  └─────────┬────────┘  └──────────────────┘  └──────────────────────┘  │
└────────────┼───────────────────────────────────────────────────────────┘
             │ HTTP Search Query (/v1/collections/search)
             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   RAGService Microservice (Port 8600)                  │
│       ChromaDB Vector Store • Sentence-Transformers Embeddings         │
│          Campus Documents, FAQs & College Knowledge Corpus             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Module Reference & Responsibilities

| File | Primary Responsibility | Key Libraries / Technologies |
|---|---|---|
| `main.py` | FastAPI application, REST endpoints, WebSocket hubs (`/ws`, `/ws/detect`, `/ws/stt`), session lifecycle, Easter eggs & deterministic responses, admin CRUD & web dashboard. | `FastAPI`, `Uvicorn`, `WebSockets`, `httpx`, `redis` |
| `detection.py` | MediaPipe FaceLandmarker (468-point mesh), Eye Aspect Ratio (EAR) blink detection, visitor presence state machine (`IDLE` $\to$ `DWELLING` $\to$ `RECOGNIZING` $\to$ `ACTIVE`). | `mediapipe`, `opencv-python`, `numpy` |
| `recognition.py` | SCRFD face detector + ArcFace `w600k_r50.onnx` 512-d L2-normalized embeddings with 5-point canonical alignment, cosine similarity matching against registered MongoDB face encodings. | `onnxruntime`, `opencv-python`, `numpy` |
| `stt.py` | Audio transcription, Silero VAD segment processing, 16kHz PCM ingestion, Whisper hallucination suppression and campus vocabulary biasing. | `faster-whisper`, `torch`, `torchaudio` |
| `audio_processing.py`| Digital Signal Processing (DSP) chain: bandpass filter (100Hz–7500Hz), noise floor suppression, energy gating for loud lobby environments. | `scipy`, `numpy` |
| `tts.py` | High-fidelity neural voice synthesis (`af_bella`), sentence chunking, gapless prefetch cache, silence trimming, punctuation & dash normalization. | `kokoro`, `soundfile`, `pydub` |
| `llm.py` & `gemini.py` | Qwen LLM as primary generation engine with automatic Gemini Flash fallback, multi-turn conversational history management, topic extraction, and prompt injection filtering. | `httpx`, `google-genai`, `google-generativeai` |
| `database.py` | Asynchronous MongoDB operations using Motor: visitor identity logs, interaction history, session transcripts, face vector storage with TTL index support. | `motor`, `pymongo`, `bson` |
| `face_landmarker.task` | Pre-trained MediaPipe landmark model binary for fast on-device inference without GPU overhead. | MediaPipe Task Asset |

---

## 🔌 API Endpoint Reference

### 1. Health & Diagnostics
- `GET /health` — Liveness and readiness check used by `run.py` orchestrator.
- `GET /` — API root service status metadata.
- `GET /logs-dashboard` — Secured administrative dashboard for monitoring visitor interactions and sessions.

### 2. Speech & Voice Pipeline
- `POST /stt/pcm` — Accepts raw 16 kHz 16-bit mono PCM audio from frontend VAD and returns transcribed text.
- `POST /tts` — Accepts text JSON `{"text": "..."}` and returns base64-encoded WAV audio (cached in-memory for instant playback).
- `GET /ask` — Primary question-answering route: validates query safety, performs RAG retrieval, runs Qwen/Gemini response generation, and attaches personality/context.
- `GET /ask/stream` — Server-Sent Events (SSE) stream for real-time word-by-word LLM token generation.
- `POST /api/chat` — Direct diagnostic route to test RAG retrieval + answer generation.

### 3. Session & Interaction State
- `POST /session/start` — Initializes a new visitor session (returns `session_id`, visitor greeting, and visit history).
- `POST /session/end` — Terminates an active session and computes duration.
- `POST /session/are_you_there` — Keeps active sessions alive or initiates re-engagement prompts.
- `GET /session/current` — Returns the metadata of the currently active kiosk session.
- `GET /session/messages/{session_id}` — Retrieves the full transcript of messages for a given session.
- `POST /message` — Logs an interaction turn (user question + kiosk reply).

### 4. Visitor Identity & Face Registration
- `POST /visitor/unknown` — Triggered when a new, unregistered face is detected at the kiosk.
- `POST /visitor/submit_name` — Submits the visitor's voice-transcribed or confirmed name, saving face encoding to MongoDB.
- `POST /visitor/rename` — Updates an existing visitor's registered name.
- `GET /visitor/name_response` — Polls status of visitor name confirmation.
- `POST /visitor/clear_response` — Clears transient name capture state.
- `GET /faces/all` — Retrieves all stored face encodings for detection caching.
- `POST /faces/register` — Direct administrative endpoint to register a new face vector.
- `POST /faces/visit` — Increments visit count and timestamps for recognized individuals.

### 5. WebSockets
- `WebSocket /ws` — General kiosk event hub (broadcasts session state changes, UI transitions, audio playback triggers).
- `WebSocket /ws/detect` — High-frequency vision stream: broadcasts real-time presence, visitor identity, bounding boxes, and blink flags (`blink_detected`, `eyes_closed`, `ear_left`, `ear_right`).
- `WebSocket /ws/stt` — Streaming speech transcription socket for real-time live captioning.

### 6. Admin & Knowledge Management (RAG)
- `POST /api/rag/upload` — Ingests PDF/DOCX/TXT files into RAGService vector collection.
- `GET /api/rag/files` — Lists all indexed documents in the active knowledge base.
- `GET /api/rag/stats` — Returns chunk count, vector dimensions, and collection status.
- `DELETE /api/rag/files/{filename}` — Removes document vectors from the RAG store.
- `DELETE /api/admin/interactions/{session_id}` — Deletes specific interaction logs.
- `DELETE /api/admin/faces/{face_id}` — Removes a face record.
- `PUT /api/admin/faces/{face_id}` — Modifies stored face metadata.
- `DELETE /api/admin/sessions/{session_id}` — Removes session records.
- `DELETE /api/admin/clear-all` — Administrative database reset.