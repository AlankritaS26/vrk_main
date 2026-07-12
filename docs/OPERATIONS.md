# VRK Kiosk — Operations Handbook
### Voice Receptionist Kiosk @ RNSIT · STT/TTS Pipeline (EP-03, EP-04, EP-06)

This handbook covers running the upgraded project on three setups: your
dev laptop (no GPU), a free cloud GPU for latency testing, and the
production college GPU server. It also documents exactly how the noise
and single-speaker problems are handled — including which parts are
solved in software, which need tuning on-site, and which need hardware.

---

## 1. Architecture (what talks to what)

```
KIOSK MACHINE (frontend)                     GPU SERVER (backend)
┌────────────────────────────┐              ┌──────────────────────────────┐
│ React app on localhost:3000│              │ FastAPI on 0.0.0.0:8000      │
│                            │              │                              │
│ mic → getUserMedia         │  POST        │ /stt/pcm                     │
│ → Silero VAD (in browser)  │ ───────────► │ int16 → float32 → bandpass   │
│ → one utterance as         │  raw PCM     │ → energy gate → Whisper      │
│   Int16 PCM @ 16 kHz       │  ~96 KB/3 s  │   large-v3-turbo (CUDA)      │
│                            │ ◄─────────── │ {text, confidence, latency}  │
│ → /ask → answer text       │              │                              │
│ → /tts → Kokoro WAV        │              │ /tts → Kokoro-82M (CPU)      │
│ → speaker plays audio      │              │                              │
└────────────────────────────┘              └──────────────────────────────┘
```

Key facts:
- The mic is NEVER attached to the server. Only bytes travel over the LAN.
- The browser must load the app from `localhost` (or HTTPS) or the mic is
  blocked by the browser. Serve the frontend ON the kiosk machine.
- One utterance = one HTTP POST. The in-browser VAD decides where speech
  starts and ends — there is no fixed recording window anywhere.

Latency budget (person stops talking → kiosk starts answering):

| Stage                              | Target      |
|------------------------------------|-------------|
| VAD end-of-speech detection        | ~300 ms     |
| PCM transfer over LAN              | ~10 ms      |
| DSP (bandpass + energy gate)       | ~15 ms      |
| Whisper large-v3-turbo on GPU      | ~300 ms     |
| **STT total**                      | **< 0.7 s** |
| LLM answer (FAQ hit / Ollama)      | 0.2 – 2 s   |
| Kokoro TTS first audio             | < 1 s       |

---

## 2. Prerequisites

**Backend machine** (server or your laptop):
- Python 3.10–3.12
- `pip install faster-whisper scipy numpy fastapi uvicorn redis python-dotenv kokoro soundfile`
- Production only: NVIDIA GPU + CUDA drivers (`nvidia-smi` must work)
- Your existing deps (redis server, Ollama, etc.) as before

**Kiosk machine:**
- Node 18+, Chrome/Edge (best VAD + echo-cancellation support)
- A decent USB mic; later the ReSpeaker 4-mic array (see §6)

---

## 3. Running it

### Mode A — everything on your dev laptop (no GPU)

This validates 95% of the system. STT auto-falls back to `small.en` int8
(1–2 s per utterance instead of 0.3 s — functional, just slower).

```bash
# Terminal 1 — backend (from the VRK_MVP folder)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend
npm install          # vad-web + onnxruntime-web already in package.json
npm start            # prestart auto-copies VAD assets into public/
```

First backend start downloads the Whisper model and (on first TTS call)
the Kokoro model (~330 MB). Both are cached afterwards.

### Mode B — free GPU latency test (Colab)

Proves the production latency number before you have server access.

```python
# Colab (GPU runtime) — upload the backend/ and data/ folders first
!pip install faster-whisper scipy fastapi uvicorn kokoro soundfile pyngrok -q
!ngrok config add-authtoken <your-free-token>      # ngrok.com
from pyngrok import ngrok
print(ngrok.connect(8000))                          # → https://xxxx.ngrok.io
!python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

On your laptop: `echo REACT_APP_BACKEND_URL=https://xxxx.ngrok.io > frontend/.env`
then `npm start`. You are now testing the exact production path: local
browser VAD + real CUDA Whisper.

### Mode C — production (college GPU server + kiosk)

```bash
# GPU server
python -c "import ctranslate2; print('GPUs:', ctranslate2.get_cuda_device_count())"  # must be ≥ 1
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# 0.0.0.0 is required so the kiosk can reach it. Open port 8000 in the firewall.

# Kiosk machine
cd frontend
echo REACT_APP_BACKEND_URL=http://<GPU_SERVER_LAN_IP>:8000 > .env
npm start            # or: npm run build + serve the build folder locally
```

Startup log must show: `[STT] Loading large-v3-turbo on cuda (float16)`.
If it says `cpu`, CUDA isn't visible — fix drivers before demoing.

### Environment variables

| Var           | Default (GPU / CPU)            | Purpose                    |
|---------------|--------------------------------|----------------------------|
| `STT_DEVICE`  | auto                           | force `cuda` or `cpu`      |
| `STT_MODEL`   | large-v3-turbo / small.en      | any faster-whisper model   |
| `STT_CPU_THREADS` | 8                          | CPU mode thread count      |
| `TTS_VOICE`   | af_heart                       | af_bella, am_michael, bf_emma… |

---

## 4. The noise & single-speaker system

Six layers, in the order audio passes through them. Layers 1–5 are in
the shipped code; layer 6 is hardware.

**Layer 1 — Browser audio constraints** (`kioskMic.js`)
`noiseSuppression: true`, `echoCancellation: true`, `autoGainControl: false`.
Echo cancellation is what stops the kiosk transcribing its own TTS.
AGC is OFF on purpose — it amplifies crowd noise during silence, which
would defeat layer 4.

**Layer 2 — Neural VAD gate** (`kioskMic.js`)
Silero VAD with `positiveSpeechThreshold: 0.8`. Distant voices produce
lower speech probabilities and are rejected before any audio leaves the
browser. `minSpeechFrames: 4` ignores coughs and screen taps.

**Layer 3 — Bandpass filter** (`backend/audio_processing.py`)
80 Hz – 7.5 kHz Butterworth. Removes AC hum, fan rumble, electrical hiss.

**Layer 4 — Energy gate** (`backend/audio_processing.py`)
A person at the kiosk is 10–15 dB louder at the mic than background
crowd. Utterances below the RMS threshold are rejected and the visitor
is effectively asked to come closer / speak up. THIS IS THE MAIN
SINGLE-SPEAKER DEFENCE and it must be tuned on-site (§5).

**Layer 5 — Whisper-level robustness** (`backend/stt.py`)
`vad_filter=True` (Silero again, inside faster-whisper) plus the campus
vocabulary `initial_prompt`, which keeps accuracy high on Indian English
and domain terms even in noise. Plus the confidence score in every
response for a future re-prompt gate.

**Layer 6 — Hardware (pending purchase)**
ReSpeaker 4-Mic USB array (~₹3,000) with onboard beamforming: only a
~60° frontal cone is amplified, everything lateral is attenuated in
silicon before software ever sees it. Plus the acoustic shroud from
EP-06. This is what takes you from "works in a demo" to "works during
lunch rush in the lobby".

**What is honestly NOT solved:** two people standing equally close,
speaking equally loudly, at the same time. No affordable real-time
software separates that. The mitigation is physical and procedural —
beamforming cone + kiosk placement + a "one visitor at a time" floor
marking. Say this plainly if asked in a review; it is the correct
engineering answer, not a weakness.

---

## 5. On-site tuning procedure (10 minutes, do this once per venue)

1. Start the backend and `tail` its logs. Every utterance prints
   `[STT] '<text>' | conf X.XX | Yms`, and rejected audio prints
   `too_quiet`.
2. Stand at normal kiosk distance (~40 cm) and speak 10 test phrases.
   All 10 should transcribe.
3. Step 2 metres away and talk at conversation volume. These should be
   rejected (`too_quiet`) or never sent (VAD didn't trigger).
4. If crowd noise gets through → raise `threshold` in `is_too_quiet()`
   (`backend/audio_processing.py`, default 0.010, try 0.015) and/or
   raise `positiveSpeechThreshold` in `kioskMic.js` (0.8 → 0.85).
5. If quiet speakers get rejected → lower the same two values.
6. Run the EP-06 acceptance test: 20 standard phrases at ambient 60 dB,
   75 dB, 85 dB (use a phone dB-meter app). Record accuracy. Target:
   >90% at 75 dB. Save the numbers — this is your promotable benchmark.

---

## 6. TTS

Kokoro-82M runs on CPU faster than real-time — the GPU is not involved.
The frontend `speak()` calls `POST /tts` and plays the returned WAV; if
the backend TTS fails for any reason it falls back to the browser voice
so the kiosk never goes mute. Change the voice with `TTS_VOICE`. If
Kokoro import fails on the server (`pip install kokoro soundfile`), the
system degrades gracefully to browser TTS — check startup logs for
`[TTS] Kokoro available.`

---

## 7. Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| Mic permission never appears | App not served from localhost/HTTPS → serve frontend on the kiosk machine |
| Console 404 on `silero_vad*.onnx` or `*.wasm` | VAD assets missing from public/ → `npm run copy-vad` |
| `[STT] Loading ... on cpu` on the GPU server | CUDA not visible → check `nvidia-smi`, reinstall CUDA-enabled ctranslate2 |
| First request takes 30+ s | Model download / warmup on first boot → normal, once per machine |
| Kiosk transcribes its own voice | Echo cancellation unsupported by browser/mic → Chrome + decent mic; VAD is also paused during TTS as backup |
| Everything transcribes as empty | Energy gate too strict → lower threshold (§5) |
| Robotic voice | Kokoro not installed on backend → `pip install kokoro soundfile`, check logs |
| CORS errors | `ALLOWED_ORIGINS` env on backend must include the frontend origin |
| Old `POST /stt` used somewhere | Legacy WebM path still works (ffmpeg required) but is deprecated — migrate to `/stt/pcm` |

---

## 8. What to demo / promote

The story with numbers: "Rebuilt the kiosk STT pipeline — moved from
fixed-window WebM recording + CPU Whisper (~10 s per query) to
in-browser neural VAD + raw PCM streaming + GPU Whisper large-v3-turbo:
**under 0.7 s speech-to-text**, fully offline, zero API cost, with a
six-layer noise-rejection chain validated at 75 dB ambient." Attach the
§5 test numbers. That is a legitimate engineering result.
