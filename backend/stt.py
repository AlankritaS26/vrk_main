"""
STT Pipeline — VRK Kiosk (EP-03)

Production merge of stt_test/stt_pipeline.py + backend/stt.py:
  * GPU (CUDA fp16) in prod, auto-fallback to CPU int8 in dev
  * Direct numpy PCM input — NO ffmpeg, NO temp files, NO webm
  * Warmup run so the first visitor doesn't pay model-init latency
  * DSP chain (bandpass + energy gate) before Whisper
  * Campus-vocabulary prompt bias for Indian English / domain terms
  * Confidence extraction for the re-prompt gate

Env vars (matches the provider-abstraction story in EP-03):
  STT_DEVICE = auto | cuda | cpu          (default: auto)
  STT_MODEL  = override model name        (default: large-v3-turbo on GPU,
                                                    small.en on CPU)
"""

import os
import time
import numpy as np
from faster_whisper import WhisperModel

from backend.audio_processing import preprocess, SAMPLE_RATE

# ---------------------------------------------------------------- config

CAMPUS_PROMPT = (
    "RNS Institute of Technology, Bengaluru, Channasandra. "
    "USN, SGPA, CGPA, CIE, SEE, attendance, hostel, Block-C, "
    "ECE, CSE, ISE, AIML, principal, HOD, placement cell, library."
)

def _pick_device_and_model():
    device = os.getenv("STT_DEVICE", "auto")
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if device == "cuda":
        model_name = os.getenv("STT_MODEL", "large-v3-turbo")
        compute = "float16"
        threads = 0
    else:
        # dev laptop: small.en int8 keeps latency usable without a GPU
        model_name = os.getenv("STT_MODEL", "small.en")
        compute = "int8"
        threads = int(os.getenv("STT_CPU_THREADS", "8"))
    return device, model_name, compute, threads


DEVICE, MODEL_NAME, COMPUTE, THREADS = _pick_device_and_model()

print(f"[STT] Loading {MODEL_NAME} on {DEVICE} ({COMPUTE})...")
t0 = time.time()
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE,
    cpu_threads=THREADS,
)
print(f"[STT] Model ready in {time.time() - t0:.1f}s")

# warmup — first real transcription is not slowed by lazy allocation
_ = list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                          language="en", beam_size=1)[0])
print("[STT] Warmup complete.")


# ---------------------------------------------------------------- helpers

def pcm16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Int16 little-endian PCM (what the frontend sends) → float32 [-1, 1]."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------- main API

def transcribe_pcm(pcm_bytes: bytes, language: "str | None" = "en") -> dict:
    """
    Transcribe one complete utterance of 16 kHz mono int16 PCM.
    (The browser VAD decides where the utterance starts/ends —
     by the time this is called we have exactly one turn of speech.)

    Returns the normalized TranscriptResult dict from EP-03:
      {text, confidence, language, latency_ms} or {..., error}
    """
    start = time.time()
    try:
        audio = pcm16_bytes_to_float32(pcm_bytes)

        if len(audio) < SAMPLE_RATE * 0.3:                      # <300 ms
            return {"text": "", "confidence": 0.0,
                    "language": language or "en", "error": "too_short"}

        # cap runaway buffers at 30 s (EP-03 acceptance criteria)
        audio = audio[: SAMPLE_RATE * 30]

        # DSP chain — EP-06
        audio = preprocess(audio)
        if audio is None:
            return {"text": "", "confidence": 0.0,
                    "language": language or "en", "error": "too_quiet"}

        segments, info = model.transcribe(
            audio,
            language=language,               # None = auto-detect (first turn)
            beam_size=1,                     # greedy: fastest, ~no quality loss
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=CAMPUS_PROMPT,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 150,
            },
            no_speech_threshold=0.6,
        )

        text, logprobs = "", []
        for seg in segments:
            text += seg.text
            logprobs.append(seg.avg_logprob)
        text = text.strip()

        confidence = (max(0.0, min(1.0, float(np.exp(np.mean(logprobs)))))
                      if logprobs else 0.0)
        latency_ms = int((time.time() - start) * 1000)

        print(f"[STT] '{text}' | conf {confidence:.2f} | {latency_ms}ms")
        return {
            "text": text,
            "confidence": round(confidence, 2),
            "language": info.language,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        print(f"[STT] Error: {e}")
        return {"text": "", "confidence": 0.0,
                "language": language or "en", "error": str(e)}


# ---------------------------------------------------------------- legacy path

def transcribe_audio(audio_bytes: bytes) -> dict:
    """
    DEPRECATED — kept only so the old POST /stt (WebM upload) endpoint
    doesn't break while the frontend migrates to the WebSocket + PCM path.
    Still uses ffmpeg; remove once /ws/stt is live everywhere.
    """
    import subprocess, tempfile, wave

    webm_path = wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            webm_path = f.name
        wav_path = webm_path.replace(".webm", ".wav")

        r = subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, timeout=10,
        )
        if r.returncode != 0:
            return {"text": "", "confidence": 0.0,
                    "language": "en", "error": "conversion_failed"}

        with wave.open(wav_path, "r") as wf:
            pcm = wf.readframes(wf.getnframes())
        return transcribe_pcm(pcm)

    except Exception as e:
        return {"text": "", "confidence": 0.0, "language": "en", "error": str(e)}
    finally:
        for p in (webm_path, wav_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
