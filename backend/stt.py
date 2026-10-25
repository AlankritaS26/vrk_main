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
import io
import re
import shutil
import time
import logging
import numpy as np
from faster_whisper import WhisperModel

from backend.audio_processing import preprocess, SAMPLE_RATE

# ---------------------------------------------------------------- config

CAMPUS_PROMPT = (
    "RNS Institute of Technology, Bengaluru, Channasandra. "
    "USN, SGPA, CGPA, CIE, SEE, attendance, hostel, Block-C, "
    "ECE, CSE, ISE, AIML, principal, HOD, placement cell, library."
)

NAME_PROMPT = (
    "Visitor names: Rahul, Priya, Alankrita, Rohan, Ananya, Aditya, Sneha, Amit, Vikram, Neha, "
    "Akshatha, Suresh, Rajesh, Ramesh, Karthik, Kavya, Pooja, Divya, Sanjay, Deepak, Swathi, Meera, Arjun, "
    "RNS Institute of Technology campus visitor."
)

SPELLING_PROMPT = (
    "Letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z. "
    "Spelled letters: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z. "
    "NATO phonetic alphabet: Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, Hotel, India, Juliet, "
    "Kilo, Lima, Mike, November, Oscar, Papa, Quebec, Romeo, Sierra, Tango, Uniform, Victor, Whiskey, "
    "X-ray, Yankee, Zulu. A as in Apple, B as in Boy, C as in Cat."
)

STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "1"))
STT_NAME_BEAM_SIZE = int(os.getenv("STT_NAME_BEAM_SIZE", "3"))

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

logger = logging.getLogger('RNSIT_Kiosk.STT')


def transcribe_pcm(pcm_bytes: bytes, language: "str | None" = "en", mode: str = "normal", candidate: str = "") -> dict:
    """
    Transcribe one complete utterance of 16 kHz mono int16 PCM.
    (The browser VAD decides where the utterance starts/ends —
     by the time this is called we have exactly one turn of speech.)

    Supports two logical STT modes:
      - "normal": fast, greedy (beam_size=STT_BEAM_SIZE), campus vocabulary
      - "name" / "spelling": accuracy-focused (beam_size=STT_NAME_BEAM_SIZE),
        letter / name prompt bias, confidence extraction and low confidence rejection.

    Returns the normalized TranscriptResult dict:
      {text, confidence, language, latency_ms, mode, low_confidence} or {..., error}
    """
    start = time.time()
    try:
        audio = pcm16_bytes_to_float32(pcm_bytes)

        import numpy as _np
        _rms = float(_np.sqrt(_np.mean(audio ** 2))) if len(audio) else 0.0
        logger.info(f"[STT] recv {len(audio)} samples "
                    f"({len(audio)/SAMPLE_RATE:.2f}s) rms={_rms:.5f} mode={mode}")

        if len(audio) < SAMPLE_RATE * 0.3:                      # <300 ms
            logger.info("[STT] rejected: too_short")
            return {"text": "", "confidence": 0.0,
                    "language": language or "en", "mode": mode, "error": "too_short"}

        # cap runaway buffers at 30 s (EP-03 acceptance criteria)
        audio = audio[: SAMPLE_RATE * 30]

        # DSP chain — EP-06
        audio = preprocess(audio)
        if audio is None:
            logger.info(f"[STT] rejected: too_quiet (rms={_rms:.5f} < gate)")
            return {"text": "", "confidence": 0.0,
                    "language": language or "en", "mode": mode, "error": "too_quiet"}

        # Configure Whisper parameters based on logical mode
        normalized_mode = (mode or "normal").lower().strip()
        if normalized_mode == "spelling":
            beam_size = STT_NAME_BEAM_SIZE
            best_of = STT_NAME_BEAM_SIZE
            if candidate and len(candidate.strip()) >= 2:
                cand_clean = candidate.strip().title()
                cand_letters = " ".join(list(re.sub(r'[^A-Za-z]', '', cand_clean).upper()))
                prompt = (
                    f"Spelling visitor name: {cand_clean}. Spelled letters: {cand_letters}. "
                    f"Letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z. "
                    f"Spelled letters: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z. "
                    f"NATO: Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, Hotel, India, Juliet, "
                    f"Kilo, Lima, Mike, November, Oscar, Papa, Quebec, Romeo, Sierra, Tango, Uniform, Victor, "
                    f"Whiskey, X-ray, Yankee, Zulu."
                )
            else:
                prompt = SPELLING_PROMPT
            min_silence = 250
        elif normalized_mode == "name":
            beam_size = STT_NAME_BEAM_SIZE
            best_of = STT_NAME_BEAM_SIZE
            prompt = NAME_PROMPT
            min_silence = 300
        else:
            beam_size = STT_BEAM_SIZE
            best_of = 1
            prompt = CAMPUS_PROMPT
            min_silence = 300

        segments, info = model.transcribe(
            audio,
            language=language,               # None = auto-detect (first turn)
            beam_size=beam_size,
            best_of=best_of,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": min_silence,
                "speech_pad_ms": 150,
            },
            no_speech_threshold=0.6,
        )

        text, logprobs = "", []
        for seg in segments:
            text += seg.text
            logprobs.append(seg.avg_logprob)
        text = text.strip()
        logger.info(f"[STT] [{normalized_mode}] transcript={text!r}")

        confidence = (max(0.0, min(1.0, float(np.exp(np.mean(logprobs)))))
                      if logprobs else 0.0)
        latency_ms = int((time.time() - start) * 1000)

        # Flag low confidence in name or spelling mode for rejection/retry
        is_low_confidence = False
        if normalized_mode in ("name", "spelling") and text:
            if confidence < 0.35:
                is_low_confidence = True
                logger.info(f"[STT] [{normalized_mode}] low confidence flagged: {confidence:.2f}")

        print(f"[STT] [{normalized_mode}] '{text}' | conf {confidence:.2f} | {latency_ms}ms")
        return {
            "text": text,
            "confidence": round(confidence, 2),
            "language": info.language,
            "latency_ms": latency_ms,
            "mode": normalized_mode,
            "low_confidence": is_low_confidence,
        }

    except Exception as e:
        print(f"[STT] Error: {e}")
        return {"text": "", "confidence": 0.0,
                "language": language or "en", "mode": mode, "error": str(e)}


# ---------------------------------------------------------------- legacy path

def transcribe_audio(audio_bytes: bytes) -> dict:
    """
    DEPRECATED — kept only so the old POST /stt (WebM upload) endpoint
    doesn't break while the frontend migrates to the WebSocket + PCM path.
    Still uses ffmpeg; remove once /ws/stt is live everywhere.
    """
    import subprocess, tempfile, wave

    # Mobile Expo recordings are usually AAC/M4A. Decode them with PyAV so
    # the backend does not depend on a system ffmpeg executable.
    try:
        import av
        container = av.open(io.BytesIO(audio_bytes))
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )
        chunks = []
        for frame in container.decode(audio=0):
            converted = resampler.resample(frame)
            if not isinstance(converted, list):
                converted = [converted]
            for output in converted:
                if output is not None:
                    chunks.append(output.to_ndarray().reshape(-1))
        if chunks:
            return transcribe_pcm(np.concatenate(chunks))
    except Exception as e:
        logging.getLogger("RNSIT_Kiosk.STT").warning(
            "[STT] PyAV decode failed, trying ffmpeg: %s", e
        )

    if not shutil.which("ffmpeg"):
        return {
            "text": "",
            "confidence": 0.0,
            "language": "en",
            "error": "audio_decoder_unavailable",
        }

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