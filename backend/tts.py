"""
TTS Pipeline — VRK Kiosk (EP-04)

Default provider: Kokoro-82M (Apache 2.0, human-quality, faster than
real-time on CPU — no GPU needed). Falls back to empty bytes if Kokoro
isn't installed, which makes the frontend use browser speechSynthesis,
so the kiosk keeps talking either way.

Install on the machine running the backend:
    pip install kokoro soundfile
(English works out of the box; installing espeak-ng additionally
improves pronunciation of rare/unknown words.)

Voices to try (set TTS_VOICE env var): af_heart, af_bella, af_nicole,
am_michael, am_adam, bf_emma (British).
"""

import io
import os
import numpy as np

TTS_VOICE = os.getenv("TTS_VOICE", "af_bella")   # warmer receptionist voice
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.05"))    # 1.0-1.15 stays natural

_TTS_CACHE: dict = {}          # (text, voice, speed) -> wav bytes
_TTS_CACHE_MAX = 500           # plenty for a day of unique sentences

_pipe = None
KOKORO_AVAILABLE = False
try:
    from kokoro import KPipeline
    import soundfile as sf
    KOKORO_AVAILABLE = True
    print("[TTS] Kokoro available.")
except ImportError as e:
    print(f"[TTS] Kokoro import failed: {e} — frontend will use browser TTS. "
          "Run: pip install kokoro soundfile")


def _get_pipe():
    """Lazy init — first call downloads the 82M model (~330MB), then cached."""
    global _pipe
    if _pipe is None:
        print("[TTS] Loading Kokoro pipeline...")
        _pipe = KPipeline(lang_code="a")   # 'a' = American English, 'h' = Hindi
        print("[TTS] Kokoro ready.")
    return _pipe


def text_to_speech(text: str, language: str = "en") -> bytes:
    """Text → WAV bytes (24 kHz mono). Empty bytes = frontend fallback."""
    if not text or not text.strip():
        return b""

    text = (text.replace("\u2014", ",").replace("\u2013", ",")   # em/en dash -> brief pause
                .replace("\u2019", "'").replace("\u2018", "'")   # curly apostrophes -> straight
                .replace("\u201c", '"').replace("\u201d", '"')   # curly quotes -> straight
                .replace("\u2026", ", "))                          # ellipsis -> brief pause
    key = (text.strip(), TTS_VOICE, TTS_SPEED)
    cached = _TTS_CACHE.get(key)
    if cached is not None:
        return cached                      # instant — repeated sentence, no synthesis

    if not KOKORO_AVAILABLE:
        return b""

    try:
        pipe = _get_pipe()
        chunks = [audio for _, _, audio in pipe(text, voice=TTS_VOICE, speed=TTS_SPEED)]
        if not chunks:
            return b""
        audio = np.concatenate(chunks)

        # Trim leading/trailing silence. Kokoro pads every clip with
        # ~0.2-0.4s of quiet; stacked per-sentence that becomes the long,
        # robotic pause at every full stop. Keep a natural 60ms breath.
        nz = np.where(np.abs(audio) > 0.004)[0]
        if len(nz):
            pad = int(24000 * 0.06)
            audio = audio[max(0, nz[0] - pad): min(len(audio), nz[-1] + pad)]

        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV")
        wav = buf.getvalue()
        if len(_TTS_CACHE) < _TTS_CACHE_MAX:
            _TTS_CACHE[key] = wav
        return wav
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return b""


# ── Warm Kokoro at startup (background thread) so the first visitor
# doesn't pay the model-load + first-synthesis cost.
if KOKORO_AVAILABLE:
    import threading

    def _warmup():
        try:
            list(_get_pipe()("Hello", voice=TTS_VOICE, speed=TTS_SPEED))
            print("[TTS] Kokoro warmed up.")
        except Exception as e:
            print(f"[TTS] Warmup failed: {e}")

    threading.Thread(target=_warmup, daemon=True).start()