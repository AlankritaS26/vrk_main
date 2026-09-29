"""
tts.py — text-to-speech for the VRK kiosk.

ENGINE DEFAULT: Kokoro. This was accidentally defaulted to "melotts" during
an earlier experiment and silently ran that way for real visitors with no
.env override — that bug is fixed here. MeloTTS remains available as an
explicit opt-in (TTS_ENGINE=melotts) if you want to revisit it later, but
it is NOT the default and never will be unless set explicitly.
"""
import os
import io
import threading

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

TTS_VOICE = os.getenv("TTS_VOICE", "af_bella")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.05"))

# ── Response cache: (text, engine, voice, speed) -> WAV bytes ────────────
_TTS_CACHE: dict = {}
_TTS_CACHE_MAX = 500

PREWARM_DONE = threading.Event()   # set once ALL fixed phrases are cached —
                                   # /health waits on this, not just "the
                                   # process is alive" (see main.py)

# ── Kokoro (default engine) ───────────────────────────────────────────────
KOKORO_AVAILABLE = False
_pipe = None
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
    print("[TTS] Kokoro available.")
except ImportError as e:
    print(f"[TTS] Kokoro import failed: {e}")


def _get_pipe():
    """Lazy init — first call downloads the 82M model (~330MB), then cached."""
    global _pipe
    if _pipe is None:
        print("[TTS] Loading Kokoro pipeline...")
        _pipe = KPipeline(lang_code="a")   # 'a' = American English
        print("[TTS] Kokoro ready.")
    return _pipe


# ── MeloTTS (optional, opt-in only — TTS_ENGINE=melotts) ──────────────────
# Install: git clone https://github.com/myshell-ai/MeloTTS, pip install -e .,
# python -m unidic download. Known issue if you revisit this: occasional
# garbled/unstable output under concurrent requests (no lock around the
# shared model instance) — treat as experimental, not production-default.
TTS_ENGINE   = os.getenv("TTS_ENGINE", "kokoro").lower()   # "kokoro" | "melotts"
MELO_SPEAKER = os.getenv("MELO_SPEAKER", "EN_INDIA")
MELO_AVAILABLE = False
_melo_model = None
_melo_lock = threading.Lock()   # serialize calls into the shared model instance
if TTS_ENGINE == "melotts":
    try:
        from melo.api import TTS as _MeloTTS
        MELO_AVAILABLE = True
        print("[TTS] MeloTTS available.")
    except ImportError as e:
        print(f"[TTS] MeloTTS import failed ({e}) — falling back to Kokoro.")
        TTS_ENGINE = "kokoro"


def _get_melo():
    global _melo_model
    if _melo_model is None:
        print("[TTS] Loading MeloTTS (CPU)...")
        _melo_model = _MeloTTS(language="EN", device="cpu")
        print("[TTS] MeloTTS ready.")
    return _melo_model


if TTS_ENGINE == "melotts" and not MELO_AVAILABLE:
    TTS_ENGINE = "kokoro"
if TTS_ENGINE == "kokoro" and not KOKORO_AVAILABLE:
    TTS_ENGINE = "melotts" if MELO_AVAILABLE else "none"

print(f"[TTS] active engine = {TTS_ENGINE}", flush=True)


def _trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Keep a natural ~60ms breath at each end instead of the long pause
    raw model output tends to have."""
    nz = np.where(np.abs(audio) > 0.004)[0]
    if len(nz):
        pad = int(sr * 0.06)
        audio = audio[max(0, nz[0] - pad): min(len(audio), nz[-1] + pad)]
    return audio


def _synthesize_kokoro(text: str) -> bytes:
    pipe = _get_pipe()
    chunks = [audio for _, _, audio in pipe(text, voice=TTS_VOICE, speed=TTS_SPEED)]
    if not chunks:
        return b""
    audio = _trim_silence(np.concatenate(chunks), 24000)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    return buf.getvalue()


def _synthesize_melo(text: str) -> bytes:
    """Locked — only one synthesis call into the shared model instance at a
    time. This is the fix for the intermittent garbled/ghost-sounding audio
    observed when two requests (e.g. an acknowledgment + the real answer)
    landed close together and both hit the model concurrently."""
    import tempfile
    with _melo_lock:
        model = _get_melo()
        speaker_ids = model.hps.data.spk2id
        spk = speaker_ids.get(MELO_SPEAKER, next(iter(speaker_ids.values())))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            model.tts_to_file(text, spk, tmp_path, speed=TTS_SPEED)
            audio, sr = sf.read(tmp_path, dtype="float32")
            audio = _trim_silence(audio, sr)
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV")
            return buf.getvalue()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def text_to_speech(text: str, language: str = "en") -> bytes:
    """Text -> WAV bytes. Empty bytes = frontend fallback (browser TTS)."""
    if not text or not text.strip():
        return b""

    text = (text.replace("\u2014", ",").replace("\u2013", ",")
                .replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2026", ", "))

    voice_tag = MELO_SPEAKER if TTS_ENGINE == "melotts" else TTS_VOICE
    key = (text.strip(), TTS_ENGINE, voice_tag, TTS_SPEED)
    cached = _TTS_CACHE.get(key)
    if cached is not None:
        return cached

    engine = TTS_ENGINE
    if engine == "none":
        return b""

    def _run(eng):
        return _synthesize_melo(text) if eng == "melotts" else _synthesize_kokoro(text)

    try:
        wav = _run(engine)
    except Exception as e:
        print(f"[TTS] {engine} failed: {e} — trying the other engine")
        other = "kokoro" if engine == "melotts" else "melotts"
        other_ok = MELO_AVAILABLE if other == "melotts" else KOKORO_AVAILABLE
        wav = b""
        if other_ok:
            try:
                wav = _run(other)
            except Exception as e2:
                print(f"[TTS] {other} fallback also failed: {e2}")

    if wav and len(_TTS_CACHE) < _TTS_CACHE_MAX:
        _TTS_CACHE[key] = wav
    return wav or b""


# ── Warm the active engine at startup + pre-cache fixed phrases ──────────
if KOKORO_AVAILABLE or MELO_AVAILABLE:
    def _warmup():
        try:
            if TTS_ENGINE == "melotts" and MELO_AVAILABLE:
                _get_melo()
            elif KOKORO_AVAILABLE:
                list(_get_pipe()("Hello", voice=TTS_VOICE, speed=TTS_SPEED))

            INSTITUTE_NAME = os.getenv("INSTITUTE_NAME", "R N S Institute of Technology")
            _PREWARM = [
                # First-visit greeting — fixed text, must match main.py's
                # build_greeting() word-for-word or the cache silently misses.
                (f"Welcome to {INSTITUTE_NAME}. I am Nova, your digital receptionist. "
                 f"I can help you with admissions, departments, placements, fees, and "
                 f"directions around campus. How may I assist you today?"),
                "Sure, let me check that for you.",
                "Good question - one moment.",
                "Let me look that up for you.",
                "Of course, just a second.",
                "Right, let me find that.",
                "You are most welcome! Have a wonderful day. Goodbye!",
                "Happy to help! Take care and have a great day.",
            ]
            for p in _PREWARM:
                try:
                    text_to_speech(p)
                except Exception:
                    pass
            print(f"[TTS] {TTS_ENGINE} warmed up and pre-cached {len(_PREWARM)} phrases.")
        except Exception as e:
            print(f"[TTS] Warmup failed: {e}")
        finally:
            # ALWAYS set this, success or failure — otherwise /health would
            # wait forever and the whole kiosk (detection included) would
            # never start.
            PREWARM_DONE.set()

    threading.Thread(target=_warmup, daemon=True).start()
else:
    print("[TTS] No engine available at all — browser TTS fallback only.")
    PREWARM_DONE.set()