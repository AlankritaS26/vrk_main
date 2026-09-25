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
import time
import logging
import threading
import cProfile
import pstats
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

logger = logging.getLogger("RNSIT_Kiosk.TTS")

load_dotenv()

# ── Dedicated single-thread executor for TTS ──────────────────────────────
# Root cause of "still slow after warmup" even with _kokoro_lock in place:
# _kokoro_lock only stops two syntheses running AT THE SAME TIME — it does
# nothing about WHICH thread each one runs on. main.py's /tts endpoint calls
# `asyncio.to_thread()`, which hands work to the event loop's default
# executor, a POOL of several worker threads — not one fixed thread.
# PyTorch/OpenMP lazily spins up its internal thread pool the first time any
# torch op runs on a given OS thread, and THAT spin-up (not the tiny
# inference itself) is what costs hundreds of ms to 1-2s. `_warmup()` used
# to run on its own dedicated `threading.Thread()`, so it only ever warmed
# THAT one thread — live requests kept landing on whichever pool thread
# happened to be free and re-paid the cold-start cost every time, lock or
# no lock. Routing warmup AND every real request through this single
# dedicated thread means torch's thread pool is spun up exactly once, on
# the one thread that ever runs inference — see text_to_speech_on_worker()
# below and main.py's /tts endpoint.
TTS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-worker")


def _default_tts_threads() -> int:
    """Torch intra-op thread cap for the TTS worker.

    Set to 12 per explicit direction after reviewing bench_tts.py's thread
    sweep on the actual kiosk hardware (12 cores): threads=12 measured
    fastest (3641.5ms mean). The sweep itself was noisy — 2 through 14
    threads mostly overlapped within stdev — so this is a judgment call on
    top of that data, not a clean statistical win; recorded here as a
    deliberate choice rather than re-derived automatically, so a future
    edit doesn't quietly change it back. TTS_TORCH_THREADS still overrides
    this explicitly if you want to test another value.
    """
    return 12


def _init_torch_threads():
    """Pin torch's thread count on the TTS worker thread. See
    _default_tts_threads() above for the thread-count reasoning (12,
    per explicit direction from the bench_tts.py results) —
    TTS_TORCH_THREADS still overrides it explicitly if needed."""
    try:
        import torch
        threads = int(os.getenv("TTS_TORCH_THREADS", str(_default_tts_threads())))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Can only be called once per process and only before any
            # interop op has run — ignore if it's too late.
            pass
        # flush_denormal(True) was tried here and REVERTED. Hypothesis was
        # that Kokoro's conv-heavy vocoder hits the classic CPU denormal
        # slowdown (near-zero float values falling back to slow microcode
        # paths) — a real, well-documented effect on SOME hardware/model
        # combos. bench_tts.py's direct A/B on THIS machine (Ryzen-class
        # AVX2, no AVX512) showed the opposite: off=3421ms vs on=3673ms,
        # i.e. no improvement (if anything slightly worse, within noise).
        # Per "don't keep an optimization that isn't measurably better":
        # left at PyTorch's own default (off) rather than kept on a
        # plausible-sounding guess that didn't hold up when measured.
        # Visible confirmation of what's actually engaged — needed to tell
        # "thread cap is still the bottleneck" apart from "thread cap was
        # already raised, the remaining cost is genuinely inference" when
        # reading [TTS-PROFILE] timings below.
        logger.info("[TTS-PROFILE] torch threads pinned: %d (cpu_count=%s)",
                    threads, os.cpu_count())
    except ImportError:
        pass


# Apply the thread settings once, right away, on the dedicated TTS thread —
# before Kokoro (and its underlying torch ops) ever run on it.
TTS_EXECUTOR.submit(_init_torch_threads)

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


# Serialize calls into the shared Kokoro pipeline instance. Kokoro's
# inference is CPU-bound (PyTorch); running two syntheses at once on the
# same process doesn't parallelize on limited kiosk hardware, it just makes
# BOTH calls slower (GIL + CPU contention). This mattered in practice:
# the frontend fires an unawaited "thinking" filler phrase (e.g. "Sure, let
# me check that for you.") right as the real answer arrives, so its TTS call
# and the real answer's first-chunk TTS call would land back-to-back and
# contend for the same CPU. Serializing them keeps each call's latency
# predictable instead of both stalling by seconds under contention.
_kokoro_lock = threading.Lock()


def _synthesize_kokoro(text: str) -> bytes:
    # Stage-level timing — kept cheap (a handful of time.monotonic() calls)
    # so it can stay on in production and actually show, per call, whether
    # a slow /tts is lock contention, phonemization+inference (the model
    # itself), or post-processing/encoding, instead of only ever seeing one
    # opaque total. pipe(text, ...) is a generator that does phonemization
    # and vocoder inference together per chunk — Kokoro doesn't expose
    # those as separate steps — so "phonemize+infer" below is that combined
    # cost, which is normally the dominant one.
    t0 = time.monotonic()
    with _kokoro_lock:
        t_lock = time.monotonic()
        pipe = _get_pipe()
        t_pipe = time.monotonic()
        chunks = [audio for _, _, audio in pipe(text, voice=TTS_VOICE, speed=TTS_SPEED)]
        t_infer = time.monotonic()
    if not chunks:
        return b""
    audio = _trim_silence(np.concatenate(chunks), 24000)
    t_trim = time.monotonic()
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    t_encode = time.monotonic()
    # INFO, not DEBUG: main.py's logging.basicConfig(level=logging.INFO)
    # meant this line was silently dropped before it ever reached a handler
    # — the "profiling isn't showing up" symptom. It's cheap (a handful of
    # time.monotonic() calls, no extra work), so it's fine to leave at INFO
    # rather than require a log-level change to see it.
    logger.info(
        "[TTS-PROFILE] lock_wait=%.0fms pipe_init=%.0fms phonemize+infer=%.0fms "
        "trim=%.0fms encode=%.0fms total=%.0fms (%d chars): '%s'",
        (t_lock - t0) * 1000, (t_pipe - t_lock) * 1000,
        (t_infer - t_pipe) * 1000, (t_trim - t_infer) * 1000,
        (t_encode - t_trim) * 1000, (t_encode - t0) * 1000, len(text),
        text[:40],
    )
    return buf.getvalue()


def _log_runtime_diagnostics(pipe):
    """One-time dump of exactly what's actually running: torch device,
    dtype, thread counts, CPU backend config, and the Kokoro model/voice
    config — requested so we can rule out "silently running fp64" or "not
    actually seeing the thread cap we set" etc. before reasoning about
    where the per-call time goes. Best-effort: KPipeline's internals
    aren't officially documented across versions, so every introspection
    attempt is guarded and falls back to an honest "unknown" rather than
    guessing. This is pure introspection — it changes nothing about how
    inference runs, so unlike the thread-count/compile/dtype experiments
    it doesn't need A/B benchmarking to justify being here."""
    try:
        import torch
        threads, interop = torch.get_num_threads(), torch.get_num_interop_threads()
        try:
            mkldnn_ok = torch.backends.mkldnn.is_available()
        except Exception:
            mkldnn_ok = "n/a"
        try:
            mkl_ok = torch.backends.mkl.is_available()
        except Exception:
            mkl_ok = "n/a"
        try:
            cpu_cap = torch.backends.cpu.get_cpu_capability()
        except Exception:
            cpu_cap = "n/a (older torch build)"
        logger.info(
            "[TTS-PROFILE] cpu backend: torch=%s mkldnn=%s mkl=%s cpu_capability=%s "
            "OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s",
            torch.__version__, mkldnn_ok, mkl_ok, cpu_cap,
            os.getenv("OMP_NUM_THREADS", "<unset>"), os.getenv("MKL_NUM_THREADS", "<unset>"),
        )
    except ImportError:
        threads = interop = "torch unavailable"

    device = dtype = "unknown"
    model_obj = getattr(pipe, "model", None)
    if model_obj is not None:
        try:
            p = next(model_obj.parameters())
            device, dtype = str(p.device), str(p.dtype)
        except Exception as e:
            device = dtype = f"introspection failed: {e}"

    logger.info(
        "[TTS-PROFILE] runtime: torch_threads=%s torch_interop_threads=%s "
        "model_device=%s model_dtype=%s voice=%s speed=%s lang_code=a engine=%s",
        threads, interop, device, dtype, TTS_VOICE, TTS_SPEED, TTS_ENGINE,
    )


def _synthesize_kokoro_profiled(text: str) -> bytes:
    """Same work as _synthesize_kokoro(), but with the phonemize+infer span
    wrapped in cProfile instead of one opaque timer, so we can see WHICH
    function inside it actually costs the time — G2P/phonemization,
    tokenization, KModel.forward, the decoder/vocoder, or something
    unexpected like the voice pack being reloaded from disk every call —
    without having to guess at kokoro's internal method names in advance
    (they aren't part of its public API and vary across versions).
    TEMPORARY diagnostic per request: used by the startup repeat-call test
    below, and per-request when TTS_DEEP_PROFILE=1. cProfile adds real
    per-Python-call-boundary overhead, so it's opt-in for live traffic
    (see _TTS_DEEP_PROFILE) rather than always-on like the cheap timing in
    _synthesize_kokoro()."""
    profiler = cProfile.Profile()
    with _kokoro_lock:
        pipe = _get_pipe()
        profiler.enable()
        chunks = [audio for _, _, audio in pipe(text, voice=TTS_VOICE, speed=TTS_SPEED)]
        profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(15)
    logger.info("[TTS-PROFILE] deep breakdown (%d chars) '%s':\n%s",
                len(text), text[:40], stream.getvalue())
    if not chunks:
        return b""
    audio = _trim_silence(np.concatenate(chunks), 24000)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    return buf.getvalue()


# Opt-in per-request deep profiling (see _synthesize_kokoro_profiled above).
# Off by default — this is for turning on temporarily against live traffic
# if the startup repeat-call test below isn't enough to reproduce the
# pattern seen in production.
_TTS_DEEP_PROFILE = os.getenv("TTS_DEEP_PROFILE", "0") == "1"


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
        logger.info(f"[TTS-TIMING] cache HIT ({len(text)} chars): '{text[:40]}...'"
                    if len(text) > 40 else f"[TTS-TIMING] cache HIT: '{text}'")
        return cached

    engine = TTS_ENGINE
    if engine == "none":
        return b""

    def _run(eng):
        if eng == "melotts":
            return _synthesize_melo(text)
        return _synthesize_kokoro_profiled(text) if _TTS_DEEP_PROFILE else _synthesize_kokoro(text)

    t0 = time.monotonic()
    try:
        wav = _run(engine)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"[TTS-TIMING] cache MISS, synthesized in {elapsed_ms:.0f}ms "
                    f"({engine}, {len(text)} chars): '{text[:40]}'")
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


def text_to_speech_on_worker(text: str, language: str = "en") -> bytes:
    """Run text_to_speech() on the dedicated TTS_EXECUTOR thread and block
    until it's done. main.py's /tts endpoint calls this (via
    run_in_executor) instead of asyncio.to_thread(text_to_speech, ...), so
    real requests land on the exact same warmed thread _warmup() uses below
    — that's the fix, see the TTS_EXECUTOR comment above. _kokoro_lock still
    protects against any future change that adds more TTS_EXECUTOR workers."""
    return TTS_EXECUTOR.submit(text_to_speech, text, language).result()


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

            # ── Repeat-call diagnostic ────────────────────────────────────
            # Same text synthesized twice back to back, BOTH after the
            # warmup above (so neither run pays cold-start cost) — this is
            # what tells apart "the multi-second cost is a one-time
            # per-process expense" (run2 much faster than run1) from
            # "it's genuinely paid on every call" (run1 ≈ run2, meaning the
            # cost is real inference work, not warmup). Also dumps a
            # cProfile breakdown of each run and the runtime config
            # (device/dtype/threads) once, so the exact expensive function
            # is visible rather than inferred. Uses a fixed sentence in the
            # same length range (~70 chars) as the slow calls actually
            # observed in production logs.
            if KOKORO_AVAILABLE and TTS_ENGINE == "kokoro":
                try:
                    _log_runtime_diagnostics(_get_pipe())
                    _repeat_text = ("The college working hours are 9:20 AM "
                                     "to 5:00 PM on all working days.")
                    t_r1 = time.monotonic()
                    _synthesize_kokoro_profiled(_repeat_text)
                    d_r1 = (time.monotonic() - t_r1) * 1000
                    t_r2 = time.monotonic()
                    _synthesize_kokoro_profiled(_repeat_text)
                    d_r2 = (time.monotonic() - t_r2) * 1000
                    verdict = ("run1 much slower than run2 -> looks like residual "
                               "cold-start/warmup cost, not steady-state inference"
                               if d_r1 > d_r2 * 1.5 else
                               "run1 ~= run2 -> cost is paid on every call; this is "
                               "genuine per-call inference cost, not a warmup gap")
                    logger.info(
                        "[TTS-PROFILE] repeat-call test (%d chars): run1=%.0fms "
                        "run2=%.0fms delta=%.0fms -> %s",
                        len(_repeat_text), d_r1, d_r2, d_r1 - d_r2, verdict,
                    )
                except Exception as e:
                    logger.warning("[TTS-PROFILE] repeat-call test failed: %s", e)
        except Exception as e:
            print(f"[TTS] Warmup failed: {e}")
        finally:
            # ALWAYS set this, success or failure — otherwise /health would
            # wait forever and the whole kiosk (detection included) would
            # never start.
            PREWARM_DONE.set()

    # Warmup now runs on TTS_EXECUTOR — the SAME dedicated thread every real
    # /tts request runs on (see main.py) — instead of its own separate,
    # never-reused thread. That mismatch was the actual root cause of
    # "still slow after warmup"; _kokoro_lock alone couldn't fix it.
    TTS_EXECUTOR.submit(_warmup)
else:
    print("[TTS] No engine available at all — browser TTS fallback only.")
    PREWARM_DONE.set()