import os
import sys
import time
import io
import numpy as np
import soundfile as sf
import platform
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.tts import speech_prep, text_to_speech, _get_pipe, TTS_VOICE, TTS_SPEED, _TTS_CACHE

def current_frontend_splitter(text: str):
    # Mirrors buildTtsSentenceChunks in WelcomeScreen.js
    import re
    raw = re.findall(r'[^.!?]+[.!?]+["\']?\s*|[^.!?]+$', text)
    if not raw:
        raw = [text]
    raw = [s.strip() for s in raw if s.strip()]
    sentences = []
    if raw:
        first = raw[0]
        if len(first) > 60:
            cut = first.find(',')
            if cut > 15:
                sentences.append(first[:cut + 1])
                first = first[cut + 1:].strip()
        if first:
            sentences.append(first)
        buf = ''
        for s in raw[1:]:
            buf = (buf + ' ' + s) if buf else s
            if len(buf) >= 90:
                sentences.append(buf)
                buf = ''
        if buf:
            sentences.append(buf)

    if len(sentences) > 1 and len(sentences[0]) < 25:
        sentences[1] = sentences[0] + ' ' + sentences[1]
        sentences.pop(0)

    return sentences

def measure_audio_silence(audio: np.ndarray, sr: int, threshold_db: float = -50.0):
    thresh = 10.0 ** (threshold_db / 20.0)
    abs_audio = np.abs(audio)
    nz = np.where(abs_audio > thresh)[0]
    if len(nz) == 0:
        return len(audio) / sr * 1000.0, len(audio) / sr * 1000.0
    head_silence_ms = (nz[0] / sr) * 1000.0
    tail_silence_ms = ((len(audio) - 1 - nz[-1]) / sr) * 1000.0
    return head_silence_ms, tail_silence_ms

def benchmark():
    print("=" * 80)
    print("TTS BENCHMARK & PHASE 0 HARDWARE AUDIT")
    print("=" * 80)
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Processor: {platform.processor()}")
    print(f"Physical / Logical cores: {os.cpu_count()}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Torch threads: {torch.get_num_threads()}")
    try:
        import psutil
        print(f"RAM total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"Battery: {psutil.sensors_battery()}")
    except Exception as e:
        print(f"psutil info error: {e}")

    # Warmup
    print("\nWarming up Kokoro pipeline...")
    t_warm_start = time.monotonic()
    pipe = _get_pipe()
    list(pipe("Hello", voice=TTS_VOICE, speed=TTS_SPEED))
    print(f"Kokoro warmup took {(time.monotonic() - t_warm_start)*1000:.1f}ms")

    test_strings = [
        "Dr. M K Venkatesha is the principal. He holds a Ph.D. in CSE.",
        "The B.E. fee is Rs. 1,20,000 per year, e.g. for CSE, ISE, etc. Call 9:30 a.m. onwards.",
        "Welcome back, Akshatha! It's a cloudy day. How can I help?",
        ("RNS Institute of Technology is an autonomous engineering college located in Bengaluru. "
         "It was established in the year 2001 by Dr. R. N. Shetty. "
         "The college offers undergraduate courses in CSE, ISE, ECE, and other branches. "
         "Admissions are conducted through KCET and COMEDK counselling. "
         "The campus features state of the art labs, libraries, and sports facilities. "
         "Placement records are consistently strong with top recruiters visiting every year.")
    ]

    for idx, s in enumerate(test_strings, 1):
        print("\n" + "-" * 80)
        print(f"STRING {idx}: \"{s[:70]}...\"" if len(s) > 70 else f"STRING {idx}: \"{s}\"")
        chunks = current_frontend_splitter(s)
        print(f"Frontend Splitter ({len(chunks)} chunks):")
        for ci, c in enumerate(chunks):
            print(f"   [{ci}] ({len(c)} chars): \"{c}\"")

        # Synthesize each chunk individually (as current frontend does via /tts)
        total_synth_ms = 0.0
        total_audio_ms = 0.0
        print("\nPer-Chunk Measurements (un-cached):")
        print(f"{'Chunk':<6}{'Chars':<7}{'Synth(ms)':<12}{'Audio(ms)':<12}{'RTF':<8}{'HeadSil(ms)':<14}{'TailSil(ms)':<14}")
        for ci, c in enumerate(chunks):
            # bypass cache for pure measurement
            prepped = speech_prep(c)
            t0 = time.monotonic()
            raw_chunks = [audio for _, _, audio in pipe(prepped, voice=TTS_VOICE, speed=TTS_SPEED)]
            synth_ms = (time.monotonic() - t0) * 1000.0
            if raw_chunks:
                from backend.tts import _trim_silence
                audio = _trim_silence(np.concatenate(raw_chunks), 24000)
                audio_ms = (len(audio) / 24000.0) * 1000.0
                rtf = synth_ms / audio_ms if audio_ms > 0 else 0
                head_ms, tail_ms = measure_audio_silence(audio, 24000)
            else:
                audio_ms, rtf, head_ms, tail_ms = 0, 0, 0, 0
            total_synth_ms += synth_ms
            total_audio_ms += audio_ms
            print(f"[{ci}]   {len(c):<7}{synth_ms:<12.1f}{audio_ms:<12.1f}{rtf:<8.2f}{head_ms:<14.1f}{tail_ms:<14.1f}")

        overall_rtf = total_synth_ms / total_audio_ms if total_audio_ms > 0 else 0
        print(f"Total: Synth={total_synth_ms:.1f}ms, Audio={total_audio_ms:.1f}ms, Overall RTF={overall_rtf:.2f}")

if __name__ == "__main__":
    benchmark()
