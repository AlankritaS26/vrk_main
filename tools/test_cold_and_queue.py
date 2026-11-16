import os
import sys
import time
import statistics
import concurrent.futures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.tts import text_to_speech, text_to_speech_on_worker, _TTS_CACHE

def run_tests():
    print("=" * 70)
    print("PHASE 0: COLD START & CONCURRENT QUEUE WAIT TEST")
    print("=" * 70)

    # 1. First /tts after warmed
    test_phrase = "Here is an un-cached unique query phrase to test first-call steady state latency."
    
    # 5 runs of fresh (uncached) phrases
    single_call_times = []
    for i in range(5):
        unique_phrase = f"Uncached test query number {i} for RNSIT digital kiosk assistant."
        t0 = time.monotonic()
        wav = text_to_speech_on_worker(unique_phrase)
        dur = (time.monotonic() - t0) * 1000.0
        audio_dur = len(wav) / 24000.0 / 2.0 * 1000.0 # 16-bit 24kHz
        single_call_times.append(dur)
        print(f"Single call run {i+1}: {dur:.1f}ms (audio: {len(wav)} bytes)")

    print(f"\nSingle /tts call steady-state: Median = {statistics.median(single_call_times):.1f}ms, Worst = {max(single_call_times):.1f}ms")

    # 2. Two simultaneous /tts calls (queue wait on single worker)
    print("\nTesting 2 simultaneous /tts calls (measuring queue wait on TTS_EXECUTOR)...")
    simul_run_stats = [] # tuple of (call1_total, call2_total, queue_wait_call2)
    for i in range(5):
        p1 = f"First simultaneous utterance number {i} sent to kiosk."
        p2 = f"Second simultaneous utterance number {i} sent to kiosk."

        t_submit = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(text_to_speech_on_worker, p1)
            f2 = pool.submit(text_to_speech_on_worker, p2)
            concurrent.futures.wait([f1, f2])
            res1 = f1.result()
            res2 = f2.result()
        t_finish = time.monotonic()
        total_time = (t_finish - t_submit) * 1000.0
        simul_run_stats.append(total_time)
        print(f"Simultaneous run {i+1}: both finished in {total_time:.1f}ms")

    print(f"\nSimultaneous 2-call total time: Median = {statistics.median(simul_run_stats):.1f}ms, Worst = {max(simul_run_stats):.1f}ms")

if __name__ == "__main__":
    run_tests()
