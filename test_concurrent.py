import urllib.request
import json
import time
import concurrent.futures

URL = "http://127.0.0.1:8001/tts"

def req_tts(text, label):
    t0 = time.perf_counter()
    req = urllib.request.Request(
        URL,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        dt = (time.perf_counter() - t0) * 1000
        audio_b64 = data.get("audio")
        alen = len(audio_b64) if audio_b64 else 0
        return label, dt, alen

def test_concurrent_uncached():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # Send filler first, then 10ms later send answer chunk
        f1 = pool.submit(req_tts, "One fresh uncached filler phrase right now.", "Filler (uncached)")
        time.sleep(0.02)
        f2 = pool.submit(req_tts, "One fresh uncached answer sentence arriving closely.", "Answer chunk (uncached)")
        for f in concurrent.futures.as_completed([f1, f2]):
            label, dt, alen = f.result()
            print(f"{label}: {dt:.1f}ms (len {alen})")

if __name__ == "__main__":
    test_concurrent_uncached()
