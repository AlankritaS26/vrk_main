import urllib.request
import json
import time

URL = "http://127.0.0.1:8001/tts"

def req_tts(text):
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
        audio_len = len(audio_b64) if audio_b64 else 0
        return dt, audio_len

def run_tests():
    phrases = [
        ("(c) Filler phrase (cached)", "Sure, let me check that for you."),
        ("(a) Short answer (1 sentence)", "The library is on the first floor of the academic block."),
        ("(b) 4-sentence answer (sentence 1)", "The Department of Computer Science and Engineering was established in 2001."),
        ("(b) 4-sentence answer (sentence 2)", "It offers undergraduate and postgraduate programs with excellent faculty."),
        ("(b) 4-sentence answer (sentence 3)", "The department has state-of-the-art laboratories and research facilities."),
        ("(b) 4-sentence answer (sentence 4)", "Placements for CSE students have consistently exceeded ninety percent."),
        ("Filler with em-dash", "Good question — one moment."),
        ("Filler with hyphen", "Good question - one moment."),
    ]
    print(f"{'Phrase Description':<38} | {'Length':<6} | {'Status':<7} | {'Time (ms)':<9}")
    print("-" * 68)
    for desc, text in phrases:
        dt, alen = req_tts(text)
        status = "OK" if alen > 0 else "FAIL"
        print(f"{desc:<38} | {alen:<6} | {status:<7} | {dt:>8.1f}ms")

if __name__ == "__main__":
    run_tests()
