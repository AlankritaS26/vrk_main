import urllib.request
import json
import time

def test_phrase(phrase):
    t0 = time.time()
    req = urllib.request.Request(
        'http://127.0.0.1:8001/tts',
        data=json.dumps({'text': phrase}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as res:
        dt = (time.time() - t0) * 1000
        data = json.loads(res.read().decode('utf-8'))
        audio = data.get('audio')
        print(f"Phrase: '{phrase}' -> {dt:.1f}ms, audio length: {len(audio) if audio else 0}")

if __name__ == '__main__':
    test_phrase("Sure, let me check that for you.")
    test_phrase("Sure, let me check that for you.")
    test_phrase("This is a short answer.")
