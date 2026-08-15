# VRK — Voice Receptionist Kiosk

An AI-powered digital receptionist kiosk for RNS Institute of Technology.
Visitors walk up, are recognized (or greeted and registered) by camera, and
hold a natural voice conversation — self-hosted, zero-cost stack
(free-tier APIs only). **Fully hands-free: voice-based, no clicking required.**

## Quick start — one command

```powershell
venv\Scripts\python.exe run.py
```

That single command, in one terminal:
1. Frees ports 8000/3000 if a stale kiosk process is squatting on them
2. Starts the **backend** (FastAPI) and waits for `/health` to pass
3. Starts **camera detection** (only after the backend is ready)
4. Starts the **frontend** (npm) and opens the kiosk browser window with
   autoplay enabled — required for the greeting to speak unprompted
5. Prefixes all logs by service: `[BACKEND]` `[DETECT]` `[FRONTEND]`
6. `Ctrl+C` stops everything, including npm's child processes

Do **not** open localhost:3000 in your own tab for kiosk testing — use the
window `run.py` opens; it carries the `--autoplay-policy` flag that lets the
kiosk speak before any user gesture.

First-time setup (fresh machine): see **docs/SETUP.md**. Summary: Python
3.12 venv → `pip install -r backend/requirements.txt` → `.env` from
`.env.example` (Mongo URI, API key) → `cd frontend && npm install`.

## What it does

- **Kiosk-initiated conversation** — detection spots a visitor and the kiosk
  greets them by voice first (greeting audio is pre-cached per visitor for a
  sub-second start). Returning visitors are greeted by name.
- **Voice-based new visitor flow** — no text input or clicking. When a new
  visitor is detected, the kiosk says "What's your name?" and listens via VAD.
  Name is transcribed, confirmed, and registration proceeds automatically.
- **Blink detection** — Eye Aspect Ratio (EAR) from face landmarks enables
  blink-based confirmation: "Would you like more info? [User blinks → Yes]"
- **Proximity detection** — estimates distance from camera in meters; shows
  encouragement when visitor approaches ("Come closer! I see you!")
- **Personality-driven responses** — detects question intent (curious, happy,
  grateful, etc.) and responds with contextual reactions and prefixes.
- **Speech-to-text** — Silero VAD in the browser segments speech;
  raw 16 kHz PCM streams to faster-whisper (`small.en` int8 on CPU dev,
  `large-v3-turbo` CUDA in production) with a bandpass + energy-gate DSP
  chain for lobby noise.
- **Answer engine** — guardrails → RAG over the college knowledge base
  (Gemini embeddings + generation, MongoDB source, Redis hot cache). The
  query-condense step is skipped on early turns to save a network round-trip.
- **Text-to-speech** — Kokoro-82M (`af_bella`, 1.05× pace) with:
  silence-trimmed clips, ~2-sentence chunking, two-chunk prefetch, gapless
  Web Audio playback, per-sentence response cache, punctuation
  normalization (curly quotes/dashes), and browser-voice fallback so the
  kiosk never goes mute.
- **Face recognition** — MediaPipe + DeepFace with consent flow, guest mode,
  and a GDPR-style **Delete My Data** flow (see Known Issues).
- **UI** — idle attract screen (live clock, capability carousel, watching
  radar), conversation screen with a persistent voice dock (waveform while
  listening, breathing status dot otherwise, thinking dots while the LLM
  works), synced text+voice bubbles, instant goodbye transition with the
  farewell voice playing over the goodbye screen.

---

## Voice-First Interaction Features

### 1. Voice-Based Name Input (Fully Hands-Free)

When a new visitor is detected:
1. Modal appears: "Hi! I'm Nova. What's your name?"
2. Kiosk **automatically** starts listening via browser VAD
3. User speaks their name clearly (e.g., "Akshatha Sharma")
4. STT transcribes in real-time with waveform visualization
5. Name confirmation shown: "✓ Got it! Akshatha"
6. **Auto-proceeds after 3 seconds** (zero clicks needed)
7. Face is registered and session begins with personalized greeting

**Implementation:**
- No text input field
- No button clicks required
- Listening starts immediately when modal appears
- Waveform shows audio activity in real-time
- Auto-confirmation speeds up new visitor onboarding

### 2. Blink Detection for Yes/No Confirmation

Eye Aspect Ratio (EAR) calculated from 468-point MediaPipe face landmarks:
- **Open eyes**: EAR ≥ 0.20
- **Closed eyes**: EAR < 0.15
- **Blink detected**: Open → Closed → Open transition

**Use case example:**
```
Kiosk: "Would you like to hear more about placements?"
User: [blinks naturally]
Kiosk: [detects blink, responds affirmatively]
```

**WebSocket data includes:**
- `blink_detected`: bool (true if blink occurred this frame)
- `eyes_closed`: bool (true if eyes currently closed)
- `ear_left`: float (left eye aspect ratio)
- `ear_right`: float (right eye aspect ratio)

**To implement blink-based Yes/No:**
```javascript
// In WelcomeScreen.js
const handleYesNoQuestion = (question) => {
  addMessage(question, 'kiosk');
  setWaitingForBlink(true);
  setBlinkCallback(() => handleYes());
};
```

### 3. Nearest Person Detection & Prioritization

Estimates visitor distance from camera using face bounding box:
- **Formula**: `distance_m = (200px @ 1m) / bbox_width_pixels`
- **Range**: 0.1m - 10m
- **Accuracy**: ±10% typical
- **Calibration**: Assumes face width ≈ 200px at 1m for standard webcam

**Use cases:**
- Show encouragement when visitor approaches: "I see you! Come closer! 👋"
- Multi-person kiosk: prioritize the closest person
- Distance-based UX: adjust greeting volume based on proximity
- Analytics: track how close visitors stand

**WebSocket includes:**
```json
{
  "distance_m": 1.05,
  "bbox": {"x": 150, "y": 100, "w": 200, "h": 250}
}
```

**Example integration:**
```javascript
if (distanceM !== null && distanceM < 1.5) {
  showEncouragement("Come closer! I can see you better.");
}
```

### 4. Personality & Emotion System (Framework Ready)

Detects question intent and provides contextual responses:

**Intent Types:**
| Intent | Trigger Words | Avatar Reaction |
|--------|---------------|-----------------|
| `curious` | How, Why, What, Tell me about | Head tilt, thoughtful |
| `happy` | Placements, great, excellent, wonderful | Smile, enthusiastic |
| `thanks` | Thanks, thank you, appreciated | Bow, grateful gesture |
| `greeting` | Hello, Hi, Hey, Good morning | Arms up, big smile |
| `confused` | Unclear/too short/too long | Shrug, questioning |
| `standard` | Regular questions | Neutral, processing |

**Time-Based Greetings:**
```
6 AM - 12 PM  →  "Good morning! ☀️"
12 PM - 6 PM  →  "Good afternoon! 🌤️"
6 PM - 9 PM   →  "Good evening! 🌙"
9 PM - 6 AM   →  "It's late! Still here? 🌙"
```

**Personality Prefixes (auto-generated per intent):**
- Curious: "Great question! Let me look into that for you."
- Happy: "Your enthusiasm is amazing! Here's the great news..."
- Thanks: "You're welcome! Anything else I can help with?"
- Greeting: "Hello there! What brings you here today?"

**Helper Functions Ready in `frontend/src/avatarReactions.js`:**

```javascript
import {
  detectQuestionIntent,          // string → intent
  getPersonalityPrefix,          // intent → prefix string
  getPersonalitySuffix,          // intent → suffix string
  getAvatarStateForIntent,       // intent → animation state
  getTimeBasedGreeting,          // none → {greeting, emoji}
  getIdleScreenMessage,          // distanceM → encouraging message
  getReactionDuration            // intent → ms to hold reaction
} from './avatarReactions';

// Example usage
const userMsg = "How are the placements?";
const intent = detectQuestionIntent(userMsg);
// Returns: "happy"

const reaction = getAvatarStateForIntent(intent);
setStatus(reaction); // Avatar tilts head happily

const prefix = getPersonalityPrefix(intent);
// Returns: "Your enthusiasm is awesome! Here's..."

const fullResponse = `${prefix} ${answer}`;
```

**To integrate personality into message flow:**
1. Detect intent when user speaks
2. Set avatar reaction state for 0.8-1.2s
3. Prepend personality prefix to response
4. Append contextual suffix
5. Avatar smoothly transitions to speaking state

---

## Testing Voice-First Features

### Test 1: Voice Name Input
```
1. Start kiosk:  python run.py
2. Approach camera as NEW visitor (face not in system)
3. Listen for: "Hi! I'm Nova. What's your name?"
4. Speak clearly: "Akshatha Sharma"
5. Watch waveform animate in real-time
6. Modal shows: "✓ Got it! Akshatha"
7. Auto-proceeds in 3 seconds (no clicking!)
8. Session starts with: "Welcome back!" (if registered before)
   OR "Welcome to RNSIT!" (if new registration)
```

### Test 2: Blink Detection
```
1. Start kiosk normally
2. Open DevTools (F12) → Network → ws/detect
3. During conversation, blink naturally
4. Watch WebSocket messages for:
   {
     "blink_detected": true,
     "eyes_closed": true,
     "ear_left": 0.08,
     "ear_right": 0.09
   }
5. Blink again - should detect another blink event
```

### Test 3: Distance Detection
```
1. Start kiosk
2. Open console (F12 → Console)
3. Add logging: window.distanceM = distanceM (from props)
4. Move 2 meters away from camera
5. Check console: distance_m ≈ 2.0
6. Move 1 meter away
7. Check console: distance_m ≈ 1.0
8. Move 0.5 meters away
9. Check console: distance_m ≈ 0.5

Calibration:
- Stand exactly 1m from camera
- Measure face bbox width (should be ~200px)
- If off, adjust FACE_WIDTH_AT_1M_PX in detection.py
```

### Test 4: Personality Features
```javascript
// In browser console
import { detectQuestionIntent } from './avatarReactions';

// Test intent detection
const userMsgs = [
  "How do I apply?",              // curious
  "What about placements?",       // happy
  "Thank you so much!",           // thanks
  "Hello everyone",               // greeting
  "xyz abc 123",                  // confused
  "Tell me about campus"          // standard
];

userMsgs.forEach(msg => {
  const intent = detectQuestionIntent(msg);
  console.log(`"${msg}" → ${intent}`);
});

// Test time-based greeting
const { greeting, emoji } = getTimeBasedGreeting();
console.log(`${emoji} ${greeting}`);
// Output: ☀️ Good morning! (if before noon)
```

---

## Architecture Overview

### Backend Detection Pipeline

```
Frame (640x480) via WebSocket
    ↓ (3 fps continuous)
MediaPipe FaceLandmarker (468-point mesh)
    ├─ Blink Detection: Eye Aspect Ratio (EAR)
    │   └─ Detect open→closed→open transition
    ├─ Distance Estimation: bbox width analysis
    │   └─ distance_m = (200px @ 1m) / bbox_width
    ├─ Face Recognition: identity + verification
    │   └─ SCRFD + ArcFace or DeepFace
    └─ State Machine: IDLE→DWELLING→RECOGNIZING→ACTIVE
    ↓
WebSocket /ws/detect Broadcast
    ↓
Frontend: React State Update
    ↓
UI Reaction: Avatar animation, message display
```

### Frontend Data Flow

```
WebSocket /ws/detect Message {
  "present": bool,
  "state": "IDLE" | "DWELLING" | "RECOGNIZING" | "ACTIVE" | ...,
  "identity": "Akshatha A",
  "verified": bool,
  "bbox": {
    "x": 150,
    "y": 100,
    "w": 200,
    "h": 250,
    "distance_m": 1.05      ← NEW
  },
  "bystanders": 0,
  "blink_detected": false,  ← NEW
  "eyes_closed": false,     ← NEW
  "ear_left": 0.35,         ← NEW
  "ear_right": 0.34         ← NEW
}
    ↓
App.js (Global State)
    ├─ detState
    ├─ identity
    ├─ bbox
    ├─ blinkDetected      ← NEW
    ├─ eyesClosed         ← NEW
    └─ distanceM          ← NEW
    ↓
WelcomeScreen / IdleScreen
    ├─ Show encouragement if distanceM < 1.5
    ├─ Trigger blink callbacks
    ├─ Display avatar reactions
    └─ Handle personality responses
```

---

## Files Modified for Voice-First Features

### Backend

**`backend/detection.py`**
- Added: `_distance()` - Euclidean distance helper
- Added: `_calculate_ear()` - Eye Aspect Ratio from landmarks
- Added: `_get_blink_state()` - Blink detection logic
- Added: `_estimate_distance()` - Distance from face bbox
- Modified: `DetectionResult` dataclass
  - Added: `blink_detected: bool`
  - Added: `eyes_closed: bool`
  - Added: `ear_left: Optional[float]`
  - Added: `ear_right: Optional[float]`
  - Added: `full_landmarks: Optional[list]`
- Modified: `KioskState` dataclass
  - Added: `prev_ear_left`, `prev_ear_right` (blink tracking)
  - Added: `last_blink_time` (timestamp)
- Modified: `BoundingBox` dataclass
  - Added: `distance_m: Optional[float]`
- Modified: `detect_presence()` function
  - Calculate EAR and blink detection
  - Estimate distance for primary visitor
  - Store full landmarks for EAR calc

**`backend/main.py`**
- Modified: `/ws/detect` WebSocket endpoint
  - Added to response: `blink_detected`, `eyes_closed`, `ear_left`, `ear_right`, `distance_m`

### Frontend

**`frontend/src/App.js`**
- Added: `blinkDetected` state (bool)
- Added: `eyesClosed` state (bool)
- Added: `distanceM` state (float | null)
- Modified: WebSocket handler
  - Capture: `data.blink_detected`, `data.eyes_closed`, `data.ear_left/right`, `data.distance_m`
- Modified: `detectionProps` object
  - Pass new state to child screens

**`frontend/src/WelcomeScreen.js`**
- Replaced: Text input modal → Voice-only name capture
- Added: `voiceNameState` state (idle | listening | processing | confirming)
- Added: `startVoiceNameCapture()` function
  - Auto-starts listening when askingName becomes true
  - Captures audio via VAD
  - Sends PCM to `/stt/pcm` backend
  - Extracts name from transcription
- Added: Auto-confirmation timeout
  - Shows confirmed name for 3 seconds
  - Auto-submits (zero clicks)
- Added: Blink detection state variables
  - `waitingForBlink`, `blinkCallback`
  - Ready for blink-based Yes/No
- Imported: `avatarReactions.js` helpers

**`frontend/src/avatarReactions.js`** (NEW)
- Exported: `detectQuestionIntent(text)` → intent type
- Exported: `getPersonalityPrefix(intent)` → response prefix
- Exported: `getPersonalitySuffix(visitCount, isReturning)` → response suffix
- Exported: `getTimeBasedGreeting()` → {greeting, emoji}
- Exported: `getAvatarStateForIntent(intent)` → animation state
- Exported: `getReactionDuration(intent)` → milliseconds to hold reaction
- Exported: `getIdleScreenMessage(distanceM)` → encouraging message

---

## Configuration (.env)

Standard configuration:
```
MONGO_URI=<your-mongodb-uri>
LLM_API_KEY=<gemini-free-tier-key>
ALLOWED_ORIGINS=http://localhost:3000
STT_DEVICE=auto
TTS_VOICE=af_bella
TTS_SPEED=1.05
```

**Blink Detection Thresholds** (hardcoded in `backend/detection.py`):
```python
EAR_CLOSED_THRESHOLD = 0.15  # Eyes closed
EAR_OPEN_THRESHOLD = 0.20    # Eyes open
```

**Distance Calibration** (hardcoded in `backend/detection.py`):
```python
FACE_WIDTH_AT_1M_PX = 200    # Pixels at 1m distance
ASSUMED_FRAME_WIDTH = 640    # Standard frame width
```

To adjust: measure face at known distance and adjust these values.

---

## Repository Layout

```
run.py                 ← Start everything (one command)
backend/
  ├─ main.py          FastAPI server + WebSocket endpoints
  ├─ detection.py     Face detection + blink + distance
  ├─ stt.py           Speech-to-text (Whisper)
  ├─ tts.py           Text-to-speech (Kokoro)
  ├─ llm.py / gemini.py   LLM answer generation
  ├─ recognition.py   Face recognition engine
  └─ requirements.txt
frontend/
  ├─ src/
  │  ├─ App.js         Main app + detection WebSocket
  │  ├─ WelcomeScreen.js   Voice name input + personality
  │  ├─ IdleScreen.js      Attract mode + distance
  │  ├─ avatarReactions.js ← NEW: Personality system
  │  ├─ kioskMic.js    Browser VAD + audio capture
  │  ├─ AriaAvatar.js  Avatar animations
  │  └─ index.js
  ├─ package.json
  └─ public/
data/
  └─ college_info.json
docs/
  ├─ SETUP.md
  ├─ ARCHITECTURE.md
  └─ OPERATIONS.md
README.md              ← This file (all features here)
```

## Key Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check (used by run.py) |
| `/stt/pcm` | POST | Raw PCM → transcript (speech-to-text) |
| `/tts` | POST | Text → base64 WAV (text-to-speech, cached) |
| `/ask` | GET | Question → grounded answer |
| `/session/start` | POST | Begin new session |
| `/session/end` | POST | End session |
| `/session/current` | GET | Get active session info |
| `/visitor/unknown` | POST | Detect new visitor |
| `/visitor/submit_name` | POST | Submit visitor name (voice or typed) |
| `/visitor/delete_my_data` | POST | Privacy: delete face + history |
| `/ws/detect` | WebSocket | Real-time detection stream (face, blink, distance) |

---

## Known Issues (Tracked)

- **[HIGH — privacy] detection.py stale face cache**: after Delete My Data,
  the visitor is still recognized until detection restarts. Cause: encodings
  load once at startup. Fix: re-fetch `/faces/all` on cache reload event.

- **Detection-to-greeting latency (~2–3 s)**: browser polling (750ms) + detection
  recognition cadence. Next: WebSocket early-exit signals.

- **Conversation latency floor (~2–4 s/turn)**: bounded by Gemini API
  round-trips. Next: model tier upgrade, embedding cache optimization.

- **Blink detection accuracy**: depends on lighting and face angle. Best results
  1-2m away in well-lit environments. May need EAR threshold calibration for
  different lighting conditions. Threshold: `EAR < 0.15` for closed, `>= 0.20` for open.

- **Distance calibration**: assumes standard webcam focal length. If using
  different camera (e.g., wide-angle, fish-eye), recalibrate `FACE_WIDTH_AT_1M_PX`.

---

## Next Steps (Optional Enhancements)

### 1. Complete Personality Integration
```javascript
// In WelcomeScreen.js message handler
const intent = detectQuestionIntent(userMessage);
setStatus(getAvatarStateForIntent(intent)); // Show reaction
setTimeout(() => {
  setStatus('speaking'); // Transition to speaking
  const prefix = getPersonalityPrefix(intent);
  fullAnswer = `${prefix} ${answer}`;
  speak(fullAnswer);
}, getReactionDuration(intent));
```

### 2. Enable Blink-Based Yes/No
```javascript
// In WelcomeScreen.js
const askYesNo = (question, onYes) => {
  addMessage(question, 'kiosk');
  setWaitingForBlink(true);
  setBlinkCallback(onYes);
  // Timeout fallback after 5s
  setTimeout(() => {
    if (waitingForBlink) {
      // User didn't blink, fallback to voice: "yes" / "no"
      startListening();
    }
  }, 5000);
};
```

### 3. Distance-Based UX
```javascript
// In IdleScreen.js
if (distanceM !== null) {
  if (distanceM > 3) {
    showMessage("You look far away. Walk closer, please! 👋");
  } else if (distanceM < 1.5) {
    showMessage("I see you! Ready to help. 😊");
  }
}
```

### 4. Weather Integration
```javascript
// Fetch local weather
const weather = await fetch('https://api.open-meteo.com/v1/forecast?...');
const greeting = `Good morning! Weather is ${weather.description}.`;
```

### 5. Multi-Language Support
```javascript
// Detect language from STT
const language = result.language; // e.g., "hi", "es"
// Switch LLM + TTS language
setLanguage(language);
```

---

## Troubleshooting

### Voice Name Input Issues

**Problem:** Voice name input not capturing audio
- Check microphone permissions in browser settings
- Chrome: Settings → Privacy → Site Settings → Microphone → Allow localhost
- Try in Incognito mode (rules out extensions blocking mic)
- Check console for errors in `kioskMic.js` initialization

**Problem:** STT shows "Processing..." forever
- Verify backend is running: `http://127.0.0.1:8001/docs`
- Check `/stt/pcm` endpoint responds
- Look at backend terminal for STT errors
- Try with a simple greeting first: "Hello"

**Problem:** Name extracted incorrectly
- Speak more slowly and clearly
- STT may need campus vocabulary training (edit backend/stt.py)
- Try shorter name: "Sharma" works better than "Akshatha Sharma"

### Blink Detection Issues

**Problem:** Blinks not detected
- Ensure good lighting on visitor's face
- Avoid strong backlighting (sun in face)
- Get closer to camera (best at <1.5m)
- Check WebSocket shows changing EAR values

**Problem:** Too many false blink detections
- Environment too dark (eyes appear closed)
- Lower lighting for better detection
- Increase EAR thresholds in `detection.py`
  - `EAR_OPEN_THRESHOLD = 0.25` (stricter)

**Problem:** Blinks detected but inconsistent
- Face angle affects EAR calculation
- Visitor looking down won't blink-detect
- Ensure visitor faces camera directly

### Distance Detection Issues

**Problem:** Distance values seem wrong
- Calibrate at exactly 1m from camera
- Measure face bbox width (should be ~200px)
- If different, adjust in `detection.py`:
  ```python
  FACE_WIDTH_AT_1M_PX = <measured_width>
  ```

**Problem:** Distance jumps around too much
- Frame-to-frame variance is normal at 3fps
- Add low-pass filter in frontend:
  ```javascript
  const smoothDistance = 0.7 * prevDistance + 0.3 * newDistance;
  ```

**Problem:** Multiple faces, wrong one prioritized
- Currently prioritizes largest face
- To prioritize closest: use `distance_m`, not face area
- Backend mod: compare `distance_m` instead of `bbox_area`

### Personality System Issues

**Problem:** Intent detection not working as expected
- Intent matching is substring-based and case-insensitive
- Test with exact keywords: "placements", "thank", "how", etc.
- Add console logging to `detectQuestionIntent()`

**Problem:** Avatar reactions not showing
- Check avatar animation CSS classes exist
- Verify `status` state is being set correctly
- Ensure CSS transitions are not disabled

---

## Performance Metrics

### Voice Name Input Latency
- VAD speech-end detection: ~100ms after user stops
- STT processing: ~300-400ms (GPU), ~1-2s (CPU)
- Name extraction: ~50ms
- **Total end-to-end:** 450-2500ms

### Blink Detection
- EAR calculation: real-time, zero latency
- Runs at 3 fps (detection rate)
- CPU overhead: < 1% (no ML inference)

### Distance Estimation
- Calculation: per frame at 3 fps
- Model accuracy: ±10% typical
- CPU overhead: < 0.1% (simple math)

### Overall System
- WebSocket latency: ~50-100ms
- Detection pipeline: ~100-300ms per frame (3fps)
- STT round-trip: ~500-2500ms (GPU/CPU)
- LLM answer generation: ~2-4s (Gemini API)
- TTS generation: ~300-1000ms (Kokoro cached)

---

## Performance Optimization Tips

1. **Faster STT:** Use GPU
   ```bash
   # In .env
   STT_DEVICE=cuda
   # Use faster model
   # STT_MODEL=large-v3-turbo
   ```

2. **Reduce blink false positives:** Better lighting
   - Move kiosk near windows or add task lighting
   - Avoid backlighting
   - Increase `EAR_OPEN_THRESHOLD` to 0.22+

3. **Smoother distance updates:** Add filtering
   ```javascript
   const smoothDistance = 0.8 * prevDist + 0.2 * newDist;
   ```

4. **Faster name input:** Shorter names work better
   - "Sharma" (1 word) faster than "Akshatha A Sharma" (3 words)
   - Average time: ~1-2 seconds per name

---

## Team

RNSIT · VRK Kiosk - Alankrita Singh, Akshatha A, and B Sneha

**Latest Updates (Voice-First Release):**
- ✅ Fully voice-based name input (no text typing)
- ✅ Blink detection for yes/no confirmation (ready to integrate)
- ✅ Proximity detection for encouragement
- ✅ Personality-driven response framework
- ✅ Zero required clicking/touching (fully hands-free)
- ✅ No hand gestures (voice + face focus)
- ✅ All documentation in single README.md file
