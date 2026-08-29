# VRK Kiosk — Frontend Application

React 18 single-page kiosk interface providing a fully hands-free, voice-driven user experience with animated avatar graphics, audio waveform visualizers, real-time presence awareness, and instant synchronized voice interactions.

Operates in an automated kiosk display window with `--autoplay-policy=no-user-gesture-required` launched via `python run.py`.

---

## 🚀 Running Standalone (Frontend Development)

For UI styling, component tweaks, or standalone frontend development:

```powershell
cd frontend
npm install
npm start
```

> **Note:** Running `npm start` automatically triggers `node scripts/copyVadAssets.js` during the `prestart` phase. This copies the Silero VAD ONNX model files and WebAssembly runtime binaries directly into `public/` so browser-side voice activity detection functions smoothly without CORS or bundler issues.

---

## 🎨 User Experience & Screen Flow

```
   ┌───────────────────────────────────────────────────────────┐
   │                  IdleScreen (Attract Mode)                │
   │  • Real-time Live Clock & Date                            │
   │  • Interactive Capabilities Carousel                      │
   │  • Vision State Display ("Walk up", "I see you", etc.)    │
   │  • MediaPipe Camera Dwell Detection                       │
   └─────────────────────────────┬─────────────────────────────┘
                                 │ Visitor Dwells / Approaches
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │              WelcomeScreen (Active Conversation)          │
   │                                                           │
   │  ┌─────────────────────────┐  ┌────────────────────────┐  │
   │  │   Nova Animated Avatar  │  │  Voice Waveform Visual │  │
   │  │  • Lip-sync to speech   │  │  • Real-time Silero VAD│  │
   │  │  • Intent animations    │  │  • Hands-free listening│  │
   │  └─────────────────────────┘  └────────────────────────┘  │
   │                                                           │
    │  • Hands-Free Name Onboarding & 5s Auto-Guest Default     │
    │  • Dynamic Mid-Session Name Change & DB Synchronization   │
    │  • Double-Blink Affirmation & NATO Spelling Mode Fallback │
    │  • Synchronized Text & Voice Streaming with Auto-Scroll   │
    │  • Multi-intent Personality Reactions & Contextual Hints  │
    │  • Session Re-engagement & Fallback Departure Prompts     │
   └─────────────────────────────┬─────────────────────────────┘
                                 │ "Thank you" / "Goodbye" / Inactivity Timeout
                                 ▼
   ┌───────────────────────────────────────────────────────────┐
   │                  GoodbyeScreen (Farewell)                 │
   │  • Personalized farewell ("Goodbye, {Name}!")             │
   │  • RNSIT Official Branding & Logo                         │
   │  • Audio playback overlay before transition to Idle       │
   └───────────────────────────────────────────────────────────┘
```

---

## 📁 Key Files & Components

### 1. `src/App.js` — State Coordinator & WebSocket Hub
- Manages top-level application state (`idle`, `welcome`, `goodbye`).
- Maintains continuous WebSocket connection to `ws://127.0.0.1:8001/ws/detect` and `ws://127.0.0.1:8001/ws`.
- Parses incoming real-time telemetry: visitor identity, verification state, face bounding boxes, and blink detections (`blink_detected`, `ear_left`, `ear_right`).
- Orchestrates clean transitions between kiosk display states.

### 2. `src/WelcomeScreen.js` — Main Voice Interaction Interface
- **Voice Name Onboarding**: When a new visitor is detected, the kiosk asks *"Hi! I'm Nova. What's your name?"* and activates Silero VAD without requiring any button clicks or touch input. Transcribes spoken names (e.g. *"Akshatha"*, *"I am Sneha"*), displays visual confirmation, and automatically starts the personalized session after 3 seconds.
- **Audio Synchronized Playback**: Utilizes Web Audio API to play chunked neural TTS audio without audio gaps, automatically scrolling conversation bubbles into view in sync with spoken sentences.
- **Re-engagement Logic**: Monitors dwell time and conversation gaps to prompt visitors (*"Are you still there?"*) or gracefully transition to the farewell state.

### 3. `src/IdleScreen.js` — Attract Mode
- Displays live vision state labels matching camera detection state (`IDLE`, `DWELLING`, `RECOGNIZING`, `ACTIVE`).
- Features dynamic carousel cards showcasing campus facilities, admissions info, department locations, and placement highlights.

### 4. `src/GoodbyeScreen.js` — Farewell Transition
- Displays a warm, personalized closing card (*"Goodbye, {Name}! Have a wonderful day."*) alongside official college branding.
- Avoids robotic fallback labels (suppresses impersonal "Guest" strings for anonymous sessions).
- Automatically resets the kiosk state machine back to `IdleScreen` after speech completion.

### 5. `src/AriaAvatar.js` (Nova Avatar) — SVG / Canvas Character Engine
- Interactive vector avatar with smooth micro-animations.
- States: `idle`, `listening`, `thinking`, `speaking`, `happy`, `curious`, `bow`, `confused`.
- Procedural eye blinking and ambient breathing cycles.
- Dynamic mouth movement amplitude modulation linked to Web Audio API playback frequencies.

### 6. `src/avatarReactions.js` — Emotion & Personality Utilities
- Helper algorithms analyzing user utterances for intent classification (`curious`, `happy`, `thanks`, `greeting`, `confused`).
- Computes time-aware greetings (*"Good morning! ☀️"*, *"Good afternoon! 🌤️"*).
- Returns personality prefixes and suffixes to provide a warm, conversational demeanor.

### 7. `src/kioskMic.js` — Client-Side Audio Pipeline
- Connects to browser `navigator.mediaDevices.getUserMedia` at 16,000 Hz mono PCM.
- Runs `@ricky0123/vad-web` with Silero VAD ONNX model to capture clean speech segments while eliminating background lobby noise.
- Transmits raw audio buffers to `/stt/pcm` for near-instant transcription.

---

## 🛠️ Assets & Scripts

- `scripts/copyVadAssets.js` — Build automation script ensuring `silero_vad.onnx`, `ort-wasm-simd.wasm`, and related dependencies are bundled into `public/` for reliable offline execution.
- `public/rnslogo.png` — High-resolution institution logo rendered across the Idle, Conversation, and Goodbye screens.
- `src/index.css` — Custom glassmorphic styles, smooth transitions, pulsing glows, and responsive kiosk viewport definitions.