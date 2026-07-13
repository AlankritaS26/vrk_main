# VRK Kiosk — Frontend

React (CRA) kiosk interface: idle attract screen, voice conversation
screen, and goodbye screen.

**Do not start this directly for kiosk use.** Run everything from the
repository root with `venv\Scripts\python.exe run.py` — it starts the
backend and camera detection first, then this frontend, and opens the
kiosk browser window with autoplay enabled (required for the greeting
to speak before any user gesture).

Standalone frontend development only:

    npm install
    npm start        # prestart auto-copies VAD/onnx assets into public/

Configuration: `.env → REACT_APP_BACKEND_URL` (defaults to
http://127.0.0.1:8000).

Key files: `src/WelcomeScreen.js` (conversation + voice pipeline),
`src/kioskMic.js` (browser VAD capture), `src/IdleScreen.js`,
`src/GoodbyeScreen.js`, `scripts/copyVadAssets.js` (the CRA .mjs fix).