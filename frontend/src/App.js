import React, { useEffect, useState, useRef } from 'react';
import IdleScreen from './IdleScreen';
import WelcomeScreen from './WelcomeScreen';
import GoodbyeScreen from './GoodbyeScreen';
import './index.css';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8001';
const WS_BACKEND = BACKEND.replace(/^http/, 'ws');

// Polling intervals — fast only when idle (waiting for face recognition),
// slow heartbeat during an active session (WebSocket handles real-time ends).
const IDLE_POLL_MS = 750;   // snappy idle→welcome transition
const ACTIVE_POLL_MS = 12000;   // ~12 s fallback; WS fires vrk-session-ended first

export default function App() {
  const [screen, setScreen] = useState('idle');
  const [session, setSession] = useState(null);
  const [lastSession, setLastSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [lastFarewell, setLastFarewell] = useState('');
  const pollRef = useRef(null);
  const goodbyeTimer = useRef(null);
  const prevActiveRef = useRef(false);

  // ── Shared camera + detection WebSocket — lives for the whole app ────────
  // Runs continuously across idle/welcome/goodbye so presence detection
  // (departure timeout, face-swap) never stops just because the screen
  // switched to WelcomeScreen.
  const [detState, setDetState] = useState('IDLE');
  const [identity, setIdentity] = useState('');
  const [bbox, setBbox] = useState(null);
  const [videoDims, setVideoDims] = useState({ w: 640, h: 480 });
  const [camError, setCamError] = useState(null);
  const [camStream, setCamStream] = useState(null);

  const hiddenVideoRef = useRef(null);      // used only for frame capture
  const captureCanvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const sendIntervalRef = useRef(null);

  const detStateRef = useRef('IDLE');
  useEffect(() => { detStateRef.current = detState; }, [detState]);

  useEffect(() => {
    let stopped = false;

    function connectWs() {
      if (stopped) return;
      const ws = new WebSocket(WS_BACKEND + '/ws/detect');
      wsRef.current = ws;

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          setDetState(data.state || 'IDLE');
          setIdentity(data.identity || '');
          setBbox(data.present && data.bbox ? data.bbox : null);
        } catch (_) { }
      };

      ws.onclose = () => {
        if (!stopped) setTimeout(connectWs, 2000);
      };
    }

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        if (!stopped) setCamError(
          'Camera unavailable: open the kiosk at http://localhost:3000 (not the machine hostname)');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        });
        if (stopped) { stream.getTracks().forEach(t => t.stop()); return; }
        streamRef.current = stream;
        setCamStream(stream);
        if (hiddenVideoRef.current) {
          hiddenVideoRef.current.srcObject = stream;
          hiddenVideoRef.current.onloadedmetadata = () => {
            const w = hiddenVideoRef.current.videoWidth;
            const h = hiddenVideoRef.current.videoHeight;
            if (w && h) setVideoDims({ w, h });
          };
        }
      } catch (err) {
        if (!stopped) setCamError('Camera unavailable: ' + err.message);
      }
    }

    function startSending() {
      sendIntervalRef.current = setInterval(() => {
        const ws = wsRef.current;
        const video = hiddenVideoRef.current;
        const cvs = captureCanvasRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (!video || video.videoWidth === 0) return;

        const ctx = cvs.getContext('2d');
        const w = video.videoWidth || 640;
        const h = video.videoHeight || 480;
        if (cvs.width !== w || cvs.height !== h) {
          cvs.width = w;
          cvs.height = h;
        }
        ctx.drawImage(video, 0, 0, w, h);
        const b64 = cvs.toDataURL('image/jpeg', 0.65).split(',')[1];
        try { ws.send(JSON.stringify({ frame: b64 })); } catch (_) { }
      }, 250);
    }

    startCamera();
    connectWs();
    startSending();

    return () => {
      stopped = true;
      clearInterval(sendIntervalRef.current);
      wsRef.current?.close();
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  // ── Session polling ────────────────────────────────────────────────────
  useEffect(() => {
    async function poll() {
      try {
        const res = await fetch(BACKEND + '/session/current');
        const data = await res.json();

        if (data && data.active) {
          if (!prevActiveRef.current) {
            setMessages([]);
            clearTimeout(goodbyeTimer.current);
            clearInterval(pollRef.current);
            pollRef.current = setInterval(poll, ACTIVE_POLL_MS);
          }
          prevActiveRef.current = true;
          setSession(data);
          setScreen('welcome');
        } else {
          if (prevActiveRef.current) {
            setLastFarewell('');
            setSession(current => { setLastSession(current); return null; });
            setScreen('goodbye');
            goodbyeTimer.current = setTimeout(() => {
              setScreen('idle');
              setLastSession(null);
            }, 7000);
            clearInterval(pollRef.current);
            pollRef.current = setInterval(poll, IDLE_POLL_MS);
          }
          prevActiveRef.current = false;
        }
      } catch (e) {
        console.error('[poll error]', e);
      }
    }

    poll();
    pollRef.current = setInterval(poll, IDLE_POLL_MS);

    const onEnded = (ev) => {
      // Capture the farewell text if WelcomeScreen sent it with the event
      if (ev.detail?.farewell) setLastFarewell(ev.detail.farewell);
      // Capture the visitor's real name from the event so the Goodbye screen
      // always shows the correct name even if session state was stale.
      if (ev.detail?.userName && ev.detail.userName !== 'Unknown') {
        setLastSession(current => current
          ? { ...current, user_name: ev.detail.userName }
          : { user_name: ev.detail.userName }
        );
      }
      clearInterval(pollRef.current);
      pollRef.current = setInterval(poll, IDLE_POLL_MS);
      poll();
    };
    window.addEventListener('vrk-session-ended', onEnded);

    // ── Instant screen switch on session START ──────────────────────────
    // Without this, App.js only learns a new visitor exists on its next
    // poll tick — up to IDLE_POLL_MS (750ms) of pure waiting BEFORE the
    // welcome screen even appears, let alone before the greeting can play.
    // The backend already broadcasts "session_start" the moment a session
    // begins (see /session/start and /visitor/greet in main.py) — listen
    // for it directly and poll() immediately instead of waiting.
    let startWs;
    let startWsStopped = false;
    function connectStartWs() {
      if (startWsStopped) return;
      startWs = new WebSocket(WS_BACKEND + '/ws');
      startWs.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'session_start' || msg.type === 'asking_name') {
            clearInterval(pollRef.current);
            poll();                          // switch screens right now
            pollRef.current = setInterval(poll, ACTIVE_POLL_MS);
          } else if (msg.type === 'session_update') {
            // Guest gave their name — update session state IMMEDIATELY so the
            // Goodbye screen and header always show the real name. Don't wait
            // for the slow 12-second poll tick.
            if (msg.session) {
              setSession(prev => prev ? { ...prev, ...msg.session } : msg.session);
            } else if (msg.user_name) {
              setSession(prev => prev ? { ...prev, user_name: msg.user_name, face_id: msg.face_id || prev.face_id } : prev);
            }
            // Also re-sync with backend to confirm
            poll();
          }
        } catch (_) { }
      };
      startWs.onclose = () => { if (!startWsStopped) setTimeout(connectStartWs, 2000); };
    }
    connectStartWs();

    return () => {
      clearInterval(pollRef.current);
      clearTimeout(goodbyeTimer.current);
      window.removeEventListener('vrk-session-ended', onEnded);
      startWsStopped = true;
      startWs?.close();
    };
  }, []);

  const askingName = session?.asking_name === true;

  const detectionProps = { detState, identity, bbox, videoDims, camError, camStream };

  return (
    <>
      {/* Hidden video element used only to feed the capture canvas — always
          mounted so the camera + WS never drop when the screen switches. */}
      <video ref={hiddenVideoRef} autoPlay playsInline muted style={{ display: 'none' }} />
      <canvas ref={captureCanvasRef} style={{ display: 'none' }} />

      {screen === 'welcome' && (
        <WelcomeScreen session={session} messages={messages} setMessages={setMessages}
          askingName={askingName} {...detectionProps} />
      )}
      {screen === 'goodbye' && <GoodbyeScreen session={lastSession} farewell={lastFarewell} />}
      {screen === 'idle' && <IdleScreen {...detectionProps} />}
    </>
  );
}