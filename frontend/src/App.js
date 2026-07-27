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

    // Tick every 333ms (3fps ceiling), but only actually SEND that often
    // while nobody's been recognised yet (IDLE/DWELLING/RECOGNIZING) — that's
    // when fast detection matters. Once a session is ACTIVE/DEPARTING we only
    // need to notice "face gone" or "face swapped", so drop to ~1fps — this
    // keeps the backend free to run STT/LLM/TTS without CPU contention from
    // continuous face detection during conversation.
    let lastSendAt = 0;
    function startSending() {
      sendIntervalRef.current = setInterval(() => {
        const ws = wsRef.current;
        const video = hiddenVideoRef.current;
        const cvs = captureCanvasRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (!video || video.videoWidth === 0) return;

        const st = detStateRef.current;
        const minGap = (st === 'ACTIVE' || st === 'DEPARTING') ? 1000 : 333;
        const now = Date.now();
        if (now - lastSendAt < minGap) return;
        lastSendAt = now;

        const ctx = cvs.getContext('2d');
        cvs.width = video.videoWidth;
        cvs.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const b64 = cvs.toDataURL('image/jpeg', 0.7).split(',')[1];
        try { ws.send(JSON.stringify({ frame: b64 })); } catch (_) { }
      }, 333);
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

    const onEnded = () => {
      clearInterval(pollRef.current);
      pollRef.current = setInterval(poll, IDLE_POLL_MS);
      poll();
    };
    window.addEventListener('vrk-session-ended', onEnded);

    return () => {
      clearInterval(pollRef.current);
      clearTimeout(goodbyeTimer.current);
      window.removeEventListener('vrk-session-ended', onEnded);
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
      {screen === 'goodbye' && <GoodbyeScreen session={lastSession} />}
      {screen === 'idle' && <IdleScreen {...detectionProps} />}
    </>
  );
}