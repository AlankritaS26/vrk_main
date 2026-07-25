import React, { useEffect, useState, useRef } from 'react';
import IdleScreen from './IdleScreen';
import WelcomeScreen from './WelcomeScreen';
import GoodbyeScreen from './GoodbyeScreen';
import './index.css';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8001';

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

  useEffect(() => {
    async function poll() {
      try {
        const res = await fetch(BACKEND + '/session/current');
        const data = await res.json();

        if (data && data.active) {
          if (!prevActiveRef.current) {
            setMessages([]);
            clearTimeout(goodbyeTimer.current);
            // idle → active: drop to slow heartbeat; WS handles real-time ends
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
            }, 7000);   // goodbye stays long enough for the farewell voice
            // active → idle: resume fast poll
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

    // WelcomeScreen fires this right after ending the session OR when the
    // backend /ws WebSocket sends a session_end event.  Either way we want
    // an immediate poll then fast idle detection.
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

  if (screen === 'welcome')
    return <WelcomeScreen session={session} messages={messages} setMessages={setMessages} askingName={askingName} />;
  if (screen === 'goodbye')
    return <GoodbyeScreen session={lastSession} />;
  return <IdleScreen />;
}