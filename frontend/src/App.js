import React, { useEffect, useState, useRef } from 'react';
import IdleScreen from './IdleScreen';
import WelcomeScreen from './WelcomeScreen';
import GoodbyeScreen from './GoodbyeScreen';
import './index.css';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [screen,      setScreen]      = useState('idle');
  const [session,     setSession]     = useState(null);
  const [lastSession, setLastSession] = useState(null);
  const [messages,    setMessages]    = useState([]);
  const pollRef      = useRef(null);
  const goodbyeTimer = useRef(null);
  const prevActiveRef = useRef(false);

  useEffect(() => {
    async function poll() {
      try {
        const res  = await fetch(BACKEND + '/session/current');
        const data = await res.json();

        if (data && data.active) {
          if (!prevActiveRef.current) {
            setMessages([]);
            clearTimeout(goodbyeTimer.current);
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
          }
          prevActiveRef.current = false;
        }
      } catch(e) {
        console.error('[poll error]', e);
      }
    }

    poll();
    pollRef.current = setInterval(poll, 750);   // snappier screen transitions

    // WelcomeScreen fires this right after ending the session, so the
    // goodbye screen appears instantly (while the farewell voice plays)
    // instead of waiting up to 1.5s for the next poll.
    const onEnded = () => poll();
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