import React, { useEffect, useRef } from 'react';
import IdlePhotoBackground from './BackgroundVideo';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8001';

export default function GoodbyeScreen({ session, farewell }) {
  // Layer 1: use session.user_name (exclude only "Unknown" — "Guest" is a
  //          valid choice the visitor made, and real names are always valid)
  let name = (session?.user_name && session.user_name !== 'Unknown') ? session.user_name : '';

  // Layer 2: try to extract the name from the farewell text itself.
  if (!name && farewell) {
    const m = farewell.match(/Goodbye,\s+([A-Z][a-zA-Z\s]+?)!/);
    if (m) name = m[1].trim();
  }

  // Suppress "Guest" in the displayed line
  const displayName = (name && name !== 'Guest') ? name : '';
  const spokenFarewell = farewell || (displayName
    ? `Goodbye, ${displayName}! Wishing you a wonderful day ahead.`
    : 'Goodbye! Wishing you a wonderful day ahead.');

  const spokenRef = useRef(false);

  useEffect(() => {
    if (spokenRef.current) return;
    spokenRef.current = true;

    async function speakFarewell() {
      try {
        const res = await fetch(BACKEND + '/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: spokenFarewell })
        });
        const data = await res.json();
        if (data.audio) {
          const snd = new Audio(`data:audio/wav;base64,${data.audio}`);
          await snd.play();
          return;
        }
      } catch (_) { }

      // Browser TTS Fallback
      try {
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(spokenFarewell);
        u.lang = 'en-US';
        window.speechSynthesis.speak(u);
      } catch (_) { }
    }

    speakFarewell();
  }, [spokenFarewell]);

  return (
    <div style={{
      minHeight: '100vh', background: 'transparent', position: 'relative',
      fontFamily: "'Segoe UI', Arial, sans-serif",
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: '28px'
    }}>
      <IdlePhotoBackground />

      <img
        src="/rnslogo.png" alt="RNSIT"
        onError={(e) => { e.currentTarget.style.display = 'none'; }}
        style={{ height: '110px', objectFit: 'contain', animation: 'gentleFloat 3.5s ease-in-out infinite', position: 'relative', zIndex: 1 }}
      />

      <div style={{
        background: 'transparent', backdropFilter: 'blur(3px)',
        WebkitBackdropFilter: 'blur(3px)',
        borderRadius: '20px',
        padding: '52px 72px', textAlign: 'center',
        boxShadow: '0 16px 36px rgba(2,8,30,0.08)',
        border: '1.5px solid rgba(255,255,255,0.18)', maxWidth: '620px',
        animation: 'riseIn 0.5s ease', position: 'relative', zIndex: 1
      }}>
        {/* Check mark in a soft ring — closure, not celebration */}
        <div style={{
          width: '76px', height: '76px', borderRadius: '50%',
          background: 'rgba(255,255,255,0.16)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 26px', border: '2px solid rgba(186,230,253,0.7)'
        }}>
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
               stroke="#bae6fd" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>

        <div style={{ fontSize: '34px', fontWeight: '800', color: '#ffffff', letterSpacing: '0.2px', textShadow: '0 2px 10px rgba(0,0,0,0.3)' }}>
          Thank you for visiting
        </div>

        <div style={{ fontSize: '19px', color: 'rgba(255,255,255,0.88)', marginTop: '14px', lineHeight: '1.6' }}>
          Goodbye{displayName ? <>, <strong style={{ color: '#bae6fd' }}>{displayName}</strong></> : ''}.
          Wishing you a wonderful day ahead.
        </div>

        <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.65)', marginTop: '22px' }}>
          Returning to the welcome screen shortly
        </div>

        {/* animated progress line */}
        <div style={{
          height: '4px', width: '160px', margin: '18px auto 0',
          borderRadius: '2px', background: 'rgba(255,255,255,0.22)',
          overflow: 'hidden', position: 'relative'
        }}>
          <div style={{
            position: 'absolute', inset: 0, borderRadius: '2px',
            background: 'linear-gradient(90deg, #38bdf8, #bae6fd)',
            transformOrigin: 'left', animation: 'drain 5s linear forwards'
          }} />
        </div>
      </div>

      <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.68)', letterSpacing: '0.4px', position: 'relative', zIndex: 1 }}>
        RNS Institute of Technology &middot; Digital Receptionist
      </div>

      <style>{`
        @keyframes riseIn { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        @keyframes gentleFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
        @keyframes drain { from{transform:scaleX(1)} to{transform:scaleX(0)} }
      `}</style>
    </div>
  );
}