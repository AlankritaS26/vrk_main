import React, { useEffect, useState, useRef } from 'react';

/**
 * Idle / attract screen — camera + detection now live in App.js so they
 * keep running across screen switches. This component only displays the
 * shared stream and detection state passed down as props.
 */
export default function IdleScreen({ detState, identity, bbox, videoDims, camError, camStream }) {
  const [visible, setVisible] = useState(false);
  const [slide, setSlide] = useState(0);
  const [now, setNow] = useState(new Date());
  const videoRef = useRef(null);

  const capabilities = [
    { icon: '🎓', title: 'Admissions & Courses', text: '"What courses does RNSIT offer?"' },
    { icon: '💼', title: 'Placements', text: '"How are the placements here?"' },
    { icon: '🏛️', title: 'Departments', text: '"Tell me about the CSE department"' },
    { icon: '🏠', title: 'Hostel & Facilities', text: '"What are the hostel options?"' },
    { icon: '🗺️', title: 'Campus Directions', text: '"Where is the admission office?"' },
  ];

  useEffect(() => {
    setTimeout(() => setVisible(true), 100);
    const s = setInterval(() => setSlide(i => (i + 1) % capabilities.length), 3800);
    const c = setInterval(() => setNow(new Date()), 1000);
    return () => { clearInterval(s); clearInterval(c); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Attach the shared stream to this screen's own <video> for display.
  useEffect(() => {
    if (videoRef.current && camStream) {
      videoRef.current.srcObject = camStream;
    }
  }, [camStream]);

  const stateLabel = {
    IDLE: 'Walk up — I\'ll recognise you',
    DWELLING: 'I see you — hold still…',
    RECOGNIZING: 'Identifying…',
    ENROLLING: 'Getting to know you…',
    ACTIVE: identity ? `Welcome, ${identity}!` : 'Welcome!',
    DEPARTING: 'Goodbye!',
    COOLDOWN: 'Ready in a moment…',
  }[detState] || 'Looking for you…';

  const borderColor =
    detState === 'ACTIVE' ? '#43a047' :
      detState === 'DWELLING' || detState === 'RECOGNIZING' ? '#fb8c00' :
        '#1a237e';

  const cap = capabilities[slide];

  return (
    <div style={{ minHeight: '100vh', background: '#ffffff', fontFamily: "'Segoe UI', Arial, sans-serif", display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

      <div style={{ position: 'absolute', width: '560px', height: '560px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(26,35,126,0.07), transparent 65%)', top: '-180px', left: '-160px', animation: 'drift 14s ease-in-out infinite' }} />
      <div style={{ position: 'absolute', width: '480px', height: '480px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(66,165,245,0.08), transparent 65%)', bottom: '-140px', right: '-120px', animation: 'drift 17s ease-in-out infinite reverse' }} />

      <div style={{ background: '#1a237e', height: '6px', width: '100%', position: 'relative', zIndex: 1 }} />

      <div style={{ position: 'absolute', top: '24px', right: '32px', textAlign: 'right', zIndex: 2 }}>
        <div style={{ fontSize: '26px', fontWeight: '700', color: '#1a237e', letterSpacing: '0.5px' }}>
          {now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
        </div>
        <div style={{ fontSize: '12px', color: '#999', letterSpacing: '0.4px' }}>
          {now.toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long' })}
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '28px', padding: '40px', position: 'relative', zIndex: 1 }}>

        <div style={{ opacity: visible ? 1 : 0, transform: visible ? 'scale(1)' : 'scale(0.85)', transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
          <img src="/rnslogo.png" alt="RNSIT Logo"
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
            style={{ height: '90px', objectFit: 'contain', display: 'block', animation: 'gentleFloat 4s ease-in-out infinite', filter: 'drop-shadow(0 10px 24px rgba(26,35,126,0.18))' }} />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '30px', fontWeight: '800', color: '#1a237e', letterSpacing: '0.5px', lineHeight: '1.2' }}>
              RNS Institute of Technology
            </div>
            <div style={{ fontSize: '13px', color: '#777', marginTop: '4px', letterSpacing: '2.5px', textTransform: 'uppercase' }}>
              Autonomous Institution
            </div>
          </div>
        </div>

        {/* ── Live camera feed with face-detection overlay ───────────────── */}
        <div style={{
          opacity: visible ? 1 : 0, transition: 'opacity 0.8s ease 0.4s',
          position: 'relative', borderRadius: '18px', overflow: 'hidden',
          boxShadow: '0 8px 36px rgba(26,35,126,0.20)',
          border: `3px solid ${borderColor}`,
          transition: 'border-color 0.4s ease',
        }}>
          <video
            ref={videoRef}
            autoPlay playsInline muted
            style={{
              display: 'block',
              width: '360px', height: '270px',
              objectFit: 'cover',
              transform: 'scaleX(-1)',
              background: '#1a237e11',
            }}
          />

          {bbox && (
            <svg
              viewBox={`0 0 ${videoDims.w} ${videoDims.h}`}
              preserveAspectRatio="none"
              style={{
                position: 'absolute', top: 0, left: 0,
                width: '100%', height: '100%',
                pointerEvents: 'none',
                transform: 'scaleX(-1)',
              }}
            >
              <rect
                x={bbox.x} y={bbox.y} width={bbox.w} height={bbox.h}
                fill="none"
                stroke={borderColor}
                strokeWidth={Math.max(2, videoDims.w / 160)}
                rx="6"
              />
              <line x1={bbox.x} y1={bbox.y + 20} x2={bbox.x} y2={bbox.y} stroke={borderColor} strokeWidth={Math.max(3, videoDims.w / 100)} />
              <line x1={bbox.x} y1={bbox.y} x2={bbox.x + 20} y2={bbox.y} stroke={borderColor} strokeWidth={Math.max(3, videoDims.w / 100)} />
              <line x1={bbox.x + bbox.w - 20} y1={bbox.y} x2={bbox.x + bbox.w} y2={bbox.y} stroke={borderColor} strokeWidth={Math.max(3, videoDims.w / 100)} />
              <line x1={bbox.x + bbox.w} y1={bbox.y} x2={bbox.x + bbox.w} y2={bbox.y + 20} stroke={borderColor} strokeWidth={Math.max(3, videoDims.w / 100)} />
              <line x1={bbox.x + bbox.w} y1={bbox.y + bbox.h - 20} x2={bbox.x + bbox.w} y2={bbox.y + bbox.h} stroke={borderColor} strokeWidth={Math.max(3, videoDims.w / 100)} />
              <line x1={bbox.x + bbox.w} y1={bbox.y + bbox.h} x2={bbox.x + bbox.w - 20} y2={bbox.y + bbox.h} stroke={borderColor} strokeWidth={Math.max(3, videoDims.w / 100)} />
              <line x1={bbox.x + 20} y1={bbox.y + bbox.h} x2={bbox.x} y2={bbox.y + bbox.h} stroke={borderColor} strokeWidth={Math.max(3, videoDims.w / 100)} />
              <line x1={bbox.x} y1={bbox.y + bbox.h} x2={bbox.x} y2={bbox.y + bbox.h - 20} stroke={borderColor} strokeWidth={Math.max(3, videoDims.w / 100)} />
            </svg>
          )}

          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0,
            background: 'rgba(26,35,126,0.75)', backdropFilter: 'blur(6px)',
            color: '#fff', fontSize: '13px', fontWeight: '600',
            padding: '8px 14px', textAlign: 'center', letterSpacing: '0.3px',
          }}>
            {camError
              ? <span style={{ color: '#ff8a80' }}>{camError}</span>
              : stateLabel}
          </div>

          {(detState === 'DWELLING' || detState === 'RECOGNIZING' || detState === 'ENROLLING') && (
            <div style={{
              position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
              border: `3px solid ${borderColor}`,
              borderRadius: '18px',
              animation: 'scanPulse 1.4s ease-in-out infinite',
              pointerEvents: 'none',
            }} />
          )}
        </div>

        <div key={slide} style={{
          background: '#f8f9ff', border: '1.5px solid #e8eaf6', borderRadius: '16px',
          padding: '20px 36px', display: 'flex', alignItems: 'center', gap: '18px',
          boxShadow: '0 4px 20px rgba(26,35,126,0.08)', minWidth: '460px',
          animation: 'slideIn 0.45s ease'
        }}>
          <div style={{ fontSize: '30px' }}>{cap.icon}</div>
          <div style={{ textAlign: 'left' }}>
            <div style={{ fontSize: '15px', fontWeight: '700', color: '#1a237e' }}>{cap.title}</div>
            <div style={{ fontSize: '14px', color: '#777', marginTop: '3px', fontStyle: 'italic' }}>Try: {cap.text}</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          {capabilities.map((_, i) => (
            <div key={i} style={{ width: i === slide ? '22px' : '8px', height: '8px', borderRadius: '4px', background: i === slide ? '#1a237e' : '#c5cae9', transition: 'all 0.35s ease' }} />
          ))}
        </div>
      </div>

      <div style={{ background: '#1a237e', color: '#fff', padding: '12px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'relative', zIndex: 1 }}>
        <span style={{ fontSize: '12px', opacity: 0.85 }}>RNSIT Digital Receptionist System</span>
        <span style={{ fontSize: '12px', opacity: 0.85 }}>Bengaluru · 560098</span>
      </div>

      <style>{`
        @keyframes ring { 0%{transform:scale(0.62);opacity:1} 100%{transform:scale(1.55);opacity:0} }
        @keyframes breathe { 0%,100%{transform:scale(1)} 50%{transform:scale(1.06)} }
        @keyframes gentleFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-9px)} }
        @keyframes drift { 0%,100%{transform:translate(0,0)} 50%{transform:translate(46px,30px)} }
        @keyframes slideIn { from{opacity:0;transform:translateX(26px)} to{opacity:1;transform:translateX(0)} }
        @keyframes scanPulse { 0%,100%{opacity:0.9} 50%{opacity:0.3} }
      `}</style>
    </div>
  );
}