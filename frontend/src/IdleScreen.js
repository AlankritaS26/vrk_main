import React, { useEffect, useState } from 'react';

/**
 * Idle / attract screen — must feel ALIVE from across the lobby.
 * A breathing radar ring signals "I'm watching for you", a rotating
 * capability carousel teaches visitors what to ask before they arrive,
 * and a live clock makes the kiosk quietly useful even when idle.
 */
export default function IdleScreen() {
  const [visible, setVisible] = useState(false);
  const [slide, setSlide] = useState(0);
  const [now, setNow] = useState(new Date());

  const capabilities = [
    { icon: '🎓', title: 'Admissions & Courses',  text: '"What courses does RNSIT offer?"' },
    { icon: '💼', title: 'Placements',            text: '"How are the placements here?"' },
    { icon: '🏛️', title: 'Departments',           text: '"Tell me about the CSE department"' },
    { icon: '🏠', title: 'Hostel & Facilities',   text: '"What are the hostel options?"' },
    { icon: '🗺️', title: 'Campus Directions',     text: '"Where is the admission office?"' },
  ];

  useEffect(() => {
    setTimeout(() => setVisible(true), 100);
    const s = setInterval(() => setSlide(i => (i + 1) % capabilities.length), 3800);
    const c = setInterval(() => setNow(new Date()), 1000);
    return () => { clearInterval(s); clearInterval(c); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cap = capabilities[slide];

  return (
    <div style={{ minHeight: '100vh', background: '#ffffff', fontFamily: "'Segoe UI', Arial, sans-serif", display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

      {/* slow ambient color drift behind everything */}
      <div style={{ position: 'absolute', width: '560px', height: '560px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(26,35,126,0.07), transparent 65%)', top: '-180px', left: '-160px', animation: 'drift 14s ease-in-out infinite' }} />
      <div style={{ position: 'absolute', width: '480px', height: '480px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(66,165,245,0.08), transparent 65%)', bottom: '-140px', right: '-120px', animation: 'drift 17s ease-in-out infinite reverse' }} />

      <div style={{ background: '#1a237e', height: '6px', width: '100%', position: 'relative', zIndex: 1 }} />

      {/* live clock — quietly useful even when idle */}
      <div style={{ position: 'absolute', top: '24px', right: '32px', textAlign: 'right', zIndex: 2 }}>
        <div style={{ fontSize: '26px', fontWeight: '700', color: '#1a237e', letterSpacing: '0.5px' }}>
          {now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
        </div>
        <div style={{ fontSize: '12px', color: '#999', letterSpacing: '0.4px' }}>
          {now.toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long' })}
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '34px', padding: '40px', position: 'relative', zIndex: 1 }}>

        {/* Logo + identity */}
        <div style={{ opacity: visible ? 1 : 0, transform: visible ? 'scale(1)' : 'scale(0.85)', transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '18px' }}>
          <img src="/rnslogo.png" alt="RNSIT Logo"
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
            style={{ height: '118px', objectFit: 'contain', display: 'block', animation: 'gentleFloat 4s ease-in-out infinite', filter: 'drop-shadow(0 10px 24px rgba(26,35,126,0.18))' }} />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '34px', fontWeight: '800', color: '#1a237e', letterSpacing: '0.5px', lineHeight: '1.2' }}>
              RNS Institute of Technology
            </div>
            <div style={{ fontSize: '14px', color: '#777', marginTop: '6px', letterSpacing: '2.5px', textTransform: 'uppercase' }}>
              Autonomous Institution
            </div>
          </div>
        </div>

        {/* Watching radar — "walk up, I'll see you" */}
        <div style={{ position: 'relative', width: '118px', height: '118px', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: visible ? 1 : 0, transition: 'opacity 0.8s ease 0.4s' }}>
          <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid rgba(26,35,126,0.30)', animation: 'ring 2.8s ease-out infinite' }} />
          <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid rgba(26,35,126,0.20)', animation: 'ring 2.8s ease-out infinite 0.95s' }} />
          <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid rgba(26,35,126,0.10)', animation: 'ring 2.8s ease-out infinite 1.9s' }} />
          <div style={{ width: '76px', height: '76px', borderRadius: '50%', background: 'linear-gradient(135deg, #1a237e, #3949ab)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 8px 26px rgba(26,35,126,0.35)', animation: 'breathe 2.8s ease-in-out infinite' }}>
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.7">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>
            </svg>
          </div>
        </div>

        <div style={{ textAlign: 'center', opacity: visible ? 1 : 0, transition: 'opacity 0.8s ease 0.6s' }}>
          <div style={{ fontSize: '22px', fontWeight: '700', color: '#1a237e' }}>
            Walk up to begin
          </div>
          <div style={{ fontSize: '15px', color: '#888', marginTop: '6px' }}>
            I'll recognise you and greet you automatically
          </div>
        </div>

        {/* Rotating capability carousel — teaches what to ask */}
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

        {/* carousel position dots */}
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
      `}</style>
    </div>
  );
}