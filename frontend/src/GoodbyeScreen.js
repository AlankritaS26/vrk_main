import React from 'react';

export default function GoodbyeScreen({ session }) {
  const name = session?.user_name && session.user_name !== 'Unknown' ? session.user_name : '';
  return (
    <div style={{
      minHeight: '100vh', background: '#f5f6fa',
      fontFamily: "'Segoe UI', Arial, sans-serif",
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: '28px'
    }}>
      <img
        src="/rnslogo.png" alt="RNSIT"
        onError={(e) => { e.currentTarget.style.display = 'none'; }}
        style={{ height: '110px', objectFit: 'contain', animation: 'gentleFloat 3.5s ease-in-out infinite' }}
      />

      <div style={{
        background: '#ffffff', borderRadius: '20px',
        padding: '52px 72px', textAlign: 'center',
        boxShadow: '0 8px 40px rgba(26,35,126,0.10)',
        border: '1.5px solid #e8eaf6', maxWidth: '620px',
        animation: 'riseIn 0.5s ease'
      }}>
        {/* Check mark in a soft ring — closure, not celebration */}
        <div style={{
          width: '76px', height: '76px', borderRadius: '50%',
          background: 'linear-gradient(135deg, #e8eaf6, #f5f6ff)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          margin: '0 auto 26px', border: '2px solid #c5cae9'
        }}>
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
               stroke="#1a237e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>

        <div style={{ fontSize: '34px', fontWeight: '800', color: '#1a237e', letterSpacing: '0.2px' }}>
          Thank you for visiting
        </div>

        <div style={{ fontSize: '19px', color: '#444', marginTop: '14px', lineHeight: '1.6' }}>
          Goodbye{name ? <>, <strong style={{ color: '#1a237e' }}>{name}</strong></> : ''}.
          Wishing you a wonderful day ahead.
        </div>

        <div style={{ fontSize: '14px', color: '#9aa0b4', marginTop: '22px' }}>
          Returning to the welcome screen shortly
        </div>

        {/* animated progress line replaces the static dash */}
        <div style={{
          height: '4px', width: '160px', margin: '18px auto 0',
          borderRadius: '2px', background: '#e8eaf6',
          overflow: 'hidden', position: 'relative'
        }}>
          <div style={{
            position: 'absolute', inset: 0, borderRadius: '2px',
            background: 'linear-gradient(90deg, #1a237e, #5c6bc0)',
            transformOrigin: 'left', animation: 'drain 5s linear forwards'
          }} />
        </div>
      </div>

      <div style={{ fontSize: '13px', color: '#b0b4c8', letterSpacing: '0.4px' }}>
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