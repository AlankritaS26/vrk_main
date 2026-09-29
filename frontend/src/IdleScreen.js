import React, { useEffect, useState, useRef } from 'react';

/**
 * IdleScreen — two stacked layers:
 *
 *  BASE LAYER  (always rendered): the original white screen with the live
 *              camera feed, face-detection bounding-box overlay, rotating
 *              capability cards, clock, and RNSIT branding — untouched.
 *
 *  TOP LAYER   (attract overlay, white background): animated screen with
 *              rotating RNSIT highlight cards, "Walk up to begin" CTA, and
 *              a scrolling ticker.  It sits on top of the base layer and
 *              smoothly fades out the moment the camera detects a face
 *              (DWELLING / RECOGNIZING / ENROLLING / ACTIVE), revealing the
 *              camera screen behind it.  App.js then switches to WelcomeScreen
 *              once a full session is live.
 */
export default function IdleScreen({ detState, identity, bbox, videoDims, camError, camStream }) {

  // ── Shared ───────────────────────────────────────────────────────────────
  const [now, setNow]         = useState(new Date());
  const [visible, setVisible] = useState(false);   // base layer entry
  const [mounted, setMounted] = useState(false);   // overlay entry

  // Base layer capability cards
  const [baseSlide, setBaseSlide] = useState(0);

  // Overlay slides
  const [ovSlide, setOvSlide] = useState(0);
  const [ovDot,   setOvDot]   = useState(0);

  // Particle rotation
  const [angle, setAngle] = useState(0);
  const animRef = useRef(null);
  const lastRef = useRef(null);

  // Video ref for base camera display
  const videoRef = useRef(null);

  // ── Overlay fades away when a face is detected ───────────────────────────
  const faceDetected = ['DWELLING', 'RECOGNIZING', 'ENROLLING', 'ACTIVE'].includes(detState);

  // ── Base layer: capability cards ─────────────────────────────────────────
  const capabilities = [
    { icon: '🎓', title: 'Admissions & Courses',  text: '"What courses does RNSIT offer?"' },
    { icon: '💼', title: 'Placements',             text: '"How are the placements here?"' },
    { icon: '🏛️', title: 'Departments',            text: '"Tell me about the CSE department"' },
    { icon: '🏠', title: 'Hostel & Facilities',    text: '"What are the hostel options?"' },
    { icon: '🗺️', title: 'Campus Directions',      text: '"Where is the admission office?"' },
  ];

  // ── Overlay: rotating highlights ─────────────────────────────────────────
  const highlights = [
    { icon: '🎓', headline: 'Top-Ranked Engineering',  sub: 'NAAC A+ Accredited · NBA Certified',      accent: '#1a237e', bg: '#e8eaf6' },
    { icon: '💼', headline: '90%+ Placement Record',   sub: 'Amazon · Microsoft · Infosys · Wipro',    accent: '#1b5e20', bg: '#e8f5e9' },
    { icon: '🔬', headline: '15+ Research Centres',    sub: 'AI · IoT · Robotics · Cybersecurity',     accent: '#b45309', bg: '#fff7ed' },
    { icon: '🏠', headline: 'World-Class Campus',      sub: 'Hostel · Labs · Sports · Cafeteria',      accent: '#880e4f', bg: '#fce4ec' },
    { icon: '🏛️', headline: '20+ Departments',         sub: 'Engineering · Management · Sciences',     accent: '#0d47a1', bg: '#e3f2fd' },
  ];

  // ── Overlay: ticker ───────────────────────────────────────────────────────
  const tickers = [
    '🎓  B.E. / M.Tech / MBA · Admissions Open',
    '📞  +91-80-2319-0000',
    '🌐  www.rnsit.ac.in',
    '📍  Dr. Vishnuvardhan Road, Channasandra, Bengaluru – 560 098',
    '🏆  Ranked among Top Engineering Colleges in Karnataka',
  ];

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  useEffect(() => {
    setTimeout(() => { setVisible(true); setMounted(true); }, 100);

    const clock = setInterval(() => setNow(new Date()), 1000);
    const bs    = setInterval(() => setBaseSlide(i => (i + 1) % capabilities.length), 3800);
    const ovs   = setInterval(() => {
      setOvSlide(i => (i + 1) % highlights.length);
      setOvDot(i   => (i + 1) % highlights.length);
    }, 4000);

    const tick = ts => {
      if (lastRef.current !== null) setAngle(a => a + (ts - lastRef.current) * 0.02);
      lastRef.current = ts;
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);

    return () => {
      clearInterval(clock); clearInterval(bs); clearInterval(ovs);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (videoRef.current && camStream) videoRef.current.srcObject = camStream;
  }, [camStream]);

  // ── Derived ───────────────────────────────────────────────────────────────
  const cap = capabilities[baseSlide];
  const h   = highlights[ovSlide];

  const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false });
  const dateStr = now.toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long' });

  const stateLabel = ({
    IDLE:        "Walk up — I'll recognise you",
    DWELLING:    'I see you — hold still…',
    RECOGNIZING: 'Identifying…',
    ENROLLING:   'Getting to know you…',
    ACTIVE:      identity ? `Welcome, ${identity}!` : 'Welcome!',
    DEPARTING:   'Goodbye!',
    COOLDOWN:    'Ready in a moment…',
  })[detState] || 'Looking for you…';

  const borderColor =
    detState === 'ACTIVE'                                 ? '#43a047' :
    detState === 'DWELLING' || detState === 'RECOGNIZING' ? '#fb8c00' :
                                                            '#1a237e';

  // Soft particle orbs for overlay
  const orbs = [0, 120, 240].map((base, i) => {
    const rad = ((angle + base) % 360) * (Math.PI / 180);
    return { x: 50 + 30 * Math.cos(rad), y: 42 + 17 * Math.sin(rad), size: [20,14,24][i], color: ['#1a237e','#1b5e20','#b45309'][i] };
  });

  return (
    <div style={{ minHeight: '100vh', position: 'relative', overflow: 'hidden', fontFamily: "'Segoe UI', Arial, sans-serif" }}>

      {/* ══════════════════════════════════════════════════════════
          BASE LAYER — original white camera screen (always visible)
          ══════════════════════════════════════════════════════════ */}
      <div style={{ minHeight: '100vh', background: '#ffffff', display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

        {/* Bg blobs */}
        <div style={{ position:'absolute', width:'560px', height:'560px', borderRadius:'50%', background:'radial-gradient(circle, rgba(26,35,126,0.07), transparent 65%)', top:'-180px', left:'-160px', animation:'drift 14s ease-in-out infinite' }} />
        <div style={{ position:'absolute', width:'480px', height:'480px', borderRadius:'50%', background:'radial-gradient(circle, rgba(66,165,245,0.08), transparent 65%)', bottom:'-140px', right:'-120px', animation:'drift 17s ease-in-out infinite reverse' }} />

        <div style={{ background:'#1a237e', height:'6px', width:'100%', position:'relative', zIndex:1 }} />

        {/* Clock */}
        <div style={{ position:'absolute', top:'24px', right:'32px', textAlign:'right', zIndex:2 }}>
          <div style={{ fontSize:'26px', fontWeight:'700', color:'#1a237e', letterSpacing:'0.5px' }}>
            {now.toLocaleTimeString('en-IN', { hour:'2-digit', minute:'2-digit' })}
          </div>
          <div style={{ fontSize:'12px', color:'#999', letterSpacing:'0.4px' }}>
            {now.toLocaleDateString('en-IN', { weekday:'long', day:'numeric', month:'long' })}
          </div>
        </div>

        <div style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:'28px', padding:'40px', position:'relative', zIndex:1 }}>

          {/* Logo */}
          <div style={{ opacity:visible?1:0, transform:visible?'scale(1)':'scale(0.85)', transition:'all 0.8s cubic-bezier(0.34,1.56,0.64,1)', display:'flex', flexDirection:'column', alignItems:'center', gap:'14px' }}>
            <img src="/rnslogo.png" alt="RNSIT Logo"
              onError={e => { e.currentTarget.style.display='none'; }}
              style={{ height:'90px', objectFit:'contain', display:'block', animation:'gentleFloat 4s ease-in-out infinite', filter:'drop-shadow(0 10px 24px rgba(26,35,126,0.18))' }} />
            <div style={{ textAlign:'center' }}>
              <div style={{ fontSize:'30px', fontWeight:'800', color:'#1a237e', letterSpacing:'0.5px', lineHeight:'1.2' }}>RNS Institute of Technology</div>
              <div style={{ fontSize:'13px', color:'#777', marginTop:'4px', letterSpacing:'2.5px', textTransform:'uppercase' }}>Autonomous Institution</div>
            </div>
          </div>

          {/* Camera feed */}
          <div style={{ opacity:visible?1:0, position:'relative', borderRadius:'18px', overflow:'hidden', boxShadow:'0 8px 36px rgba(26,35,126,0.20)', border:`3px solid ${borderColor}`, transition:'opacity 0.8s ease 0.4s, border-color 0.4s ease' }}>
            <video ref={videoRef} autoPlay playsInline muted
              style={{ display:'block', width:'360px', height:'270px', objectFit:'cover', transform:'scaleX(-1)', background:'#1a237e11' }} />

            {bbox && (
              <svg viewBox={`0 0 ${videoDims.w} ${videoDims.h}`} preserveAspectRatio="none"
                style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%', pointerEvents:'none', transform:'scaleX(-1)' }}>
                <rect x={bbox.x} y={bbox.y} width={bbox.w} height={bbox.h} fill="none" stroke={borderColor} strokeWidth={Math.max(2,videoDims.w/160)} rx="6" />
                <line x1={bbox.x} y1={bbox.y+20} x2={bbox.x} y2={bbox.y} stroke={borderColor} strokeWidth={Math.max(3,videoDims.w/100)} />
                <line x1={bbox.x} y1={bbox.y} x2={bbox.x+20} y2={bbox.y} stroke={borderColor} strokeWidth={Math.max(3,videoDims.w/100)} />
                <line x1={bbox.x+bbox.w-20} y1={bbox.y} x2={bbox.x+bbox.w} y2={bbox.y} stroke={borderColor} strokeWidth={Math.max(3,videoDims.w/100)} />
                <line x1={bbox.x+bbox.w} y1={bbox.y} x2={bbox.x+bbox.w} y2={bbox.y+20} stroke={borderColor} strokeWidth={Math.max(3,videoDims.w/100)} />
                <line x1={bbox.x+bbox.w} y1={bbox.y+bbox.h-20} x2={bbox.x+bbox.w} y2={bbox.y+bbox.h} stroke={borderColor} strokeWidth={Math.max(3,videoDims.w/100)} />
                <line x1={bbox.x+bbox.w} y1={bbox.y+bbox.h} x2={bbox.x+bbox.w-20} y2={bbox.y+bbox.h} stroke={borderColor} strokeWidth={Math.max(3,videoDims.w/100)} />
                <line x1={bbox.x+20} y1={bbox.y+bbox.h} x2={bbox.x} y2={bbox.y+bbox.h} stroke={borderColor} strokeWidth={Math.max(3,videoDims.w/100)} />
                <line x1={bbox.x} y1={bbox.y+bbox.h} x2={bbox.x} y2={bbox.y+bbox.h-20} stroke={borderColor} strokeWidth={Math.max(3,videoDims.w/100)} />
              </svg>
            )}

            <div style={{ position:'absolute', bottom:0, left:0, right:0, background:'rgba(26,35,126,0.75)', backdropFilter:'blur(6px)', color:'#fff', fontSize:'13px', fontWeight:'600', padding:'8px 14px', textAlign:'center', letterSpacing:'0.3px' }}>
              {camError ? <span style={{ color:'#ff8a80' }}>{camError}</span> : stateLabel}
            </div>

            {(detState === 'DWELLING' || detState === 'RECOGNIZING' || detState === 'ENROLLING') && (
              <div style={{ position:'absolute', top:0, left:0, right:0, bottom:0, border:`3px solid ${borderColor}`, borderRadius:'18px', animation:'scanPulse 1.4s ease-in-out infinite', pointerEvents:'none' }} />
            )}
          </div>

          {/* Capability card */}
          <div key={baseSlide} style={{ background:'#f8f9ff', border:'1.5px solid #e8eaf6', borderRadius:'16px', padding:'20px 36px', display:'flex', alignItems:'center', gap:'18px', boxShadow:'0 4px 20px rgba(26,35,126,0.08)', minWidth:'460px', animation:'slideIn 0.45s ease' }}>
            <div style={{ fontSize:'30px' }}>{cap.icon}</div>
            <div style={{ textAlign:'left' }}>
              <div style={{ fontSize:'15px', fontWeight:'700', color:'#1a237e' }}>{cap.title}</div>
              <div style={{ fontSize:'14px', color:'#777', marginTop:'3px', fontStyle:'italic' }}>Try: {cap.text}</div>
            </div>
          </div>

          {/* Dots */}
          <div style={{ display:'flex', gap:'8px' }}>
            {capabilities.map((_,i) => (
              <div key={i} style={{ width:i===baseSlide?'22px':'8px', height:'8px', borderRadius:'4px', background:i===baseSlide?'#1a237e':'#c5cae9', transition:'all 0.35s ease' }} />
            ))}
          </div>
        </div>

        <div style={{ background:'#1a237e', color:'#fff', padding:'12px 32px', display:'flex', justifyContent:'space-between', alignItems:'center', position:'relative', zIndex:1 }}>
          <span style={{ fontSize:'12px', opacity:0.85 }}>RNSIT Digital Receptionist System</span>
          <span style={{ fontSize:'12px', opacity:0.85 }}>Bengaluru · 560098</span>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════
          TOP LAYER — attract overlay with WHITE background
          Fades out automatically when a face is detected.
          ══════════════════════════════════════════════════════════ */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 20,
        background: '#ffffff',
        opacity: faceDetected ? 0 : 1,
        pointerEvents: faceDetected ? 'none' : 'auto',
        transition: 'opacity 0.65s cubic-bezier(0.4, 0, 0.2, 1)',
        display: 'flex', flexDirection: 'column',
        overflow: 'hidden', userSelect: 'none',
      }}>

        {/* CSS keyframes (scoped names to avoid clashing with base) */}
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
          @keyframes ovShimmer   { 0%{background-position:-200% center} 100%{background-position:200% center} }
          @keyframes ovSlideUp   { from{opacity:0;transform:translateY(22px)} to{opacity:1;transform:translateY(0)} }
          @keyframes ovFadeSlide { from{opacity:0;transform:translateX(16px)} to{opacity:1;transform:translateX(0)} }
          @keyframes ovRingPulse { 0%{transform:scale(0.9);opacity:0.5} 50%{transform:scale(1.12);opacity:0.15} 100%{transform:scale(0.9);opacity:0.5} }
          @keyframes ovTicker    { 0%{transform:translateX(0%)} 100%{transform:translateX(-50%)} }
          @keyframes ovBreathe   { 0%,100%{opacity:0.7} 50%{opacity:1} }
          @keyframes ovFloat     { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
          @keyframes drift       { 0%,100%{transform:translate(0,0)} 50%{transform:translate(46px,30px)} }
          @keyframes gentleFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-9px)} }
          @keyframes slideIn     { from{opacity:0;transform:translateX(26px)} to{opacity:1;transform:translateX(0)} }
          @keyframes scanPulse   { 0%,100%{opacity:0.9} 50%{opacity:0.3} }
          .ov-card    { animation: ovSlideUp 0.85s cubic-bezier(0.16,1,0.3,1) both; }
          .ov-slide   { animation: ovFadeSlide 0.5s cubic-bezier(0.16,1,0.3,1) both; }
          .ov-logo    { animation: ovFloat 5s ease-in-out infinite; }
          .ov-ring1   { animation: ovRingPulse 2.2s ease-in-out infinite; }
          .ov-ring2   { animation: ovRingPulse 2.2s ease-in-out infinite; animation-delay: 0.45s; }
          .ov-ticker  { display:inline-flex; gap:80px; animation:ovTicker 32s linear infinite; white-space:nowrap; }
        `}</style>

        {/* Light glow blobs */}
        <div style={{ position:'absolute', inset:0, overflow:'hidden', pointerEvents:'none' }}>
          <div style={{ position:'absolute', top:'-90px', left:'-90px', width:'440px', height:'440px', borderRadius:'50%', background:'radial-gradient(circle, rgba(26,35,126,0.06) 0%, transparent 65%)', filter:'blur(28px)' }} />
          <div style={{ position:'absolute', bottom:'-70px', right:'-70px', width:'380px', height:'380px', borderRadius:'50%', background:'radial-gradient(circle, rgba(27,94,32,0.05) 0%, transparent 65%)', filter:'blur(28px)' }} />
          {/* Grid */}
          <div style={{ position:'absolute', inset:0, backgroundImage:'linear-gradient(rgba(26,35,126,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(26,35,126,0.025) 1px, transparent 1px)', backgroundSize:'60px 60px' }} />
          {/* Orbs */}
          {orbs.map((o,i) => (
            <div key={i} style={{ position:'absolute', left:`${o.x}%`, top:`${o.y}%`, width:`${o.size}px`, height:`${o.size}px`, borderRadius:'50%', background:o.color, opacity:0.06, filter:'blur(3px)', transform:'translate(-50%,-50%)' }} />
          ))}
        </div>

        {/* Top bar */}
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'14px 32px 12px', borderBottom:'1px solid #e8eaf6', position:'relative', zIndex:10 }}>
          <div style={{ display:'inline-flex', alignItems:'center', gap:'8px', background:'#eff3ff', border:'1px solid #c5cae9', borderRadius:'999px', padding:'5px 14px' }}>
            <div className="ov-ring1" style={{ width:'7px', height:'7px', borderRadius:'50%', background:'#43a047', boxShadow:'0 0 0 3px rgba(67,160,71,0.2)' }} />
            <span style={{ fontSize:'11.5px', fontWeight:600, color:'#1a237e', letterSpacing:'0.5px' }}>RNSIT Digital Receptionist</span>
          </div>
          <div style={{ textAlign:'right' }}>
            <div style={{ fontSize:'26px', fontWeight:800, color:'#1a237e', letterSpacing:'-0.5px', lineHeight:1 }}>{timeStr}</div>
            <div style={{ fontSize:'11px', color:'#9e9e9e', marginTop:'2px' }}>{dateStr}</div>
          </div>
        </div>

        {/* Main content */}
        <div style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', padding:'24px 40px 16px', position:'relative', zIndex:5, gap:'26px' }}>

          {/* Hero card */}
          <div className="ov-card" style={{ background:'#ffffff', border:'1.5px solid #e8eaf6', borderRadius:'24px', padding:'30px 44px', minWidth:'540px', maxWidth:'680px', boxShadow:'0 8px 40px rgba(26,35,126,0.09), 0 2px 8px rgba(0,0,0,0.04)', display:'flex', flexDirection:'column', alignItems:'center', gap:'20px', opacity:mounted?1:0, transition:'opacity 0.5s ease' }}>

            {/* Logo + shimmer title */}
            <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:'10px' }}>
              <div className="ov-logo">
                <img src="/rnslogo.png" alt="RNSIT"
                  onError={e => { e.currentTarget.style.display='none'; }}
                  style={{ height:'76px', objectFit:'contain', display:'block', filter:'drop-shadow(0 6px 16px rgba(26,35,126,0.14))' }} />
              </div>
              <div style={{ textAlign:'center' }}>
                <div style={{ fontSize:'26px', fontWeight:900, lineHeight:1.2, background:'linear-gradient(90deg,#1a237e,#1565c0,#283593,#1a237e)', backgroundSize:'300% 100%', WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent', animation:'ovShimmer 4s linear infinite' }}>
                  RNS Institute of Technology
                </div>
                <div style={{ fontSize:'11px', color:'#9e9e9e', letterSpacing:'3px', textTransform:'uppercase', marginTop:'4px' }}>
                  Autonomous Institution · Bengaluru
                </div>
              </div>
            </div>

            {/* Rotating highlight */}
            <div className="ov-slide" key={ovSlide} style={{ width:'100%', background:h.bg, border:`1.5px solid ${h.accent}22`, borderRadius:'14px', padding:'16px 20px', display:'flex', alignItems:'center', gap:'14px' }}>
              <div style={{ fontSize:'32px', lineHeight:1 }}>{h.icon}</div>
              <div>
                <div style={{ fontSize:'16px', fontWeight:800, color:h.accent, marginBottom:'2px' }}>{h.headline}</div>
                <div style={{ fontSize:'12.5px', color:'#616161', fontWeight:500 }}>{h.sub}</div>
              </div>
              <div style={{ marginLeft:'auto', width:'8px', height:'8px', borderRadius:'50%', background:h.accent, boxShadow:`0 0 10px ${h.accent}66`, flexShrink:0 }} />
            </div>

            {/* Dots */}
            <div style={{ display:'flex', gap:'6px', alignItems:'center' }}>
              {highlights.map((hl,i) => (
                <div key={i} style={{ width:i===ovDot?'20px':'6px', height:'6px', borderRadius:'3px', background:i===ovDot?hl.accent:'#e0e0e0', transition:'all 0.4s ease' }} />
              ))}
            </div>
          </div>

          {/* Approach CTA */}
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:'12px', opacity:mounted?1:0, transition:'opacity 0.8s ease 0.3s' }}>
            <div style={{ position:'relative', display:'flex', alignItems:'center', justifyContent:'center' }}>
              <div className="ov-ring1" style={{ position:'absolute', width:'76px', height:'76px', borderRadius:'50%', border:'2px solid rgba(26,35,126,0.18)' }} />
              <div className="ov-ring2" style={{ position:'absolute', width:'56px', height:'56px', borderRadius:'50%', border:'2px solid rgba(26,35,126,0.1)' }} />
              <div style={{ width:'42px', height:'42px', borderRadius:'50%', background:'linear-gradient(135deg, #1a237e, #1565c0)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'19px', boxShadow:'0 4px 16px rgba(26,35,126,0.28)' }}>
                🧑
              </div>
            </div>
            <div style={{ textAlign:'center' }}>
              <div style={{ fontSize:'19px', fontWeight:800, color:'#1a237e', letterSpacing:'0.2px', animation:'ovBreathe 3s ease-in-out infinite' }}>
                Walk up to begin
              </div>
              <div style={{ fontSize:'12.5px', color:'#9e9e9e', marginTop:'3px', fontWeight:500 }}>
                Your AI campus guide is ready
              </div>
            </div>
          </div>
        </div>

        {/* Scrolling ticker */}
        <div style={{ background:'#f5f5f5', borderTop:'1px solid #e0e0e0', padding:'9px 0', overflow:'hidden', position:'relative', zIndex:10 }}>
          <div className="ov-ticker">
            {[...tickers, ...tickers].map((t,i) => (
              <span key={i} style={{ fontSize:'12px', fontWeight:600, color:'#757575', letterSpacing:'0.3px' }}>{t}</span>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'9px 32px', background:'#1a237e', position:'relative', zIndex:10 }}>
          <span style={{ fontSize:'11px', color:'rgba(255,255,255,0.75)', fontWeight:500 }}>Nova · AI Campus Receptionist</span>
          <span style={{ fontSize:'11px', color:'rgba(255,255,255,0.75)', fontWeight:500 }}>Dr. Vishnuvardhan Road, Channasandra · Bengaluru 560 098</span>
        </div>
      </div>
    </div>
  );
}