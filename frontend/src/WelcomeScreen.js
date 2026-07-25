import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createKioskMic, float32ToInt16 } from './kioskMic';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8001';

export default function WelcomeScreen({ session, messages, setMessages, askingName }) {
  const scrollRef      = useRef(null);
  const inputRef       = useRef(null);
  const isMounted      = useRef(true);
  const isSpeaking     = useRef(false);
  const isListening    = useRef(false);
  const analyserRef    = useRef(null);
  const animFrameRef   = useRef(null);
  const canvasRef      = useRef(null);
  const audioCtxRef    = useRef(null);
  const statusRef      = useRef('ready');        // readable inside callbacks
  const streamRef      = useRef(null);           // persistent mic stream
  const pendingUtteranceRef = useRef(null);
  const playCtxRef  = useRef(null);              // Web Audio playback context
  const playCursorRef = useRef(0);               // schedule cursor for gapless clips
  const pendingSpeechRef = useRef(null);         // speech blocked by autoplay policy

  // Browsers create AudioContext 'suspended' until a user gesture.
  // Unlock on the first pointer/key event and replay anything pending.
  useEffect(() => {
    const unlock = async () => {
      try { await playCtxRef.current?.resume(); } catch (e) {}
      if (pendingSpeechRef.current) {
        const { text, onStart } = pendingSpeechRef.current;
        pendingSpeechRef.current = null;
        speak(text, onStart);
      }
    };
    window.addEventListener('pointerdown', unlock);
    window.addEventListener('keydown', unlock);
    return () => {
      window.removeEventListener('pointerdown', unlock);
      window.removeEventListener('keydown', unlock);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);      // speech captured while busy

  const [name,       setName]       = useState('');
  const [saveData,   setSaveData]   = useState(true);
  const [submitted,  setSubmitted]  = useState(false);
  const [deleteMode, setDeleteMode] = useState(false);
  const [deleteName, setDeleteName] = useState('');
  const [deleted,    setDeleted]    = useState(false);
  const [hintIndex, setHintIndex] = useState(0);
  const hints = [
    'Try asking: "What courses does RNSIT offer?"',
    'Try asking: "How are the placements here?"',
    'Try asking: "Where is the admission office?"',
    'Try asking: "Tell me about campus facilities"',
    'Try asking: "What are the hostel options?"',
  ];
  useEffect(() => {
    const t = setInterval(() => setHintIndex(i => (i + 1) % 5), 6500);
    return () => clearInterval(t);
  }, []);
  const [liveText,   setLiveText]   = useState('');
  const [listening,  setListening]  = useState(false);
  const [status,     setStatus]     = useState('ready');

  const visitorName = session?.user_name    || 'Guest';
  const isReturning = session?.is_returning || false;
  const visitCount  = session?.visit_count  || 1;

  // The backend composes the greeting (it knows resume-vs-new and the
  // institute intro line); these local strings are only a fallback.
  const greeting = session?.greeting || (isReturning
    ? (visitCount > 2
        ? 'Welcome back, ' + visitorName + '! Great to see you again. How may I assist you today?'
        : 'Welcome back, ' + visitorName + '! How may I assist you today?')
    : 'Welcome, ' + visitorName + '! I am the digital receptionist of R N S Institute of Technology. '
      + 'I can help you with admissions, departments, placements, fees, and directions. '
      + 'How may I assist you today?');

  useEffect(() => { statusRef.current = status; }, [status]);

  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, liveText]);

  useEffect(() => {
    isMounted.current = true;
    return () => { isMounted.current = false; stopWaveform(); };
  }, []);

  useEffect(() => {
    if (askingName) {
      setSubmitted(false); setName(''); setSaveData(true);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [askingName]);

  const cleanText = (t) => (t || '').replace(/\u2014|\u2013/g, ', ').replace(/\s+,/g, ',');

  const addMessage = useCallback((text, speaker) => {
    text = cleanText(text);
    setMessages(prev => [...prev, {
      text, speaker,
      timestamp: new Date().toLocaleTimeString()
    }]);
  }, [setMessages]);

  // ── WAVEFORM ─────────────────────────────────────────────────────────────
  const stopWaveform = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (audioCtxRef.current) {
      try { audioCtxRef.current.close(); } catch(e) {}
      audioCtxRef.current = null;
    }
    analyserRef.current = null;
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  const startWaveform = useCallback((stream) => {
    if (audioCtxRef.current) return;             // already running — don't stack contexts
    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    analyserRef.current = analyser;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      const canvas = canvasRef.current;            // re-read every frame — canvas
      if (!canvas) return;                         // may mount after we start
      const ctx = canvas.getContext('2d');
      analyser.getByteFrequencyData(dataArray);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const barWidth = 3;
      const gap = 2;
      const bars = Math.floor(canvas.width / (barWidth + gap));
      const step = Math.floor(dataArray.length / bars);

      for (let i = 0; i < bars; i++) {
        const value = dataArray[i * step] / 255;
        const barHeight = Math.max(4, value * canvas.height * 0.9);
        const x = i * (barWidth + gap);
        const y = (canvas.height - barHeight) / 2;
        const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
        gradient.addColorStop(0, `rgba(100, 200, 255, ${0.4 + value * 0.6})`);
        gradient.addColorStop(1, `rgba(26, 35, 126, ${0.4 + value * 0.6})`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 2);
        ctx.fill();
      }
    };
    draw();
  }, []);

  // ── STT: browser VAD → Int16 PCM → POST /stt/pcm (GPU backend) ──────────
  const micRef = useRef(null);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const handleUtterance = useCallback(async (float32Audio) => {
    if (!isMounted.current || askingName) return;
    if (isSpeaking.current || statusRef.current === 'processing') {
      // Visitor spoke while we were busy — save it as the next prompt
      pendingUtteranceRef.current = float32Audio;
      return;
    }
    isListening.current = false;
    setListening(false);
    setStatus('processing');

    try {
      const i16 = float32ToInt16(float32Audio);   // halves bytes over the LAN
      const response = await fetch(BACKEND + '/stt/pcm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: i16.buffer
      });

      const result = await response.json();
      console.log('[STT WHISPER]', result);
      const heard = (result.text || '').trim();

      if (heard && heard.length > 1 && !isSpeaking.current) {
        if (isMounted.current) setLiveText(heard);
        sendToBackend(heard);
      } else {
        setStatus('ready');   // VAD keeps listening — no restart needed
      }
    } catch (err) {
      console.error('[STT] Error:', err);
      setStatus('ready');
    }
  }, [askingName]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const startListening = useCallback(async () => {
    if (!isMounted.current || askingName) return;

    // Mic already initialized — just resume the VAD (e.g. after TTS finished)
    if (micRef.current) {
      micRef.current.resume();
      setStatus('ready');
      return;
    }

    try {
      const mic = await createKioskMic({
        onStream: (stream) => { streamRef.current = stream; },
        onSpeechStart: () => {
          if (isSpeaking.current || !isMounted.current) return;
          isListening.current = true;
          setListening(true);
          setLiveText('');
          setStatus('listening');
          if (streamRef.current) startWaveform(streamRef.current);  // canvas is visible now
        },
        onSpeechEnd: (audio) => handleUtterance(audio),
        onMisfire: () => {
          isListening.current = false;
          if (isMounted.current) { setListening(false); setStatus('ready'); }
        },
      });
      micRef.current = mic;
      setStatus('ready');
    } catch (err) {
      console.error('[MIC] Error:', err);
      setStatus('ready');
    }
  }, [askingName, startWaveform, handleUtterance]);

  // pause the mic while the name modal is open; the mount effect resumes it
  useEffect(() => {
    if (askingName) micRef.current?.pause();
  }, [askingName]);

  // release mic + VAD on unmount (session end)
  useEffect(() => () => {
    micRef.current?.destroy();
    micRef.current = null;
  }, []);

  // if the visitor spoke while we were processing/speaking, handle it now
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (status === 'ready' && pendingUtteranceRef.current && !askingName) {
      const queued = pendingUtteranceRef.current;
      pendingUtteranceRef.current = null;
      handleUtterance(queued);
    }
  }, [status, askingName]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const sendToBackend = useCallback(async (text) => {
    if (!text) return;
    setLiveText('');
    const sid = session?.session_id || 'guest';
    addMessage(text, 'user');

    const goodbyeWords = ['thank you', 'thanks', 'bye', 'goodbye', 'see you', 'ok bye', 'thank you so much'];
    if (goodbyeWords.some(w => text.toLowerCase().includes(w))) {
      const farewell = 'You are most welcome! Have a wonderful day. Goodbye!';
      micRef.current?.pause();
      speak(farewell);                       // WebAudio keeps playing across unmount
      try { await fetch(BACKEND + '/session/end?session_id=' + sid, { method: 'POST' }); } catch(e) {}
      window.dispatchEvent(new Event('vrk-session-ended'));   // App switches NOW
      return;
    }

    try {
      const [, askRes] = await Promise.all([
        fetch(BACKEND + '/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sid, text, speaker: 'user' })
        }),
        fetch(BACKEND + '/ask?question=' + encodeURIComponent(text))
      ]);
      const data   = await askRes.json();
      const answer = data.answer || 'Sorry, I do not have that information. Please visit the Admin Block.';
      fetch(BACKEND + '/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, text: answer, speaker: 'kiosk' })
      });
      speak(answer, () => addMessage(answer, 'kiosk'));
    } catch(e) {
      console.error('[sendToBackend]', e);
      isSpeaking.current = false;
      setStatus('ready');
      if (isMounted.current) startListening();
    }
  }, [session, addMessage]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const speak = useCallback(async (text, onStart) => {
    window.speechSynthesis.cancel();
    micRef.current?.pause();          // don't let the kiosk hear itself
    isSpeaking.current = true;
    setStatus('speaking');

    const finish = () => {
      isSpeaking.current = false;
      setStatus('ready');
      if (isMounted.current) startListening();
    };

    const fireStart = () => { if (onStart) { onStart(); onStart = null; } };

    // Fallback: robotic browser voice, only if backend TTS is unavailable
    const browserSpeak = () => {
      fireStart();
      const utter  = new SpeechSynthesisUtterance(text);
      utter.lang   = 'en-US';
      utter.rate   = 1.0;
      utter.volume = 1;
      utter.onend   = finish;
      utter.onerror = finish;
      window.speechSynthesis.speak(utter);
    };

    // Primary: Kokoro voice, sentence-by-sentence pipeline —
    // sentence N plays while sentence N+1 synthesizes, so first audio
    // arrives after ONE sentence instead of the whole reply.
    const fetchClip = (s) =>
      fetch(BACKEND + '/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: s })
      }).then(r => r.json()).then(d => d.audio || null).catch(() => null);

    // Web Audio: decode (~10ms) + schedule on a running cursor = gapless.
    if (!playCtxRef.current) {
      playCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    const pctx = playCtxRef.current;
    if (pctx.state === 'suspended') { try { await pctx.resume(); } catch (e) {} }
    if (pctx.state === 'suspended') {
      // Autoplay policy blocked us (no user gesture yet, e.g. the very
      // first greeting). speechSynthesis is exempt — never stay silent.
      console.warn('[TTS] AudioContext blocked by autoplay policy — using browser voice. ' +
                   'Launch the kiosk browser with --autoplay-policy=no-user-gesture-required (run.py does this).');
      browserSpeak();
      return;
    }
    playCursorRef.current = pctx.currentTime;

    const playClip = (b64) => new Promise(async (resolve) => {
      try {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const buf = await pctx.decodeAudioData(bytes.buffer);
        const node = pctx.createBufferSource();
        node.buffer = buf;
        node.connect(pctx.destination);
        node.onended = resolve;
        fireStart();                     // text appears the moment audio starts
        const at = Math.max(pctx.currentTime, playCursorRef.current);
        node.start(at);
        playCursorRef.current = at + buf.duration;
      } catch (e) {
        resolve();                       // any decode failure -> skip clip
      }
    });

    try {
      const raw = (text.match(/[^.!?]+[.!?]+["']?\s*|[^.!?]+$/g) || [text])
        .map(s => s.trim()).filter(Boolean);

      // Chunking for natural pacing:
      //  - first chunk stays SHORT (fast time-to-first-audio)
      //  - later sentences MERGE into ~2-sentence chunks so Kokoro speaks
      //    across full stops itself with human-length pauses, instead of
      //    one clip per sentence with a synthesis gap at every full stop
      const sentences = [];
      if (raw.length) {
        let first = raw[0];
        if (first.length > 60) {
          const cut = first.indexOf(',');
          if (cut > 15) {
            sentences.push(first.slice(0, cut + 1));
            first = first.slice(cut + 1).trim();
          }
        }
        if (first) sentences.push(first);
        let buf = '';
        for (let i = 1; i < raw.length; i++) {
          buf = buf ? buf + ' ' + raw[i] : raw[i];
          if (buf.length >= 90) { sentences.push(buf); buf = ''; }
        }
        if (buf) sentences.push(buf);
      }

      // A tiny opener ("Hello!", "Sure.") as its own clip creates an
      // audible seam right after it — merge it into the next chunk.
      if (sentences.length > 1 && sentences[0].length < 25) {
        sentences[1] = sentences[0] + ' ' + sentences[1];
        sentences.shift();
      }

      // Prefetch two chunks ahead — playback almost never waits on synthesis
      let anyPlayed = false;
      let p0 = fetchClip(sentences[0]);
      let p1 = sentences.length > 1 ? fetchClip(sentences[1]) : null;

      for (let i = 0; i < sentences.length; i++) {
        const b64 = await p0;
        p0 = p1;
        p1 = i + 2 < sentences.length ? fetchClip(sentences[i + 2]) : null;
        if (b64) { anyPlayed = true; await playClip(b64); }
      }

      if (!anyPlayed) { browserSpeak(); return; }
      finish();
    } catch (e) {
      console.error('[TTS] backend unavailable, using browser voice', e);
      browserSpeak();
    }
  }, [startListening]);

  useEffect(() => {
    if (askingName) return;
    const t = setTimeout(startListening, 500);
    return () => clearTimeout(t);
  }, [askingName, startListening]);

  // ── Kiosk opens the conversation ────────────────────────────────────────
  // When a visitor is detected (session starts) and the name flow is done,
  // the kiosk speaks the greeting first — the visitor never has to start.
  const greetedRef = useRef(null);
  const lastGreetRef = useRef({ text: '', ts: 0 });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (askingName) return;
    const sid = session?.session_id;
    // Day-2 resume reuses the SAME session_id, so key the guard on the
    // visit instant too — otherwise a resumed visitor is never greeted.
    const visitKey = sid ? sid + '|' + (session?.resumed_at || '') : null;
    if (!visitKey || greetedRef.current === visitKey) return;
    greetedRef.current = visitKey;

    // Even if the session id churns (detection re-firing), never repeat
    // the same greeting within 20s — kills the double "welcome back"
    const now = Date.now();
    if (lastGreetRef.current.text === greeting && now - lastGreetRef.current.ts < 20000) return;
    lastGreetRef.current = { text: greeting, ts: now };

    // Pre-warm the greeting audio: the backend synthesizes + caches it
    // during our beat, so speak() below plays it near-instantly.
    fetch(BACKEND + '/tts', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: greeting })
    }).catch(() => {});

    const t = setTimeout(() => {
      if (isSpeaking.current) return;     // something else already talking
      speak(greeting, () => addMessage(greeting, 'kiosk'));
    }, 250);
    return () => clearTimeout(t);
  }, [session?.session_id, session?.resumed_at, askingName]);

  const handleSubmitName = async (overrideName, overrideSave) => {
    const finalName = (overrideName ?? name).trim() || 'Guest';
    const finalSave = overrideSave ?? saveData;
    setSubmitted(true);
    try {
      await fetch(BACKEND + '/visitor/submit_name?name=' + encodeURIComponent(finalName) + '&save=' + finalSave, { method: 'POST' });
    } catch(e) { console.error(e); }
  };

  const handleDeleteData = async () => {
    const trimmed = deleteName.trim();
    if (!trimmed) return;
    try {
      await fetch(BACKEND + '/visitor/delete_my_data?name=' + encodeURIComponent(trimmed), { method: 'POST' });
      setDeleted(true);
      setTimeout(() => { setDeleteMode(false); setDeleted(false); setDeleteName(''); }, 3500);
    } catch(e) { console.error(e); }
  };

  const statusLabel = {
    ready:      'Ready',
    listening:  'Listening',
    processing: 'Thinking',
    speaking:   'Speaking'
  }[status] || 'Ready';

  const statusColor = {
    ready:      '#ffb300',
    listening:  '#43a047',
    processing: '#7e57c2',
    speaking:   '#ef5350'
  }[status] || '#ffb300';

  const inputStyle = {
    width: '100%', padding: '12px 16px', border: '1.5px solid #c5cae9',
    borderRadius: '8px', fontSize: '15px', boxSizing: 'border-box',
    outline: 'none', color: '#1a237e', background: '#f8f9ff', transition: 'border 0.2s'
  };
  const btnPrimary = {
    padding: '11px 24px', border: 'none', borderRadius: '8px',
    background: '#1a237e', color: '#fff', cursor: 'pointer',
    fontSize: '14px', fontWeight: '600', letterSpacing: '0.3px'
  };
  const btnSecondary = {
    padding: '11px 24px', border: '1.5px solid #c5cae9', borderRadius: '8px',
    background: '#fff', color: '#555', cursor: 'pointer', fontSize: '14px'
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f5f6fa', fontFamily: "'Segoe UI', Arial, sans-serif", display: 'flex', flexDirection: 'column' }}>

      <header style={{ background: '#ffffff', borderBottom: '2px solid #e8eaf6', padding: '14px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', boxShadow: '0 2px 10px rgba(26,35,126,0.07)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <img src="/rnslogo.png" onError={(e) => { e.currentTarget.style.display = 'none'; }} alt="RNSIT" style={{ height: '56px', objectFit: 'contain' }} />
          <div>
            <div style={{ fontSize: '18px', fontWeight: '800', color: '#1a237e', letterSpacing: '0.3px' }}>RNS Institute of Technology</div>
            <div style={{ fontSize: '12px', color: '#888', marginTop: '2px', letterSpacing: '0.5px' }}>Digital Receptionist · Interactive Kiosk</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px', justifyContent: 'flex-end' }}>
              <div style={{ width: '9px', height: '9px', borderRadius: '50%', background: statusColor, transition: 'background 0.3s' }} />
              <span style={{ fontSize: '13px', color: statusColor, fontWeight: '700', transition: 'color 0.3s', letterSpacing: '0.2px' }}>{statusLabel}</span>
            </div>
          </div>
          <button onClick={() => setDeleteMode(d => !d)} style={{ background: '#fff5f5', border: '1.5px solid #ef9a9a', color: '#c62828', borderRadius: '8px', padding: '9px 18px', fontSize: '13px', cursor: 'pointer', fontWeight: '600' }}>
            Delete My Data
          </button>
        </div>
      </header>

      {/* The greeting is SPOKEN, not printed. It also appears as the first
          chat bubble via speak(greeting, () => addMessage(...)), so showing
          it again in a banner was duplicate clutter. */}

      {deleteMode && (
        <div onClick={e => e.target === e.currentTarget && setDeleteMode(false)}
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: '16px', padding: '40px', width: '420px', boxShadow: '0 12px 48px rgba(0,0,0,0.18)' }}>
            {deleted ? (
              <div style={{ textAlign: 'center' }}>
                <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: '#e8f5e9', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#43a047" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#1a237e' }}>Data Deleted Successfully</div>
                <p style={{ color: '#666', marginTop: '8px', fontSize: '14px' }}>Your face data has been permanently removed from the system.</p>
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                  <div style={{ width: '44px', height: '44px', borderRadius: '50%', background: '#ffebee', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#c62828" strokeWidth="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>
                  </div>
                  <div>
                    <div style={{ fontSize: '17px', fontWeight: '700', color: '#c62828' }}>Delete My Data</div>
                    <div style={{ fontSize: '12px', color: '#999' }}>This action cannot be undone</div>
                  </div>
                </div>
                <p style={{ color: '#666', marginBottom: '16px', fontSize: '14px', lineHeight: '1.6' }}>Enter your registered name to permanently remove your face data from the system.</p>
                <input style={inputStyle} placeholder="Enter your registered name"
                  value={deleteName} onChange={e => setDeleteName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleDeleteData()} autoFocus />
                <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '20px' }}>
                  <button onClick={() => setDeleteMode(false)} style={btnSecondary}>Cancel</button>
                  <button onClick={handleDeleteData} style={{ ...btnPrimary, background: '#c62828' }}>Delete Permanently</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {askingName && !deleteMode && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: '16px', padding: '40px', width: '440px', boxShadow: '0 12px 48px rgba(0,0,0,0.18)' }}>
            {submitted ? (
              <div style={{ textAlign: 'center' }}>
                <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: '#e8eaf6', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1a237e" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#1a237e' }}>
                  {saveData ? 'Welcome, ' + (name || 'Guest') + '!' : 'Welcome, Guest!'}
                </div>
                <p style={{ color: '#666', marginTop: '10px', fontSize: '14px', lineHeight: '1.6' }}>
                  {saveData ? 'Your face has been registered. We will recognize you on your next visit.' : 'You are visiting as a guest. No data has been saved.'}
                </p>
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <div style={{ width: '44px', height: '44px', borderRadius: '50%', background: '#e8eaf6', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1a237e" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                  </div>
                  <div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: '#1a237e' }}>Hello! Welcome to RNSIT</div>
                    <div style={{ fontSize: '13px', color: '#888' }}>We do not recognize you yet</div>
                  </div>
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ fontSize: '13px', fontWeight: '600', color: '#444', display: 'block', marginBottom: '6px' }}>Your Full Name</label>
                  <input ref={inputRef} style={inputStyle} placeholder="e.g. Akshatha A"
                    value={name} onChange={e => setName(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSubmitName()} autoFocus />
                </div>
                <div style={{ background: '#f8f9ff', border: '1.5px solid #e8eaf6', borderRadius: '10px', padding: '14px 16px', marginBottom: '20px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: '600', color: '#333' }}>Remember me for next visit</div>
                      <div style={{ fontSize: '12px', color: '#999', marginTop: '2px' }}>{saveData ? 'Your face will be saved securely' : 'No data will be stored'}</div>
                    </div>
                    <div onClick={() => setSaveData(s => !s)} style={{ width: '48px', height: '26px', borderRadius: '13px', background: saveData ? '#1a237e' : '#ddd', cursor: 'pointer', position: 'relative', transition: 'background 0.25s', flexShrink: 0 }}>
                      <div style={{ position: 'absolute', top: '3px', left: saveData ? '25px' : '3px', width: '20px', height: '20px', borderRadius: '50%', background: '#fff', transition: 'left 0.25s', boxShadow: '0 1px 4px rgba(0,0,0,0.2)' }} />
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button onClick={() => handleSubmitName('Guest', false)} style={{ ...btnSecondary, flex: 1 }}>Continue as Guest</button>
                  <button onClick={() => handleSubmitName()} style={{ ...btnPrimary, flex: 1 }}>{saveData ? 'Register & Continue' : 'Continue'}</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <div ref={scrollRef} style={{ flex: '1 1 0', overflowY: 'auto', minHeight: 0, padding: '28px 32px', display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '900px', width: '100%', margin: '0 auto', alignSelf: 'stretch' }}>
        {messages.length === 0 && !liveText ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px', paddingTop: '40px' }}>
            <div style={{ width: '340px', height: '150px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ position: 'relative', width: '130px', height: '130px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {/* expanding rings — the kiosk visibly "breathes" while waiting */}
                  <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid rgba(26,35,126,0.25)', animation: status === 'ready' ? 'ring 3.6s ease-out infinite' : 'none' }} />
                  <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid rgba(26,35,126,0.18)', animation: status === 'ready' ? 'ring 3.6s ease-out infinite 1.2s' : 'none' }} />
                  <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid rgba(26,35,126,0.10)', animation: status === 'ready' ? 'ring 3.6s ease-out infinite 2.4s' : 'none' }} />
                  <div style={{ width: '84px', height: '84px', borderRadius: '50%', background: 'linear-gradient(135deg, #1a237e, #3949ab)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 8px 28px rgba(26,35,126,0.35)', animation: status === 'ready' ? 'breathe 3.6s ease-in-out infinite' : 'none' }}>
                    <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="1.7">
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                      <line x1="12" y1="19" x2="12" y2="23"/>
                      <line x1="8" y1="23" x2="16" y2="23"/>
                    </svg>
                  </div>
                </div>
            </div>
            <div style={{ fontSize: '26px', fontWeight: '800', color: '#1a237e', letterSpacing: '0.2px', display: 'flex', alignItems: 'center', gap: '10px' }}>
              {status === 'listening'   && 'Listening'}
              {status === 'processing'  && <>Thinking<span className="dots"><i/><i/><i/></span></>}
              {status === 'speaking'    && 'Speaking'}
              {status === 'ready'       && 'How may I help you?'}
            </div>
            <div style={{ fontSize: '16px', color: '#9aa0b4', textAlign: 'center', maxWidth: '380px', lineHeight: '1.6' }}>
              {status === 'listening' ? 'Please speak your question clearly' : hints[hintIndex]}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: msg.speaker === 'kiosk' ? 'flex-start' : 'flex-end' }}>
                <div style={{ fontSize: '11px', color: '#bbb', marginBottom: '4px', paddingLeft: msg.speaker === 'kiosk' ? '4px' : 0, paddingRight: msg.speaker !== 'kiosk' ? '4px' : 0, fontWeight: '500' }}>
                  {msg.speaker === 'kiosk' ? 'RNSIT Kiosk' : visitorName} &nbsp;·&nbsp; {msg.timestamp}
                </div>
                <div style={{
                  animation: 'fadeUp 0.3s ease',
                  maxWidth: '70%', padding: '15px 19px',
                  borderRadius: msg.speaker === 'kiosk' ? '4px 18px 18px 18px' : '18px 4px 18px 18px',
                  background: msg.speaker === 'kiosk' ? '#ffffff' : '#1a237e',
                  color: msg.speaker === 'kiosk' ? '#222' : '#ffffff',
                  border: msg.speaker === 'kiosk' ? '1.5px solid #e8eaf6' : 'none',
                  fontSize: '17px', lineHeight: '1.6',
                  boxShadow: msg.speaker === 'kiosk' ? '0 2px 8px rgba(0,0,0,0.06)' : '0 2px 8px rgba(26,35,126,0.18)'
                }}>
                  {msg.text}
                </div>
              </div>
            ))}
            {liveText && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                <div style={{ fontSize: '11px', color: '#bbb', marginBottom: '4px', paddingRight: '4px' }}>{visitorName} (speaking...)</div>
                <div style={{ maxWidth: '60%', padding: '14px 18px', borderRadius: '18px 4px 18px 18px', background: '#e8eaf6', color: '#1a237e', fontSize: '16px', fontStyle: 'italic', lineHeight: '1.65', border: '1.5px solid #c5cae9' }}>
                  {liveText}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      <footer style={{ background: '#ffffff', borderTop: '1.5px solid #e8eaf6', padding: '10px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', boxShadow: '0 -2px 8px rgba(26,35,126,0.05)', minHeight: '64px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          {/* persistent voice dock: waveform while listening, breathing dot otherwise */}
          <canvas ref={canvasRef} width={240} height={44}
            style={{ borderRadius: '10px', background: 'rgba(26,35,126,0.05)', display: listening ? 'block' : 'none' }} />
          {!listening && (
            <div style={{ position: 'relative', width: '34px', height: '34px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {status === 'ready' && <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid rgba(26,35,126,0.25)', animation: 'ring 3.4s ease-out infinite' }} />}
              <div style={{ width: '16px', height: '16px', borderRadius: '50%', background: statusColor, transition: 'background 0.3s', animation: status === 'speaking' ? 'pulse 1.1s infinite' : status === 'ready' ? 'breathe 3.4s ease-in-out infinite' : 'none' }} />
            </div>
          )}
          <span style={{ fontSize: '14px', color: '#444', fontWeight: '600', transition: 'color 0.3s' }}>{statusLabel}</span>
        </div>
        <div style={{ fontSize: '12px', color: '#bbb' }}>RNSIT Digital Receptionist &nbsp;·&nbsp; Bengaluru 560098</div>
      </footer>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        @keyframes ring { 0%{transform:scale(0.65);opacity:1} 100%{transform:scale(1.5);opacity:0} }
        .dots { display:inline-flex; gap:5px; align-items:center; }
        .dots i { width:7px; height:7px; border-radius:50%; background:#7e57c2; animation: dotp 1.2s infinite ease-in-out; }
        .dots i:nth-child(2){ animation-delay:0.18s } .dots i:nth-child(3){ animation-delay:0.36s }
        @keyframes dotp { 0%,80%,100%{opacity:0.25; transform:scale(0.8)} 40%{opacity:1; transform:scale(1)} }
        @keyframes breathe { 0%,100%{transform:scale(1)} 50%{transform:scale(1.06)} }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input:focus { border-color: #1a237e !important; box-shadow: 0 0 0 3px rgba(26,35,126,0.1); }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #f5f6fa; }
        ::-webkit-scrollbar-thumb { background: #c5cae9; border-radius: 3px; }
      `}</style>
    </div>
  );
}