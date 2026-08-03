import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createKioskMic, float32ToInt16 } from './kioskMic';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8001';

/* ────────────────────────────────────────────────────────────────────────
   ARIA — animated receptionist avatar. Purely additive: does not read or
   modify any state/logic from WelcomeScreen. Takes `status` and animates
   itself with plain refs + CSS (blink, breathe, head-tilt, talk-cycle
   mouth movement while status === 'speaking'). Mouth animation uses a
   simple interval, not real audio amplitude, specifically so the existing
   speak()/TTS pipeline below never had to change.
   ──────────────────────────────────────────────────────────────────────── */
function AriaAvatar({ status }) {
  const mouthPathRef = useRef(null);
  const eyeLeftRef = useRef(null);
  const eyeRightRef = useRef(null);
  const irisLeftRef = useRef(null);
  const irisRightRef = useRef(null);
  const headGroupRef = useRef(null);
  const talkTimerRef = useRef(null);

  // idle blink loop — runs regardless of status
  useEffect(() => {
    let blinkTimer;
    const scheduleBlink = () => {
      const delay = 2600 + Math.random() * 3200;
      blinkTimer = setTimeout(() => {
        [eyeLeftRef.current, eyeRightRef.current].forEach(el => {
          if (!el) return;
          el.style.transform = 'scaleY(0.08)';
          setTimeout(() => { if (el) el.style.transform = 'scaleY(1)'; }, 120);
        });
        scheduleBlink();
      }, delay);
    };
    scheduleBlink();
    return () => clearTimeout(blinkTimer);
  }, []);

  // head tilt per status
  useEffect(() => {
    if (!headGroupRef.current) return;
    if (status === 'listening') {
      headGroupRef.current.style.transform = 'rotate(-3deg) translateY(-2px)';
    } else if (status === 'processing') {
      headGroupRef.current.style.transform = 'rotate(5deg) translateY(-1px)';
    } else if (status === 'speaking') {
      headGroupRef.current.style.transform = 'rotate(0deg)';
    } else {
      headGroupRef.current.style.transform = 'rotate(-1deg)';
    }
  }, [status]);

  // gaze shift — while 'processing', her eyes drift up and to the side like
  // she's recalling something, instead of just staring straight ahead. This
  // is what actually reads as "thinking" rather than the head tilt alone.
  useEffect(() => {
    [irisLeftRef.current, irisRightRef.current].forEach(el => {
      if (!el) return;
      el.style.transform = status === 'processing' ? 'translate(2px, -5px)' : 'translate(0px, 0px)';
    });
  }, [status]);

  // simple talk-cycle mouth animation while speaking — not tied to real
  // audio amplitude on purpose, so the existing TTS/speak() code below
  // never needed to change.
  useEffect(() => {
    if (status !== 'speaking') {
      if (talkTimerRef.current) clearInterval(talkTimerRef.current);
      if (mouthPathRef.current) mouthPathRef.current.setAttribute('d', 'M112 168 Q130 172 148 168');
      return;
    }
    talkTimerRef.current = setInterval(() => {
      if (!mouthPathRef.current) return;
      const openness = 6 + Math.random() * 18;
      mouthPathRef.current.setAttribute('d', `M112 168 Q130 ${168 + openness} 148 168`);
    }, 130);
    return () => clearInterval(talkTimerRef.current);
  }, [status]);

  return (
    <div className={`aria-figure aria-${status}`} style={{ width: '280px', height: '400px', position: 'relative' }}>
      <svg viewBox="0 0 260 400" width="280" height="400" style={{ overflow: 'visible' }}>
        <defs>
          <radialGradient id="ariaSkin" cx="42%" cy="32%" r="75%">
            <stop offset="0%" stopColor="#ffdcb8" />
            <stop offset="100%" stopColor="#f0bd8e" />
          </radialGradient>
          <linearGradient id="ariaBlazer" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2b3fa8" />
            <stop offset="100%" stopColor="#1a237e" />
          </linearGradient>
          <linearGradient id="ariaHair" x1="0.15" y1="0" x2="0.9" y2="1">
            <stop offset="0%" stopColor="#4a3527" />
            <stop offset="45%" stopColor="#2a1c14" />
            <stop offset="100%" stopColor="#180f0a" />
          </linearGradient>
          <radialGradient id="ariaBlush" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ff8f8f" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#ff8f8f" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* body / blazer, longer for a full-body figure */}
        <path d="M50 400 C50 300 70 245 130 245 C190 245 210 300 210 400 Z" fill="url(#ariaBlazer)" />
        {/* left arm, resting at side */}
        <path d="M76 262 C52 278 40 320 42 375 L62 375 C62 330 72 292 92 270 Z" fill="url(#ariaBlazer)" />
        <circle cx="52" cy="375" r="11" fill="url(#ariaSkin)" />
        {/* right arm, resting at side */}
        <path d="M184 262 C208 278 220 320 218 375 L198 375 C198 330 188 292 168 270 Z" fill="url(#ariaBlazer)" />
        <circle cx="208" cy="375" r="11" fill="url(#ariaSkin)" />
        {/* collar */}
        <path d="M118 249 L130 271 L142 249 L130 261 Z" fill="#fdfaf3" />
        <path d="M104 251 C112 261 122 267 130 269 L118 249 Z" fill="#28399c" />
        <path d="M156 251 C148 261 138 267 130 269 L142 249 Z" fill="#28399c" />
        {/* blazer buttons */}
        <circle cx="130" cy="288" r="3" fill="#28399c" />
        <circle cx="130" cy="308" r="3" fill="#28399c" />
        <circle cx="130" cy="328" r="3" fill="#28399c" />

        {/* head + neck */}
        <g ref={headGroupRef} className="aria-head-group" style={{ transformOrigin: '130px 185px' }}>
          <rect x="116" y="218" width="28" height="36" fill="url(#ariaSkin)" />

          {/* low side ponytail, sits behind everything */}
          <path d="M182 150 C202 168 208 202 196 234 C190 248 182 250 180 238
                    C186 214 182 184 168 164 Z" fill="url(#ariaHair)" />
          <path d="M184 156 C198 172 202 198 194 222" stroke="#5a4030" strokeWidth="2"
                fill="none" strokeLinecap="round" opacity="0.5" />

          {/* face base */}
          <ellipse cx="130" cy="168" rx="64" ry="68" fill="url(#ariaSkin)" />

          {/* hair back / crown, center-parted and swept to the side */}
          <path d="M66 158 C60 100 92 54 130 54 C170 54 200 100 194 158
                    C193 130 182 108 158 100 C170 118 172 140 168 158
                    C150 128 140 108 130 100 C120 108 110 128 96 158
                    C92 140 90 118 100 100 C78 108 68 130 66 158 Z"
                fill="url(#ariaHair)" />

          {/* loose face-framing strands */}
          <path d="M68 150 C62 180 66 212 78 236" stroke="url(#ariaHair)" strokeWidth="9"
                fill="none" strokeLinecap="round" />
          <path d="M192 150 C196 176 190 202 180 220" stroke="url(#ariaHair)" strokeWidth="8"
                fill="none" strokeLinecap="round" />
          <path d="M72 156 C67 178 70 202 80 220" stroke="#5a4030" strokeWidth="1.6"
                fill="none" strokeLinecap="round" opacity="0.45" />

          {/* eyebrows — thicker, arched */}
          <path d="M92 142 Q108 128 128 138" stroke="#2a1c14" strokeWidth="5" fill="none" strokeLinecap="round" />
          <path d="M132 138 Q152 128 168 142" stroke="#2a1c14" strokeWidth="5" fill="none" strokeLinecap="round" />

          {/* blush */}
          <ellipse cx="88" cy="188" rx="16" ry="10" fill="url(#ariaBlush)" />
          <ellipse cx="172" cy="188" rx="16" ry="10" fill="url(#ariaBlush)" />

          {/* big expressive eyes — scaleY toggled for blinking */}
          <g ref={eyeLeftRef} className="aria-eye" style={{ transformOrigin: '108px 163px' }}>
            <ellipse cx="108" cy="163" rx="15" ry="16.5" fill="#fff" />
            <g ref={irisLeftRef} style={{ transition: 'transform 0.4s ease' }}>
              <circle cx="109" cy="165" r="10.5" fill="#4a2f1c" />
              <circle cx="106" cy="161" r="3.4" fill="#fff" />
              <circle cx="112" cy="169" r="1.6" fill="#fff" opacity="0.7" />
            </g>
            {/* upper lash */}
            <path d="M94 154 Q108 144 124 152" stroke="#1c130d" strokeWidth="3.4" fill="none" strokeLinecap="round" />
            <path d="M92 153 L86 148" stroke="#1c130d" strokeWidth="2" strokeLinecap="round" />
          </g>
          <g ref={eyeRightRef} className="aria-eye" style={{ transformOrigin: '152px 163px' }}>
            <ellipse cx="152" cy="163" rx="15" ry="16.5" fill="#fff" />
            <g ref={irisRightRef} style={{ transition: 'transform 0.4s ease' }}>
              <circle cx="151" cy="165" r="10.5" fill="#4a2f1c" />
              <circle cx="154" cy="161" r="3.4" fill="#fff" />
              <circle cx="148" cy="169" r="1.6" fill="#fff" opacity="0.7" />
            </g>
            {/* upper lash */}
            <path d="M136 152 Q152 144 166 154" stroke="#1c130d" strokeWidth="3.4" fill="none" strokeLinecap="round" />
            <path d="M168 153 L174 148" stroke="#1c130d" strokeWidth="2" strokeLinecap="round" />
          </g>

          {/* nose */}
          <path d="M128 170 Q125 184 131 187" stroke="#d69f74" strokeWidth="2.2" fill="none" strokeLinecap="round" />

          {/* mouth — animated during 'speaking' via the talk-cycle interval */}
          <path ref={mouthPathRef} d="M112 168 Q130 172 148 168" transform="translate(0, 40)" stroke="#b5555f"
                strokeWidth="3.5" fill="#d98189" strokeLinecap="round" />

          {/* pearl earrings */}
          <circle cx="65" cy="182" r="4" fill="#fdf6ea" stroke="#e0d3b8" strokeWidth="1" />
          <circle cx="195" cy="182" r="4" fill="#fdf6ea" stroke="#e0d3b8" strokeWidth="1" />
        </g>

        {/* thinking dots — only visible while status === 'processing', via
            the .aria-processing rule below. Small trail rising toward her
            temple, like a thought bubble forming. */}
        <g className="aria-think-dots">
          <circle className="aria-think-dot" cx="196" cy="96" r="4.5" fill="#7e57c2" />
          <circle className="aria-think-dot" cx="210" cy="78" r="6" fill="#7e57c2" />
          <circle className="aria-think-dot" cx="228" cy="58" r="7.5" fill="#7e57c2" />
        </g>
      </svg>
    </div>
  );
}

export default function WelcomeScreen({ session, messages, setMessages, askingName, camStream }) {
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const camVideoRef = useRef(null);
  const isMounted = useRef(true);
  const isSpeaking = useRef(false);
  const isListening = useRef(false);
  const analyserRef = useRef(null);
  const animFrameRef = useRef(null);
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const statusRef = useRef('ready');        // readable inside callbacks
  const streamRef = useRef(null);           // persistent mic stream
  const pendingUtteranceRef = useRef(null);
  const playCtxRef = useRef(null);              // Web Audio playback context
  const playCursorRef = useRef(0);               // schedule cursor for gapless clips
  const pendingSpeechRef = useRef(null);         // speech blocked by autoplay policy

  // Browsers create AudioContext 'suspended' until a user gesture.
  // Unlock on the first pointer/key event and replay anything pending.
  useEffect(() => {
    const unlock = async () => {
      try { await playCtxRef.current?.resume(); } catch (e) { }
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

  const [name, setName] = useState('');
  const [saveData, setSaveData] = useState(true);
  const [submitted, setSubmitted] = useState(false);
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
  const [liveText, setLiveText] = useState('');
  const [listening, setListening] = useState(false);
  const [status, setStatus] = useState('ready');

  const visitorName = session?.user_name || 'Guest';
  const isReturning = session?.is_returning || false;
  const visitCount = session?.visit_count || 1;

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

  // ── Backend event WebSocket — server-pushed session_end ─────────────────
  // Handles inactivity timeout and detection-triggered session ends so the
  // goodbye screen appears immediately without waiting for the poll heartbeat.
  useEffect(() => {
    const WS = BACKEND.replace(/^http/, 'ws');
    let ws;
    let dead = false;
    function connect() {
      if (dead) return;
      ws = new WebSocket(WS + '/ws');
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'session_end') {
            window.dispatchEvent(new Event('vrk-session-ended'));
          }
        } catch (_) { }
      };
      ws.onclose = () => { if (!dead) setTimeout(connect, 3000); };
    }
    connect();
    return () => { dead = true; ws?.close(); };
  }, []);

  // ── Camera sidebar ────────────────────────────────────────────────────────
  // The camera + detection WebSocket now live in App.js so presence
  // detection (departure timeout, face-swap) keeps running while this
  // screen is shown. This just displays the shared stream.
  useEffect(() => {
    if (camVideoRef.current && camStream) {
      camVideoRef.current.srcObject = camStream;
    }
  }, [camStream]);

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
      try { audioCtxRef.current.close(); } catch (e) { }
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

    // Mic already initialized — just resume the VAD (e.g. after TTS finished).
    // Only mark status 'ready' if TTS is not currently playing; when called from
    // inside finish() isSpeaking is already false and finish() itself sets 'ready'
    // first, so either way the state transition is correct.
    if (micRef.current) {
      if (!isSpeaking.current) {
        micRef.current.resume();
        setStatus('ready');
      }
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
      // If TTS is already playing (greeting started before mic was ready),
      // immediately park the VAD so it doesn't pick up TTS audio.  finish()
      // will call startListening() again once speaking is done, at which point
      // isSpeaking will be false and we take the resume path above.
      if (isSpeaking.current) {
        mic.pause();
      } else {
        setStatus('ready');
      }
    } catch (err) {
      console.error('[MIC] Error:', err);
      if (!isSpeaking.current) setStatus('ready');
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
      try { await fetch(BACKEND + '/session/end?session_id=' + sid, { method: 'POST' }); } catch (e) { }
      window.dispatchEvent(new Event('vrk-session-ended'));   // App switches NOW
      return;
    }

    // 35 s hard cap — prevents status getting stuck at 'processing' if the
    // LLM is slow or the network drops after the request was sent.
    const askController = new AbortController();
    const askTimeout = setTimeout(() => askController.abort(), 35000);
    try {
      const [, askRes] = await Promise.all([
        fetch(BACKEND + '/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sid, text, speaker: 'user' })
        }),
        fetch(BACKEND + '/ask?question=' + encodeURIComponent(text),
          { signal: askController.signal })
      ]);
      clearTimeout(askTimeout);
      const data = await askRes.json();
      const answer = data.answer || 'Sorry, I do not have that information. Please visit the Admin Block.';
      fetch(BACKEND + '/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, text: answer, speaker: 'kiosk' })
      });
      speak(answer, () => addMessage(answer, 'kiosk'));
    } catch (e) {
      clearTimeout(askTimeout);
      console.error('[sendToBackend]', e);
      const fallback = e.name === 'AbortError'
        ? "I'm sorry, that's taking longer than expected. Please try asking again."
        : 'Sorry, something went wrong. Please try again.';
      speak(fallback, () => addMessage(fallback, 'kiosk'));
    }
  }, [session, addMessage]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const speak = useCallback(async (text, onStart) => {
    window.speechSynthesis.cancel();
    micRef.current?.pause();          // don't let the kiosk hear itself
    isSpeaking.current = true;
    // NOTE: status intentionally stays whatever it currently is (usually
    // 'processing'/'ready') here — NOT 'speaking' yet. Setting it this early
    // made Aria's mouth start moving and the status pill/dot turn red during
    // the TTS fetch+decode delay, before any actual audio had played. It now
    // flips inside fireStart(), which only fires once real playback begins.

    const finish = () => {
      isSpeaking.current = false;
      setStatus('ready');
      if (isMounted.current) startListening();
    };

    const fireStart = () => {
      setStatus('speaking');   // audio is actually starting NOW
      if (onStart) { onStart(); onStart = null; }
    };

    // Fallback: robotic browser voice, only if backend TTS is unavailable
    const browserSpeak = () => {
      fireStart();
      const utter = new SpeechSynthesisUtterance(text);
      utter.lang = 'en-US';
      utter.rate = 1.0;
      utter.volume = 1;
      utter.onend = finish;
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
    if (pctx.state === 'suspended') { try { await pctx.resume(); } catch (e) { } }
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
    }).catch(() => { });

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
    } catch (e) { console.error(e); }
  };

  const statusLabel = {
    ready: 'Voice Ready',
    listening: 'Listening',
    processing: 'Thinking',
    speaking: 'Speaking'
  }[status] || 'Voice Ready';

  const statusColor = {
    ready: '#5c6bc0',
    listening: '#43a047',
    processing: '#7e57c2',
    speaking: '#ef5350'
  }[status] || '#5c6bc0';

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

      <header style={{ background: '#1a237e', padding: '14px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', boxShadow: '0 2px 10px rgba(26,35,126,0.15)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(255,255,255,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '20px' }}>❄</div>
          <div>
            <span style={{ fontSize: '17px', fontWeight: '800', color: '#fff', letterSpacing: '0.2px' }}>RNS Institute of Technology</span>
            <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.65)', marginLeft: '10px' }}>Digital Receptionist</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '18px' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '14px', fontWeight: '700', color: '#fff' }}>{visitorName}</div>
            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)' }}>
              {isReturning ? `Visit #${visitCount}` : 'New Visitor'}
            </div>
          </div>
        </div>
      </header>

      {askingName && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: '16px', padding: '40px', width: '440px', boxShadow: '0 12px 48px rgba(0,0,0,0.18)' }}>
            {submitted ? (
              <div style={{ textAlign: 'center' }}>
                <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: '#e8eaf6', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1a237e" strokeWidth="2.5"><polyline points="20 6 9 17 4 12" /></svg>
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
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1a237e" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></svg>
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

      <div style={{ flex: '1 1 0', display: 'flex', overflow: 'hidden' }}>

        {/* ── Aria panel — matches reference layout ── */}
        <div style={{
          width: '46%', flexShrink: 0, position: 'relative', overflow: 'hidden',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: '14px', padding: '32px 24px',
          background: 'linear-gradient(160deg, #dfe3f7 0%, #eef0fb 55%, #e6e9f9 100%)',
        }}>
          {/* small live-camera PiP, bottom-right corner */}
          <div style={{
            position: 'absolute', bottom: '20px', right: '20px', width: '60px', height: '60px',
            borderRadius: '50%', overflow: 'hidden', border: '2.5px solid #fff',
            boxShadow: '0 4px 14px rgba(26,35,126,0.25)', zIndex: 2,
          }}>
            <video ref={camVideoRef} autoPlay playsInline muted
              style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
          </div>

          <div className={`aria-wrap aria-wrap-${status}`}>
            <AriaAvatar status={status} />
          </div>

          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '24px', fontWeight: '800', color: '#1a237e' }}>Aria</div>
            <div style={{ fontSize: '13px', color: '#888', marginTop: '2px' }}>RNSIT Digital Receptionist</div>
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '7px 16px', borderRadius: '20px', background: '#fff',
            border: `1.5px solid ${statusColor}44`, boxShadow: '0 2px 10px rgba(26,35,126,0.08)',
          }}>
            <div style={{
              width: '8px', height: '8px', borderRadius: '50%', background: statusColor,
              transition: 'background 0.3s',
              animation: status === 'speaking' ? 'ariaPulse 0.9s infinite'
                : status === 'listening' ? 'ariaPulse 1.4s infinite' : 'none',
            }} />
            <span style={{ fontSize: '13px', fontWeight: '700', color: statusColor }}>{statusLabel}</span>
          </div>

          <div style={{
            fontSize: '13px', color: '#7280a3',
            background: 'rgba(255,255,255,0.7)', padding: '8px 16px', borderRadius: '10px',
            fontStyle: 'italic', textAlign: 'center', maxWidth: '300px',
          }}>
            {status === 'listening' ? 'Please speak your question clearly' : hints[hintIndex]}
          </div>
        </div>

        {/* ── Conversation panel ── */}
        <div style={{ flex: '1 1 0', display: 'flex', flexDirection: 'column', minWidth: 0, borderLeft: '1.5px solid #e8eaf6' }}>

          <div style={{ padding: '16px 28px', borderBottom: '1.5px solid #e8eaf6', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#43a047' }} />
              <span style={{ fontSize: '15px', fontWeight: '700', color: '#222' }}>Conversation</span>
            </div>
            <span style={{ fontSize: '12px', color: '#aaa' }}>
              {messages.length === 0 ? 'Just started' : `${messages.length} messages`}
            </span>
          </div>

          <div ref={scrollRef} style={{ flex: '1 1 0', overflowY: 'auto', minHeight: 0, padding: '28px 32px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {messages.length === 0 && !liveText ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '14px' }}>
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#c5cae9" strokeWidth="1.6">
                  <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
                </svg>
                <div style={{ fontSize: '16px', fontWeight: '700', color: '#9aa0b4' }}>
                  Your conversation with Aria will appear here
                </div>
                <div style={{ fontSize: '13px', color: '#c2c7db' }}>
                  {status === 'listening' ? 'Listening...'
                    : status === 'processing' ? 'Thinking...'
                    : status === 'speaking' ? 'Speaking...'
                    : "Just speak — she's ready"}
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg, i) => (
                  <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: msg.speaker === 'kiosk' ? 'flex-start' : 'flex-end' }}>
                    <div style={{ fontSize: '11px', color: '#bbb', marginBottom: '4px', paddingLeft: msg.speaker === 'kiosk' ? '4px' : 0, paddingRight: msg.speaker !== 'kiosk' ? '4px' : 0, fontWeight: '500' }}>
                      {msg.speaker === 'kiosk' ? 'Aria' : visitorName} &nbsp;·&nbsp; {msg.timestamp}
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

          <div style={{ padding: '12px 28px', borderTop: '1.5px solid #e8eaf6', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <canvas ref={canvasRef} width={200} height={38}
              style={{ borderRadius: '10px', background: 'rgba(26,35,126,0.05)', display: listening ? 'block' : 'none' }} />
            {!listening && (
              <div style={{ width: '30px', height: '30px', borderRadius: '50%', background: '#f0f1f8', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={statusColor} strokeWidth="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                </svg>
              </div>
            )}
            <span style={{ fontSize: '13px', color: '#888', fontWeight: '600' }}>{statusLabel}</span>
          </div>
        </div>
      </div>{/* end flex row */}

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }
        @keyframes ariaPulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(1.6)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        @keyframes ring { 0%{transform:scale(0.65);opacity:1} 100%{transform:scale(1.5);opacity:0} }
        @keyframes breathe { 0%,100%{transform:scale(1)} 50%{transform:scale(1.06)} }
        @keyframes ariaBreathe { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
        @keyframes ariaSway { 0%,100%{transform:rotate(-1deg)} 50%{transform:rotate(1deg)} }
        .aria-wrap { transition: filter 0.4s ease; }
        .aria-wrap-ready { animation: ariaBreathe 4.2s ease-in-out infinite; }
        .aria-wrap-listening { animation: ariaBreathe 2.2s ease-in-out infinite; filter: drop-shadow(0 8px 20px rgba(67,160,71,0.3)); }
        .aria-wrap-processing { animation: ariaSway 2.4s ease-in-out infinite; filter: drop-shadow(0 8px 20px rgba(126,87,194,0.3)); }
        .aria-wrap-speaking { animation: ariaBreathe 1.6s ease-in-out infinite; filter: drop-shadow(0 8px 20px rgba(239,83,80,0.3)); }
        .aria-head-group, .aria-eye { transition: transform 0.35s ease; }
        .aria-think-dots { opacity: 0; transition: opacity 0.3s ease; }
        .aria-processing .aria-think-dots { opacity: 1; }
        .aria-think-dot { animation: ariaThinkDot 1.4s ease-in-out infinite; transform-origin: center; }
        .aria-think-dot:nth-child(1) { animation-delay: 0s; }
        .aria-think-dot:nth-child(2) { animation-delay: 0.2s; }
        .aria-think-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes ariaThinkDot { 0%,100% { opacity: 0.35; transform: scale(0.85); } 50% { opacity: 1; transform: scale(1.1); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input:focus { border-color: #1a237e !important; box-shadow: 0 0 0 3px rgba(26,35,126,0.1); }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #f5f6fa; }
        ::-webkit-scrollbar-thumb { background: #c5cae9; border-radius: 3px; }
      `}</style>
    </div>
  );
}