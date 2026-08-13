import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createKioskMic, float32ToInt16 } from './kioskMic';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8001';

export default function WelcomeScreen({ session, messages, setMessages, askingName }) {
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const camVideoRef = useRef(null);
  const camStreamRef = useRef(null);
  const isMounted = useRef(true);
  const isSpeaking = useRef(false);
  const awaitingAnswerRef = useRef(false);     // true from "ack started" until the real answer's speech starts/fails —
                                               // keeps status at 'processing' (not 'ready') through that gap
  const interruptSpeakingRef = useRef(null);   // lets the WS handler stop TTS
  const farewellPlayingRef = useRef(false);    // true while the goodbye line plays
  const greetingPlayingRef = useRef(false);    // true while the NEW-VISITOR greeting plays
                                               // (explicitly non-interruptible, per spec — it
                                               // is one short message that must always finish)
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
  const activeSpeakIdRef = useRef(null);         // identifies the current speak() call; used to cancel it on barge-in
  const activeNodesRef = useRef([]);             // currently scheduled/playing AudioBufferSourceNodes for the active speak()
  const ttsGainRef = useRef(null);               // shared gain node — lets us duck/restore TTS volume smoothly

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
  const [deleteMode, setDeleteMode] = useState(false);
  const [deleteName, setDeleteName] = useState('');
  const [deleted, setDeleted] = useState(false);
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
  const [processingHint, setProcessingHint] = useState('');   // transient "let me check that" indicator
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
    : 'Welcome to R N S Institute of Technology. I am Voix Nova, your digital receptionist. '
    + 'I can help you with admissions, departments, placements, fees, and directions around campus. '
    + 'How may I assist you today?');

  useEffect(() => { statusRef.current = status; }, [status]);
  const sessionRef = useRef(null);
  useEffect(() => { sessionRef.current = session; }, [session]);

  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, liveText, processingHint]);

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
            // Kill any audio + pending work from the ended session so nothing
            // bleeds into the next visitor — EXCEPT the farewell, which is the
            // one line meant to play as the session ends.
            if (!farewellPlayingRef.current) {
              try { interruptSpeakingRef.current && interruptSpeakingRef.current(); } catch (_) {}
              pendingUtteranceRef.current = null;
            }
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
  useEffect(() => {
    let active = true;
    navigator.mediaDevices?.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      .then(stream => {
        if (!active) { stream.getTracks().forEach(t => t.stop()); return; }
        camStreamRef.current = stream;
        if (camVideoRef.current) camVideoRef.current.srcObject = stream;
      })
      .catch(() => { }); // camera unavailable — sidebar just stays blank
    return () => {
      active = false;
      camStreamRef.current?.getTracks().forEach(t => t.stop());
      camStreamRef.current = null;
    };
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

  // Starts an EMPTY kiosk bubble and returns an appender that grows it one
  // sentence at a time (used by sendToBackend + speakStream's onSentence so
  // the bubble fills in exactly as fast as the voice speaks it).
  const startProgressiveMessage = useCallback((speaker) => {
    setMessages(prev => [...prev, {
      text: '', speaker,
      timestamp: new Date().toLocaleTimeString()
    }]);
    return (sentence) => {
      setMessages(prev => {
        if (!prev.length) return prev;
        const next = prev.slice();
        const last = next[next.length - 1];
        const sep = last.text ? ' ' : '';
        next[next.length - 1] = { ...last, text: cleanText(last.text + sep + sentence) };
        return next;
      });
    };
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
    if (greetingPlayingRef.current) {
      // The one-time first-visit greeting is non-interruptible by design —
      // drop anything spoken while it's still playing rather than cutting
      // it off. The visitor can speak again the instant it finishes.
      return;
    }
    if (isSpeaking.current) {
      // The VAD just CONFIRMED real speech (this is onSpeechEnd) while TTS
      // was still playing — commit the barge-in now and process it right away.
      interruptSpeaking();
    } else if (statusRef.current === 'processing') {
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

  // Barge-in has two stages, matching the two things the VAD can tell us:
  //
  //  1. onSpeechStart fires OPTIMISTICALLY — the instant sound crosses the
  //     probability threshold, before the VAD knows if it's real speech or
  //     a cough/click/echo blip. We respond by DUCKING the TTS volume —
  //     fast, but non-destructive and reversible.
  //  2. The VAD itself later tells us which it was:
  //       - onSpeechEnd   -> real speech, confirmed. NOW we hard-stop TTS.
  //       - onVADMisfire  -> false alarm. We restore TTS volume and carry on.
  //
  // This avoids killing the kiosk's sentence over noise/echo while still
  // reacting within ~80ms when someone genuinely starts talking.

  const duckSpeaking = useCallback(() => {
    if (!isSpeaking.current || !ttsGainRef.current || !playCtxRef.current) return;
    const g = ttsGainRef.current.gain;
    const now = playCtxRef.current.currentTime;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(0.12, now + 0.08);   // quick, gentle duck — not a hard cut
  }, []);

  const restoreSpeaking = useCallback(() => {
    if (!ttsGainRef.current || !playCtxRef.current) return;
    const g = ttsGainRef.current.gain;
    const now = playCtxRef.current.currentTime;
    g.cancelScheduledValues(now);
    g.setValueAtTime(g.value, now);
    g.linearRampToValueAtTime(1, now + 0.15);      // false alarm — fade back to full volume
  }, []);

  // Hard commit: only called once the VAD has CONFIRMED real speech
  // (onSpeechEnd), or on session-ending events (e.g. goodbye). Stops all
  // scheduled/playing TTS audio immediately and hands control back to the mic.
  const interruptSpeaking = useCallback(() => {
    if (!isSpeaking.current) return;
    activeSpeakIdRef.current = null;        // any in-flight speak() loop sees this and stops
    activeNodesRef.current.forEach((n) => { try { n.stop(); } catch (e) { /* already stopped */ } });
    activeNodesRef.current = [];
    window.speechSynthesis.cancel();        // in case the browser-voice fallback was speaking
    if (playCtxRef.current) playCursorRef.current = playCtxRef.current.currentTime;
    if (ttsGainRef.current && playCtxRef.current) {
      const now = playCtxRef.current.currentTime;
      ttsGainRef.current.gain.cancelScheduledValues(now);
      ttsGainRef.current.gain.setValueAtTime(1, now);   // reset for the next speak() call
    }
    isSpeaking.current = false;
  }, []);
  useEffect(() => { interruptSpeakingRef.current = interruptSpeaking; }, [interruptSpeaking]);

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
          if (!isMounted.current) return;
          // HARD barge-in: the instant the visitor starts speaking, STOP the
          // kiosk's voice immediately — don't just duck and wait for the VAD
          // to confirm at speech-end. A receptionist stops talking the moment
          // you speak; so does this. EXCEPTION: the first-visit greeting is
          // deliberately NOT interruptible — it's one short message and
          // every visitor should hear the whole thing once.
          if (isSpeaking.current && !greetingPlayingRef.current) interruptSpeaking();
          isListening.current = true;
          setListening(true);
          setLiveText('');
          setStatus('listening');
          if (streamRef.current) startWaveform(streamRef.current);  // canvas is visible now
        },
        onSpeechEnd: (audio) => handleUtterance(audio),
        onMisfire: () => {
          // We hard-stopped TTS on speech-start, so there's nothing to
          // restore. A misfire just means no real question followed — return
          // to ready and let the visitor speak again.
          isListening.current = false;
          if (isMounted.current) { setListening(false); setStatus('ready'); }
        },
      });
      micRef.current = mic;
      // Leave the VAD running even if TTS is already playing (e.g. the
      // greeting started before the mic finished initializing) — this lets
      // the visitor barge in on the very first greeting too. Status stays
      // whatever speak() already set ('speaking'); only set 'ready' when
      // nothing is currently talking.
      if (!isSpeaking.current) setStatus('ready');
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

  // speakStream(text, { onStart, onSentence, onDone }):
  //   - onStart fires once, the moment the FIRST clip's audio starts playing
  //     (kept for callers that just want a single "speech began" hook, e.g.
  //     the ack bubble / farewell / greeting).
  //   - onSentence(sentenceText, index) fires once PER CLIP, exactly when
  //     that clip's audio starts playing — this is what keeps the on-screen
  //     text in sync with what's actually being heard, instead of dumping
  //     the whole answer the moment the first sentence starts.
  //   - onDone fires once, when playback finishes (or is interrupted/falls
  //     back to the browser voice).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const speakStream = useCallback(async (text, { onStart, onSentence, onDone } = {}) => {
    // CRITICAL: stop any audio still playing from a PREVIOUS speak() call
    // (e.g. an acknowledgment like "let me check that" that hasn't finished
    // yet) before starting this one. Without this, two clips play at once —
    // this was the cause of garbled/overlapping speech after we added the
    // instant-acknowledgment feature.
    if (isSpeaking.current) interruptSpeaking();

    window.speechSynthesis.cancel();
    const myId = Symbol('speak');           // identifies this call so interruptSpeaking() can invalidate it
    activeSpeakIdRef.current = myId;
    // Mic is intentionally NOT paused here (unlike before) — it stays live
    // through TTS so the visitor can barge in. echoCancellation on the mic
    // stream (kioskMic.js) is what keeps it from hearing its own voice.
    isSpeaking.current = true;
    // NOTE: status is intentionally NOT set to 'speaking' here. Setting it
    // this early makes the avatar (and anything else keyed off `status`)
    // start its "speaking" animation before any audio has actually started
    // playing — e.g. right after the network answer arrives, while TTS is
    // still being fetched/synthesized. That reads as the avatar "speaking"
    // before the voice/text actually show up. It's set inside fireStart()
    // below instead, at the exact moment the first clip's audio begins.

    const finish = () => {
      if (activeSpeakIdRef.current !== myId) return;   // superseded/interrupted — do nothing
      isSpeaking.current = false;
      if (isMounted.current) startListening();   // resume mic for barge-in regardless
      // Only settle on 'ready' if there's genuinely nothing left to do. If
      // this was the instant-acknowledgment ("let me check that for you")
      // finishing before the real answer has arrived, go back to
      // 'processing' instead — otherwise the avatar sits idle/ready for a
      // few seconds while the kiosk is still actually working, which reads
      // as "did it hear me?" to the visitor. startListening() above may
      // have just set 'ready' synchronously; this runs right after and wins.
      setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
      if (onDone) { try { onDone(); } catch (e) {} onDone = null; }
    };

    const fireStart = () => {
      setStatus('speaking');    // avatar flips to "speaking" exactly when audio starts
      if (onStart) { onStart(); onStart = null; }
    };

    // Fallback: robotic browser voice, only if backend TTS is unavailable.
    // No per-sentence audio boundaries here, so the full text reveals at once
    // (still correct: it's the moment THIS voice actually starts talking).
    const browserSpeak = () => {
      fireStart();
      if (onSentence) { try { onSentence(text, 0); } catch (e) {} }
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
    if (!ttsGainRef.current) {
      ttsGainRef.current = pctx.createGain();
      ttsGainRef.current.connect(pctx.destination);
    }
    ttsGainRef.current.gain.cancelScheduledValues(pctx.currentTime);
    ttsGainRef.current.gain.setValueAtTime(1, pctx.currentTime);   // full volume for this new utterance

    // playClip resolves once the clip's audio has actually STARTED (not once
    // it finishes) — the caller loop awaits it just long enough to fire
    // onSentence in sync, then moves on to prefetch/schedule the next clip.
    // Playback itself is scheduled back-to-back on playCursorRef regardless,
    // so audio stays gapless even though we don't await full playback here.
    const playClip = (b64, sentenceText, sentenceIndex) => new Promise(async (resolveStarted) => {
      if (activeSpeakIdRef.current !== myId) return resolveStarted();   // interrupted before this clip started
      try {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const buf = await pctx.decodeAudioData(bytes.buffer);
        if (activeSpeakIdRef.current !== myId) return resolveStarted();  // interrupted while decoding
        const node = pctx.createBufferSource();
        node.buffer = buf;
        node.connect(ttsGainRef.current);
        activeNodesRef.current.push(node);
        node.onended = () => {
          activeNodesRef.current = activeNodesRef.current.filter((n) => n !== node);
        };

        const at = Math.max(pctx.currentTime, playCursorRef.current);
        const delayMs = Math.max(0, (at - pctx.currentTime) * 1000);
        node.start(at);
        playCursorRef.current = at + buf.duration;

        // Fire onStart/onSentence exactly when THIS clip's audio begins —
        // if it's scheduled to start later than "now" (queued behind an
        // earlier clip that's still playing), wait for that moment instead
        // of firing immediately, so text and voice stay in lockstep.
        const announce = () => {
          fireStart();                                     // status + first-clip-only hook
          if (onSentence) { try { onSentence(sentenceText, sentenceIndex); } catch (e) {} }
          resolveStarted();
        };
        if (delayMs > 0) setTimeout(announce, delayMs);
        else announce();
      } catch (e) {
        resolveStarted();                       // any decode failure -> skip clip
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

      // Prefetch two chunks ahead — playback almost never waits on synthesis.
      // Each playClip() resolves as soon as ITS audio starts (see above), so
      // this loop moves to fetching/queuing the next chunk immediately, while
      // the actual audio for every chunk still plays back-to-back via the
      // shared playCursorRef — sound stays gapless, text reveal stays synced.
      let anyPlayed = false;
      let lastClipPromise = Promise.resolve();
      let p0 = fetchClip(sentences[0]);
      let p1 = sentences.length > 1 ? fetchClip(sentences[1]) : null;

      for (let i = 0; i < sentences.length; i++) {
        if (activeSpeakIdRef.current !== myId) break;   // interrupted — stop scheduling more chunks
        const b64 = await p0;
        p0 = p1;
        p1 = i + 2 < sentences.length ? fetchClip(sentences[i + 2]) : null;
        if (b64) {
          anyPlayed = true;
          lastClipPromise = playClip(b64, sentences[i], i);
          await lastClipPromise;
        }
      }

      if (activeSpeakIdRef.current !== myId) return;    // interrupted — don't fall back to browser voice
      if (!anyPlayed) { browserSpeak(); return; }

      // Wait for the actual audio (not just the "started" signal) of the
      // final scheduled clip before calling finish() — otherwise finish()
      // (and startListening()) can fire while the last sentence is still
      // being heard.
      const lastEnd = playCursorRef.current;
      const remainingMs = Math.max(0, (lastEnd - pctx.currentTime) * 1000);
      await lastClipPromise;
      if (remainingMs > 0) await new Promise(r => setTimeout(r, remainingMs));
      if (activeSpeakIdRef.current !== myId) return;
      finish();
    } catch (e) {
      if (activeSpeakIdRef.current !== myId) return;
      console.error('[TTS] backend unavailable, using browser voice', e);
      browserSpeak();
    }
  }, [startListening]);

  // Thin wrapper over speakStream for callers that don't need per-sentence
  // sync (ack bubble, farewell, greeting, error fallback) — same (onStart,
  // onDone) signature as before.
  const speak = useCallback((text, onStart, onDone) => (
    speakStream(text, { onStart, onDone })
  ), [speakStream]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const sendToBackend = useCallback(async (text) => {
    if (!text) return;
    setLiveText('');
    setProcessingHint('');
    const sid = session?.session_id || 'guest';
    addMessage(text, 'user');

    const goodbyeWords = ['thank you', 'thanks', 'bye', 'goodbye', 'see you', 'ok bye', 'thank you so much'];
    if (goodbyeWords.some(w => text.toLowerCase().includes(w))) {
      const farewells = [
        'You are most welcome! Have a wonderful day. Goodbye!',
        'Happy to help! Take care and have a great day.',
        'Anytime! Wishing you a lovely day ahead. Goodbye!',
        'My pleasure! All the best, and see you around campus.',
      ];
      const farewell = farewells[Math.floor(Math.random() * farewells.length)];
      micRef.current?.pause();
      farewellPlayingRef.current = true;     // protect this audio from the session_end stop

      // Switch to the goodbye screen NOW so the farewell voice plays OVER it
      // (they should appear together). The audio uses Web Audio, which keeps
      // playing across this component unmounting — and farewellPlayingRef
      // keeps the session_end handler from stopping it. We clear the flag
      // when the voice actually finishes.
      speak(farewell, null, () => { farewellPlayingRef.current = false; });
      fetch(BACKEND + '/session/end?session_id=' + sid, { method: 'POST' }).catch(() => {});
      window.dispatchEvent(new Event('vrk-session-ended'));   // goodbye screen appears now
      return;
    }

    // Set BEFORE the ack fires: from this point until the real answer's
    // speech actually starts (or the request fails/is dropped), finish()
    // in speakStream will treat any in-between "ready" moment (e.g. the ack
    // finishing early) as still 'processing' — see awaitingAnswerRef above.
    awaitingAnswerRef.current = true;

    // INSTANT ACKNOWLEDGMENT: a real receptionist reacts the moment you
    // finish speaking — not after a silent pause. We play a short filler
    // right away while the actual answer is still being fetched, so there's
    // never dead air with a spinner. Kept short so it doesn't collide with
    // the real answer. Skipped for very short/greeting-like inputs.
    const acks = [
      'Sure, let me check that for you.',
      'Good question — one moment.',
      'Let me look that up for you.',
      'Of course, just a second.',
      'Right, let me find that.',
    ];
    if (text.split(' ').length >= 3) {
      const ack = acks[Math.floor(Math.random() * acks.length)];
      // Show it as a transient indicator bubble too — otherwise the visitor
      // sees nothing at all while the real answer is being fetched, which
      // is exactly the "left confused, feels frozen" problem.
      speak(ack, () => setProcessingHint(ack));
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

      // STALE-ANSWER GUARD: if the session changed while this request was in
      // flight (visitor said goodbye and left, next visitor arrived), this
      // answer belongs to nobody on screen — drop it so it never bleeds into
      // the next person's session.
      const liveSid = sessionRef.current?.session_id || 'guest';
      if (data.dropped || liveSid !== sid) {
        console.info('[sendToBackend] dropped stale answer for', sid);
        awaitingAnswerRef.current = false;
        isSpeaking.current = false;
        setStatus('ready');
        return;
      }

      const answer = data.answer || 'Sorry, I do not have that information. Please visit the Admin Block.';
      fetch(BACKEND + '/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, text: answer, speaker: 'kiosk' })
      });

      // Text and voice move together: an empty kiosk bubble opens the
      // instant speech begins, then grows one sentence at a time — each
      // sentence appears exactly when its audio starts playing, never
      // before. No more "reveal the whole answer as soon as it's fetched".
      let appendSentence = null;
      speakStream(answer, {
        onStart: () => {
          awaitingAnswerRef.current = false;   // real answer is speaking now — resting state is 'ready' again
          setProcessingHint('');
          appendSentence = startProgressiveMessage('kiosk');
        },
        onSentence: (sentence) => { if (appendSentence) appendSentence(sentence); },
      });
    } catch (e) {
      clearTimeout(askTimeout);
      awaitingAnswerRef.current = false;
      setProcessingHint('');   // never leave the "thinking" bubble stuck on a failed request
      console.error('[sendToBackend]', e);
      const fallback = e.name === 'AbortError'
        ? "I'm sorry, that's taking longer than expected. Please try asking again."
        : 'Sorry, something went wrong. Please try again.';
      speak(fallback, () => addMessage(fallback, 'kiosk'));
    }
  }, [session, addMessage, speakStream, startProgressiveMessage]);

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

    // Fire the greeting as SOON as the session/window is active — no
    // artificial delay. The first-visit line is pre-cached server-side at
    // startup (see tts.py _PREWARM), so this plays near-instantly without
    // needing a client-side pre-warm round-trip first.
    if (isSpeaking.current) return;       // something else already talking

    if (!isReturning) {
      // NEW VISITOR: this greeting is explicitly non-interruptible and
      // fires exactly once (guarded by visitKey above).
      greetingPlayingRef.current = true;
      speak(greeting, () => addMessage(greeting, 'kiosk'), () => {
        greetingPlayingRef.current = false;
      });
    } else {
      // Returning-visitor greeting keeps normal barge-in behavior.
      speak(greeting, () => addMessage(greeting, 'kiosk'));
    }
  }, [session?.session_id, session?.resumed_at, askingName]);

  const handleSubmitName = async (overrideName, overrideSave) => {
    const finalName = (overrideName ?? name).trim() || 'Guest';
    const finalSave = overrideSave ?? saveData;
    setSubmitted(true);
    try {
      await fetch(BACKEND + '/visitor/submit_name?name=' + encodeURIComponent(finalName) + '&save=' + finalSave, { method: 'POST' });
    } catch (e) { console.error(e); }
  };

  const handleDeleteData = async () => {
    const trimmed = deleteName.trim();
    if (!trimmed) return;
    try {
      await fetch(BACKEND + '/visitor/delete_my_data?name=' + encodeURIComponent(trimmed), { method: 'POST' });
      setDeleted(true);
      setTimeout(() => { setDeleteMode(false); setDeleted(false); setDeleteName(''); }, 3500);
    } catch (e) { console.error(e); }
  };

  const inputStyle = {
    width: '100%', padding: '12px 16px', border: '1.5px solid #c5cae9',
    borderRadius: '8px', fontSize: '15px', boxSizing: 'border-box',
    outline: 'none', color: '#1a237e', background: '#f8f9ff', transition: 'border 0.2s'
  };
  const btnPrimary = { padding: '11px 24px', border: 'none', borderRadius: '8px', background: '#1a237e', color: '#fff', cursor: 'pointer', fontSize: '14px', fontWeight: '600' };
  const btnSecondary = { padding: '11px 24px', border: '1.5px solid #c5cae9', borderRadius: '8px', background: '#fff', color: '#555', cursor: 'pointer', fontSize: '14px' };

  /* ── ANIMATED ARIA CHARACTER ─────────────────────────────────────────── */
  const AriaCharacter = ({ st }) => (
    <svg className={`aria-svg aria-${st}`} viewBox="0 0 320 500"
      style={{ width: '100%', maxWidth: '340px', overflow: 'visible', display: 'block' }}>
      <defs>
        <linearGradient id="skinG" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FFCFA0" /><stop offset="100%" stopColor="#F0A06A" />
        </linearGradient>
        <linearGradient id="suitG" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#1e2e96" /><stop offset="100%" stopColor="#0d1860" />
        </linearGradient>
        <radialGradient id="shadowG" cx="50%" cy="50%">
          <stop offset="0%" stopColor="#0000001a" /><stop offset="100%" stopColor="#00000000" />
        </radialGradient>
      </defs>

      {/* ── floor shadow ── */}
      <ellipse cx="160" cy="498" rx="88" ry="11" fill="url(#shadowG)" />

      {/* ════════ BODY (breathing group) ════════ */}
      <g className="body-grp" style={{ transformOrigin: '160px 360px' }}>

        {/* suit */}
        <path d="M55 228 Q55 202 160 207 Q265 202 265 228 L270 460 Q160 474 50 460 Z" fill="url(#suitG)" />
        {/* shirt */}
        <path d="M130 207 L160 245 L190 207" fill="white" />
        {/* lapels */}
        <path d="M88 207 L130 207 L160 245 Q110 265 80 298 Z" fill="#152070" />
        <path d="M232 207 L190 207 L160 245 Q210 265 240 298 Z" fill="#152070" />
        {/* buttons */}
        <circle cx="160" cy="280" r="4.5" fill="#3a4ec8" />
        <circle cx="160" cy="308" r="4.5" fill="#3a4ec8" />
        <circle cx="160" cy="336" r="4.5" fill="#3a4ec8" />
        <line x1="160" y1="245" x2="160" y2="465" stroke="#0d1860" strokeWidth="1.5" />

        {/* ── LEFT ARM (stays normal in all states) ── */}
        <path d="M55 228 Q22 270 18 325 Q15 355 28 366"
          stroke="#1e2e96" strokeWidth="44" fill="none" strokeLinecap="round" />
        <ellipse cx="28" cy="372" rx="22" ry="15" fill="url(#skinG)" />

        {/* ── RIGHT ARM — normal (hidden during processing) ── */}
        {st !== 'processing' && <>
          <path d="M265 228 Q298 270 302 325 Q305 355 292 366"
            stroke="#1e2e96" strokeWidth="44" fill="none" strokeLinecap="round" />
          <ellipse cx="292" cy="372" rx="22" ry="15" fill="url(#skinG)" />
        </>}

        {/* ── RIGHT ARM — thinking pose ── */}
        {st === 'processing' && <>
          <path className="arm-think" d="M265 228 Q288 212 272 176 Q264 158 244 152"
            stroke="#1e2e96" strokeWidth="44" fill="none" strokeLinecap="round" />
          <ellipse className="hand-think" cx="242" cy="158" rx="24" ry="15" fill="url(#skinG)" />
        </>}
      </g>{/* end body-grp */}

      {/* ════════ HEAD (expression group) ════════ */}
      <g className="head-grp" style={{ transformOrigin: '160px 120px' }}>

        {/* neck */}
        <rect x="145" y="175" width="30" height="38" rx="10" fill="url(#skinG)" />

        {/* hair back */}
        <path d="M74 158 Q68 86 108 44 Q133 16 160 13 Q187 16 212 44 Q252 86 246 158" fill="#2B1A0C" />
        {/* head skin */}
        <circle cx="160" cy="105" r="80" fill="url(#skinG)" />
        {/* hair front */}
        <path d="M80 86 Q92 34 160 28 Q228 34 240 86 Q218 48 160 46 Q102 48 80 86" fill="#2B1A0C" />
        {/* hair sides */}
        <path d="M80 86 Q66 124 70 170" stroke="#2B1A0C" strokeWidth="15" fill="none" strokeLinecap="round" />
        <path d="M240 86 Q254 124 250 170" stroke="#2B1A0C" strokeWidth="15" fill="none" strokeLinecap="round" />

        {/* ── EYE AREA ── */}
        {/* whites */}
        <ellipse cx="131" cy="106" rx="16" ry="17" fill="white" opacity="0.97" />
        <ellipse cx="189" cy="106" rx="16" ry="17" fill="white" opacity="0.97" />
        {/* iris */}
        <circle className="iris-l" cx="133" cy="107" r="10" fill="#3A2010" />
        <circle className="iris-r" cx="191" cy="107" r="10" fill="#3A2010" />
        {/* pupil */}
        <circle className="pupil-l" cx="134" cy="108" r="5.5" fill="#0C0706" />
        <circle className="pupil-r" cx="192" cy="108" r="5.5" fill="#0C0706" />
        {/* shine */}
        <circle cx="136" cy="104" r="2.8" fill="white" />
        <circle cx="194" cy="104" r="2.8" fill="white" />
        {/* bottom lash line */}
        <path d="M115 118 Q131 124 147 118" stroke="#2B1A0C" strokeWidth="1.5" fill="none" />
        <path d="M173 118 Q189 124 205 118" stroke="#2B1A0C" strokeWidth="1.5" fill="none" />
        {/* BLINK eyelids — animated via SMIL */}
        <ellipse cx="131" cy="106" rx="16.5" ry="1" fill="url(#skinG)">
          <animate attributeName="ry" values="1;1;1;1;1;1;1;1;1;18;1;1;1" dur="4.2s" repeatCount="indefinite" />
        </ellipse>
        <ellipse cx="189" cy="106" rx="16.5" ry="1" fill="url(#skinG)">
          <animate attributeName="ry" values="1;1;1;1;1;1;1;1;1;18;1;1;1" dur="4.2s" begin="0.07s" repeatCount="indefinite" />
        </ellipse>

        {/* ── eyebrows ── */}
        <path className={`brow-l ${st === 'processing' ? 'brow-think' : ''}`}
          d="M 117 89 Q 131 82 145 89" stroke="#2B1A0C" strokeWidth="3.5" fill="none" strokeLinecap="round" />
        <path className={`brow-r ${st === 'processing' ? 'brow-think' : ''}`}
          d="M 175 89 Q 189 82 203 89" stroke="#2B1A0C" strokeWidth="3.5" fill="none" strokeLinecap="round" />

        {/* nose */}
        <path d="M157 120 Q152 132 154 136 Q159 140 165 136 Q168 132 163 120" fill="none" stroke="#D4906A" strokeWidth="1.5" />

        {/* ── MOUTH states ── */}
        {/* neutral smile */}
        {st !== 'speaking' &&
          <path d="M142 149 Q160 161 178 149" stroke="#B84055" strokeWidth="2.8" fill="none" strokeLinecap="round" />}
        {/* talking — alternates via CSS */}
        {st === 'speaking' && <>
          <g className="mouth-a">
            <path d="M143 149 Q160 163 177 149" fill="#B84055" stroke="#B84055" strokeWidth="2" strokeLinecap="round" />
            <ellipse cx="160" cy="156" rx="14" ry="8" fill="#7B2030" />
            <path d="M147 150 Q160 148 173 150" stroke="#FFBBC0" strokeWidth="1.5" fill="none" />
          </g>
          <g className="mouth-b">
            <path d="M142 148 Q160 166 178 148" fill="#B84055" stroke="#B84055" strokeWidth="2" strokeLinecap="round" />
            <ellipse cx="160" cy="158" rx="17" ry="11" fill="#7B2030" />
            <path d="M147 149 Q160 147 173 149" stroke="#FFBBC0" strokeWidth="1.5" fill="none" />
          </g>
        </>}

        {/* blush */}
        <ellipse cx="108" cy="124" rx="17" ry="12" fill="#F4A0B0" opacity="0.28" />
        <ellipse cx="212" cy="124" rx="17" ry="12" fill="#F4A0B0" opacity="0.28" />
        {/* earrings */}
        <circle cx="80" cy="113" r="5.5" fill="#FFD700" />
        <circle cx="240" cy="113" r="5.5" fill="#FFD700" />

        {/* ── SPEAKING sound waves (right of head) ── */}
        {st === 'speaking' && <>
          <path className="wave1" d="M250 92 Q264 105 250 118" stroke="#7c4dff" strokeWidth="3" fill="none" strokeLinecap="round" />
          <path className="wave2" d="M260 80 Q278 105 260 130" stroke="#7c4dff" strokeWidth="2.5" fill="none" strokeLinecap="round" />
          <path className="wave3" d="M270 68 Q292 105 270 142" stroke="#7c4dff" strokeWidth="2" fill="none" strokeLinecap="round" />
        </>}

        {/* ── LISTENING pulse ring ── */}
        {st === 'listening' && <>
          <circle cx="160" cy="105" r="92" fill="none" stroke="#43a047" strokeWidth="2.5" className="listen-r1" />
          <circle cx="160" cy="105" r="92" fill="none" stroke="#43a047" strokeWidth="1.5" className="listen-r2" />
        </>}

        {/* ── THINKING bubble ── */}
        {st === 'processing' && <>
          <circle className="tbub" cx="226" cy="70" r="6" fill="rgba(255,255,255,0.88)" />
          <circle className="tbub" cx="240" cy="54" r="10" fill="rgba(255,255,255,0.92)" />
          <circle className="tbub" cx="258" cy="36" r="15" fill="rgba(255,255,255,0.96)" />
          <text x="258" y="41" textAnchor="middle" fontSize="15" fill="#7e57c2" fontWeight="700">?</text>
        </>}

      </g>{/* end head-grp */}
    </svg>
  );

  /* background tint per state */
  const charBg = {
    ready: 'linear-gradient(175deg, #dde4ff 0%, #c8d4fc 100%)',
    listening: 'linear-gradient(175deg, #d8f5dc 0%, #b8eec0 100%)',
    processing: 'linear-gradient(175deg, #ede4ff 0%, #d8caff 100%)',
    speaking: 'linear-gradient(175deg, #fff0dd 0%, #ffd8a8 100%)',
  }[status] || 'linear-gradient(175deg, #dde4ff 0%, #c8d4fc 100%)';

  const statusLabel = { ready: 'Ready', listening: 'Listening…', processing: 'Thinking…', speaking: 'Speaking…' }[status] || 'Ready';
  const statusColor = { ready: '#1a237e', listening: '#2e7d32', processing: '#6a1b9a', speaking: '#bf360c' }[status] || '#1a237e';
  const statusBg = { ready: '#e8eaf6', listening: '#e8f5e9', processing: '#f3e5f5', speaking: '#fff3e0' }[status] || '#e8eaf6';

  return (
    <div style={{
      height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column',
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif", background: '#f0f4ff'
    }}>

      {/* ── MODALS ── */}
      {deleteMode && (
        <div onClick={e => e.target === e.currentTarget && setDeleteMode(false)}
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: '20px', padding: '40px', width: '420px', boxShadow: '0 24px 64px rgba(0,0,0,0.22)' }}>
            {deleted ? (
              <div style={{ textAlign: 'center' }}>
                <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: '#e8f5e9', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#43a047" strokeWidth="2.5"><polyline points="20 6 9 17 4 12" /></svg>
                </div>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#1a237e' }}>Data Deleted</div>
                <p style={{ color: '#666', marginTop: '8px', fontSize: '14px' }}>Your face data has been permanently removed.</p>
              </div>
            ) : (<>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                <div style={{ width: '44px', height: '44px', borderRadius: '50%', background: '#ffebee', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#c62828" strokeWidth="2"><polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14H6L5 6" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4h6v2" /></svg>
                </div>
                <div>
                  <div style={{ fontSize: '17px', fontWeight: '700', color: '#c62828' }}>Delete My Data</div>
                  <div style={{ fontSize: '12px', color: '#999' }}>This cannot be undone</div>
                </div>
              </div>
              <p style={{ color: '#666', marginBottom: '16px', fontSize: '14px', lineHeight: '1.6' }}>Enter your registered name to permanently remove your face data.</p>
              <input style={inputStyle} placeholder="Your registered name" value={deleteName} onChange={e => setDeleteName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleDeleteData()} autoFocus />
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '20px' }}>
                <button onClick={() => setDeleteMode(false)} style={btnSecondary}>Cancel</button>
                <button onClick={handleDeleteData} style={{ ...btnPrimary, background: '#c62828' }}>Delete Permanently</button>
              </div>
            </>)}
          </div>
        </div>
      )}

      {askingName && !deleteMode && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: '20px', padding: '40px', width: '440px', boxShadow: '0 24px 64px rgba(0,0,0,0.22)' }}>
            {submitted ? (
              <div style={{ textAlign: 'center' }}>
                <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: '#e8eaf6', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1a237e" strokeWidth="2.5"><polyline points="20 6 9 17 4 12" /></svg>
                </div>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#1a237e' }}>
                  {saveData ? 'Welcome, ' + (name || 'Guest') + '!' : 'Welcome, Guest!'}
                </div>
                <p style={{ color: '#666', marginTop: '10px', fontSize: '14px', lineHeight: '1.6' }}>
                  {saveData ? "Your face is registered. We'll recognise you next time." : 'Visiting as a guest — no data saved.'}
                </p>
              </div>
            ) : (<>
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '20px' }}>
                <svg viewBox="0 0 80 80" width="56" height="56" style={{ flexShrink: 0 }}>
                  <circle cx="40" cy="40" r="40" fill="url(#suitG2)" />
                  <defs><linearGradient id="suitG2" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#1e2e96" /><stop offset="100%" stopColor="#0d1860" /></linearGradient></defs>
                  <circle cx="40" cy="30" r="14" fill="#FFCFA0" />
                  <path d="M12 72 Q12 54 40 54 Q68 54 68 72" fill="#FFCFA0" />
                </svg>
                <div>
                  <div style={{ fontSize: '18px', fontWeight: '700', color: '#1a237e' }}>Hi! I&apos;m Aria 👋</div>
                  <div style={{ fontSize: '13px', color: '#888' }}>I don&apos;t recognise you yet — what&apos;s your name?</div>
                </div>
              </div>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ fontSize: '13px', fontWeight: '600', color: '#444', display: 'block', marginBottom: '6px' }}>Your Full Name</label>
                <input ref={inputRef} style={inputStyle} placeholder="e.g. Akshatha A" value={name} onChange={e => setName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSubmitName()} autoFocus />
              </div>
              <div style={{ background: '#f8f9ff', border: '1.5px solid #e8eaf6', borderRadius: '10px', padding: '14px 16px', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <div style={{ fontSize: '13px', fontWeight: '600', color: '#333' }}>Remember me for next visit</div>
                    <div style={{ fontSize: '12px', color: '#999', marginTop: '2px' }}>{saveData ? 'Face saved securely' : 'No data stored'}</div>
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
            </>)}
          </div>
        </div>
      )}

      {/* ── SLIM HEADER ── */}
      <header style={{
        background: 'linear-gradient(90deg,#1a237e,#283593)', padding: '9px 20px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        boxShadow: '0 2px 10px rgba(0,0,0,0.22)', flexShrink: 0, zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <img src="/rnslogo.png" onError={e => { e.currentTarget.style.display = 'none'; }} alt="RNSIT"
            style={{ height: '36px', objectFit: 'contain', filter: 'brightness(0) invert(1)', opacity: 0.9 }} />
          <div style={{ fontSize: '15px', fontWeight: '700', color: '#fff', letterSpacing: '0.2px' }}>
            RNS Institute of Technology
            <span style={{ fontSize: '11px', fontWeight: '400', color: 'rgba(255,255,255,0.5)', marginLeft: '8px' }}>Digital Receptionist</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '13px', color: '#fff', fontWeight: '700' }}>{visitorName}</div>
            <div style={{ fontSize: '11px', color: isReturning ? '#a5d6a7' : 'rgba(255,255,255,0.5)', fontWeight: '600' }}>
              {isReturning ? `🌟 Visit #${visitCount}` : 'New Visitor'}
            </div>
          </div>
          <button onClick={() => setDeleteMode(d => !d)}
            style={{ background: 'rgba(255,255,255,0.12)', border: '1px solid rgba(255,255,255,0.2)', color: 'rgba(255,255,255,0.8)', borderRadius: '7px', padding: '6px 12px', fontSize: '12px', cursor: 'pointer', fontWeight: '600' }}>
            ⚙ Privacy
          </button>
        </div>
      </header>

      {/* ── MAIN BODY ── */}
      <div style={{ flex: '1 1 0', display: 'flex', overflow: 'hidden' }}>

        {/* ══════════ LEFT: ANIMATED ARIA CHARACTER ══════════ */}
        <div style={{
          width: '58%', flexShrink: 0, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'flex-end', padding: '0 24px 20px',
          background: charBg, transition: 'background 0.8s ease', position: 'relative',
          overflow: 'hidden'
        }}>

          {/* subtle radial glow behind character */}
          <div style={{
            position: 'absolute', bottom: '60px', left: '50%', transform: 'translateX(-50%)',
            width: '320px', height: '320px', borderRadius: '50%',
            background: 'rgba(255,255,255,0.18)', filter: 'blur(40px)', pointerEvents: 'none'
          }} />

          {/* ── Aria SVG character ── */}
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', position: 'relative', zIndex: 1 }}>
            <AriaCharacter st={status} />
          </div>

          {/* ── Name + status badge ── */}
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px',
            zIndex: 1, marginTop: '8px'
          }}>
            <div style={{ fontSize: '20px', fontWeight: '800', color: '#1a237e', letterSpacing: '0.3px' }}>Aria</div>
            <div style={{ fontSize: '12px', color: '#5c6bc0', fontWeight: '600', letterSpacing: '0.5px' }}>RNSIT Digital Receptionist</div>
            <div style={{
              padding: '5px 18px', borderRadius: '20px', background: statusBg,
              color: statusColor, fontSize: '13px', fontWeight: '700',
              transition: 'all 0.4s ease', boxShadow: '0 2px 10px rgba(0,0,0,0.10)'
            }}>
              {{ ready: '● Voice Ready', listening: '🎤 Listening to you…', processing: '💭 Thinking…', speaking: '🔊 Speaking…' }[status]}
            </div>
          </div>

          {/* hint pill (ready state only) */}
          {status === 'ready' && messages.length === 0 && (
            <div style={{
              marginTop: '14px', padding: '8px 20px', borderRadius: '20px',
              background: 'rgba(255,255,255,0.65)', backdropFilter: 'blur(8px)',
              border: '1px solid rgba(255,255,255,0.8)', fontSize: '12px',
              color: '#5c6bc0', fontStyle: 'italic', textAlign: 'center',
              maxWidth: '280px', zIndex: 1
            }}>
              {hints[hintIndex]}
            </div>
          )}
        </div>

        {/* ══════════ RIGHT: COMPACT CHAT ══════════ */}
        <div style={{
          flex: 1, display: 'flex', flexDirection: 'column', background: '#f8f9ff',
          borderLeft: '1.5px solid #e0e4ff', overflow: 'hidden'
        }}>

          {/* chat header */}
          <div style={{
            padding: '10px 16px', background: '#fff',
            borderBottom: '1px solid #e8eaf6', flexShrink: 0,
            display: 'flex', alignItems: 'center', gap: '8px'
          }}>
            <div style={{
              width: '8px', height: '8px', borderRadius: '50%',
              background: { ready: '#43a047', listening: '#43a047', processing: '#7e57c2', speaking: '#e53935' }[status] || '#43a047',
              transition: 'background 0.3s', boxShadow: '0 0 0 3px rgba(67,160,71,0.15)'
            }} />
            <span style={{ fontSize: '13px', fontWeight: '700', color: '#444' }}>Conversation</span>
            <span style={{ fontSize: '11px', color: '#bbb', marginLeft: 'auto' }}>
              {messages.length > 0 ? `${messages.length} message${messages.length > 1 ? 's' : ''}` : 'Just started'}
            </span>
          </div>

          {/* messages */}
          <div ref={scrollRef} style={{ flex: '1 1 0', overflowY: 'auto', padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>

            {messages.length === 0 && !liveText && status !== 'processing' && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '10px', padding: '30px 12px', textAlign: 'center' }}>
                <div style={{ fontSize: '36px', lineHeight: 1 }}>💬</div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#9fa8da' }}>Your conversation with Aria will appear here</div>
                <div style={{ fontSize: '12px', color: '#c5cae9' }}>Just speak — she&apos;s ready</div>
              </div>
            )}

            {messages.map((msg, i) => {
              const isAria = msg.speaker === 'kiosk';
              const prevSame = i > 0 && messages[i - 1].speaker === msg.speaker;
              return (
                <div key={i} style={{
                  display: 'flex', flexDirection: 'column',
                  alignItems: isAria ? 'flex-start' : 'flex-end',
                  marginTop: prevSame ? '2px' : '8px'
                }}>
                  {!prevSame && (
                    <span style={{
                      fontSize: '10px', color: '#bbb', marginBottom: '2px',
                      paddingLeft: isAria ? '6px' : 0, paddingRight: !isAria ? '6px' : 0, fontWeight: '600'
                    }}>
                      {isAria ? 'Aria' : visitorName}
                    </span>
                  )}
                  <div className="msg-in" style={{
                    maxWidth: '88%', padding: '8px 12px',
                    borderRadius: isAria
                      ? (prevSame ? '4px 14px 14px 14px' : '14px 14px 14px 4px')
                      : (prevSame ? '14px 4px 14px 14px' : '14px 14px 4px 14px'),
                    background: isAria ? '#ffffff' : '#1a237e',
                    color: isAria ? '#1a1a1a' : '#ffffff',
                    fontSize: '13.5px', lineHeight: '1.5',
                    boxShadow: isAria ? '0 1px 3px rgba(0,0,0,0.08)' : '0 1px 4px rgba(26,35,126,0.25)',
                    wordBreak: 'break-word'
                  }}>
                    {msg.text}
                    <span style={{ fontSize: '9px', color: isAria ? '#ccc' : 'rgba(255,255,255,0.5)', marginLeft: '6px', float: 'right', marginTop: '3px', whiteSpace: 'nowrap' }}>
                      {msg.timestamp}
                    </span>
                  </div>
                </div>
              );
            })}

              {/* FOLLOW-UP CHIPS: shown after the kiosk's most recent reply,
                  while idle (not mid-question). Turns "answer machine" into
                  something that keeps the conversation moving — tapping a
                  chip routes through the SAME sendToBackend() pipeline as a
                  spoken question, so it inherits every existing guard
                  (barge-in, stale-answer checks, goodbye handling) for free. */}
              {status === 'ready' && !processingHint && !liveText &&
                messages.length > 0 && messages[messages.length - 1].speaker === 'kiosk' && (
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', paddingLeft: '4px', marginTop: '2px' }}>
                  <button
                    onClick={() => sendToBackend('Anything else you can help with?')}
                    style={{ padding: '9px 18px', background: '#fff', border: '1.5px solid #c7cbe8',
                             borderRadius: '20px', fontSize: '14px', fontWeight: '600', color: '#3c4370',
                             cursor: 'pointer', boxShadow: '0 2px 6px rgba(26,35,126,0.05)' }}>
                    Ask something else
                  </button>
                  <button
                    onClick={() => sendToBackend('That is all, thank you')}
                    style={{ padding: '9px 18px', background: '#fff', border: '1.5px solid #c7cbe8',
                             borderRadius: '20px', fontSize: '14px', fontWeight: '600', color: '#3c4370',
                             cursor: 'pointer', boxShadow: '0 2px 6px rgba(26,35,126,0.05)' }}>
                    That's all, thanks
                  </button>
                </div>
              )}

              {processingHint && !liveText && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                  <div style={{ fontSize: '11px', color: '#bbb', marginBottom: '4px', paddingLeft: '4px', fontWeight: '500' }}>
                    RNSIT Kiosk &nbsp;·&nbsp; thinking
                  </div>
                  <div style={{ maxWidth: '60%', padding: '13px 18px', borderRadius: '4px 18px 18px 18px',
                                background: '#f3f2fb', color: '#6a6f8c', fontSize: '15.5px', fontStyle: 'italic',
                                lineHeight: '1.6', border: '1.5px dashed #d8d6ea' }}>
                    {processingHint}
                  </div>
                </div>
              )}
              {liveText && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', marginTop: '8px' }}>
                  <div style={{ fontSize: '11px', color: '#bbb', marginBottom: '4px', paddingRight: '4px' }}>{visitorName} (speaking...)</div>
                  <div style={{ maxWidth: '60%', padding: '14px 18px', borderRadius: '18px 4px 18px 18px', background: '#e8eaf6', color: '#1a237e', fontSize: '16px', fontStyle: 'italic', lineHeight: '1.65', border: '1.5px solid #c5cae9' }}>
                    {liveText}
                  </div>
                </div>
              )}
          </div>

          {/* voice footer */}
          <div style={{
            padding: '8px 12px', background: '#fff', borderTop: '1px solid #e8eaf6',
            flexShrink: 0, display: 'flex', alignItems: 'center', gap: '10px'
          }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '50%', flexShrink: 0,
              background: { ready: '#f5f5f5', listening: '#e8f5e9', processing: '#ede7f6', speaking: '#fce4ec' }[status],
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: status === 'listening' ? '0 0 0 5px rgba(67,160,71,0.12)' : 'none',
              transition: 'all 0.3s'
            }}>
              {status === 'speaking' ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#e53935" strokeWidth="2.2">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                  stroke={status === 'listening' ? '#43a047' : status === 'processing' ? '#7e57c2' : '#aaa'} strokeWidth="2.2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" y1="19" x2="12" y2="23" />
                  <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
              )}
            </div>
            <canvas ref={canvasRef} width={130} height={32}
              style={{ borderRadius: '6px', background: 'rgba(67,160,71,0.05)', display: listening ? 'block' : 'none' }} />
            {!listening && (
              <span style={{
                fontSize: '12px', fontWeight: '700',
                color: { ready: '#aaa', listening: '#43a047', processing: '#7e57c2', speaking: '#e53935' }[status],
                transition: 'color 0.3s'
              }}>
                {statusLabel}
              </span>
            )}
            <span style={{ fontSize: '10px', color: '#ddd', marginLeft: 'auto' }}>RNSIT · Aria AI</span>
          </div>
        </div>
      </div>

      {/* ── FLOATING CAMERA PIP ── */}
      <div style={{
        position: 'fixed', bottom: '16px', right: '16px', width: '80px', height: '80px',
        borderRadius: '50%', overflow: 'hidden', border: '3px solid #fff',
        boxShadow: '0 4px 16px rgba(0,0,0,0.22)', zIndex: 20, background: '#c5cae9'
      }}>
        <video ref={camVideoRef} autoPlay playsInline muted
          style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', display: 'block' }} />
      </div>

      {/* ── STYLES ── */}
      <style>{`
        * { box-sizing:border-box; margin:0; padding:0; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-thumb { background:rgba(0,0,0,0.12); border-radius:4px; }
        input:focus { border-color:#1a237e !important; box-shadow:0 0 0 3px rgba(26,35,126,0.1); }

        /* ── message bubble spring-in ── */
        .msg-in { animation: msgIn 0.2s cubic-bezier(0.18,0.89,0.32,1.28) both; }
        @keyframes msgIn { from{opacity:0;transform:translateY(5px) scale(0.97)} to{opacity:1;transform:none} }

        /* ── typing dots ── */
        .td { display:inline-block; width:7px; height:7px; border-radius:50%; background:#c5cae9;
              animation:tdBounce 1.1s ease-in-out infinite; animation-delay:var(--d); }
        @keyframes tdBounce { 0%,60%,100%{transform:translateY(0);background:#c5cae9} 30%{transform:translateY(-6px);background:#7e57c2} }

        /* ══════════════════════════════
           ARIA CHARACTER ANIMATIONS
        ══════════════════════════════ */

        /* BODY — gentle breathing (always on) */
        .aria-svg .body-grp { animation: charBreathe 5s ease-in-out infinite; }
        @keyframes charBreathe { 0%,100%{transform:scaleY(1)} 50%{transform:scaleY(1.016)} }

        /* HEAD — base: idle micro-float */
        .aria-svg .head-grp { animation: idleFloat 6s ease-in-out infinite; }
        @keyframes idleFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }

        /* ── LISTENING ── */
        .aria-listening .head-grp { animation: listenLean 0.7s ease-out forwards, idleFloat 0s; }
        @keyframes listenLean { to{transform:translateX(10px) rotate(6deg)} }

        /* pulse ring */
        .listen-r1 { animation:lRing 2s ease-out infinite; }
        .listen-r2 { animation:lRing 2s 0.7s ease-out infinite; }
        @keyframes lRing { 0%{r:92;opacity:0.55} 100%{r:140;opacity:0} }

        /* ── PROCESSING ── */
        .aria-processing .head-grp { animation: thinkTilt 0.6s ease-out forwards, idleFloat 0s; }
        @keyframes thinkTilt { to{transform:translateX(-12px) rotate(-7deg)} }

        /* thinking arm rise */
        .aria-processing .arm-think { animation: armRise 0.6s ease-out both; transform-origin:265px 228px; }
        @keyframes armRise { from{transform:translateY(30px);opacity:0} to{transform:none;opacity:1} }

        /* thinking bubble float */
        .tbub { animation:tBubFloat 1.6s ease-in-out infinite; }
        @keyframes tBubFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }

        /* furrowed brow when thinking */
        .brow-think { animation: browFurrow 0.5s ease-out forwards; transform-origin:133px 86px; }
        @keyframes browFurrow { to{transform:translateY(4px)} }

        /* ── SPEAKING ── */
        .aria-speaking .head-grp { animation: headBob 0.55s ease-in-out infinite; }
        @keyframes headBob { 0%,100%{transform:translateY(0) rotate(0)} 30%{transform:translateY(-5px) rotate(1.5deg)} 70%{transform:translateY(2px) rotate(-1deg)} }

        /* mouth alternates: a visible ↔ b visible */
        .aria-speaking .mouth-a { animation: mA 0.38s ease-in-out infinite; }
        .aria-speaking .mouth-b { animation: mB 0.38s ease-in-out infinite; }
        @keyframes mA { 0%,49%{opacity:1} 50%,100%{opacity:0} }
        @keyframes mB { 0%,49%{opacity:0} 50%,100%{opacity:1} }

        /* sound waves stagger */
        .wave1 { animation: wv 1s 0.0s ease-in-out infinite; }
        .wave2 { animation: wv 1s 0.2s ease-in-out infinite; }
        .wave3 { animation: wv 1s 0.4s ease-in-out infinite; }
        @keyframes wv { 0%,100%{opacity:0.2;stroke-width:2} 50%{opacity:0.9;stroke-width:3} }
      `}</style>
    </div>
  );
}


// ── Aria's avatar: used in message rows and header ───────────────────────
const AriaAvatar = ({ size = 38, speaking = false }) => (
  <div style={{ position: 'relative', flexShrink: 0 }}>
    {speaking && (
      <>
        <div className="avatar-ring r1" style={{ width: size + 14, height: size + 14, top: -7, left: -7 }} />
        <div className="avatar-ring r2" style={{ width: size + 22, height: size + 22, top: -11, left: -11 }} />
      </>
    )}
    <div style={{
      width: size, height: size, borderRadius: '50%',
      background: 'linear-gradient(135deg, #1a237e 0%, #5c35c9 60%, #7c4dff 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      boxShadow: speaking ? '0 0 0 3px #7c4dff55' : '0 2px 8px rgba(26,35,126,0.30)',
      flexShrink: 0, overflow: 'hidden', position: 'relative',
    }}>
      {/* stylised person silhouette */}
      <svg width={size * 0.62} height={size * 0.62} viewBox="0 0 40 40" fill="none">
        <circle cx="20" cy="14" r="7" fill="rgba(255,255,255,0.92)" />
        <path d="M4 38 C4 28 36 28 36 38" fill="rgba(255,255,255,0.92)" />
      </svg>
    </div>
  </div>
);
