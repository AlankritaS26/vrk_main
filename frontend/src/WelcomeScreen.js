import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createKioskMic, float32ToInt16 } from './kioskMic';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8001';

export default function WelcomeScreen({ session, messages, setMessages, askingName, detState, doubleBlink, blink }) {
  const scrollRef = useRef(null);
  const camVideoRef = useRef(null);
  const camStreamRef = useRef(null);
  const isMounted = useRef(true);
  const isSpeaking = useRef(false);
  const awaitingAnswerRef = useRef(false);     // true from "ack started" until the real answer's speech starts/fails —
  // keeps status at 'processing' (not 'ready') through that gap
  const requestSeqRef = useRef(0);             // increments per question asked; used to drop an answer ONLY when a
  // *newer* question has since been asked — NOT when the backend's session
  // bookkeeping (session_id) happens to churn while the answer is in flight
  // (e.g. a slow LLM fallback overlapping a face-detection re-engagement
  // cycle). Comparing session_id for "staleness" was dropping perfectly
  // valid answers whenever the backend ended/resumed the session mid-request.
  const interruptSpeakingRef = useRef(null);   // lets the WS handler stop TTS
  const farewellPlayingRef = useRef(false);    // true while the goodbye line plays
  const greetingPlayingRef = useRef(false);    // true while the NEW-VISITOR greeting plays
  // (explicitly non-interruptible, per spec — it
  // is one short message that must always finish)
  const handlingDepartureRef = useRef(false);  // true while handling 3s face departure prompt
  const isListening = useRef(false);
  const analyserRef = useRef(null);
  const animFrameRef = useRef(null);
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const statusRef = useRef('ready');        // readable inside callbacks
  const detStateRef = useRef(detState || 'IDLE');
  useEffect(() => { detStateRef.current = detState || 'IDLE'; }, [detState]);
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

  const [privacyOpen, setPrivacyOpen] = useState(false);
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

  // ── Voice-based name capture (NO TYPING) ──────────────────────────────────
  // Replaces the old "type your name" modal entirely. Nova asks out loud,
  // listens for the spoken answer via the same STT pipeline used for
  // regular questions, then asks (out loud) whether to remember the
  // visitor — answerable by voice ("yes"/"no") or, experimentally, by
  // blinking once for "yes". A single tap fallback ("Continue as Guest")
  // stays available for accessibility/robustness, but there is no keyboard
  // entry anywhere in this flow.
  const [nameStage, setNameStage] = useState('idle');
  // idle | asking | listening_name | confirming | listening_confirm | saving | done
  const nameFlowIdRef = useRef(0);          // bumped to invalidate an in-flight run

  const [localName, setLocalName] = useState('');
  const pendingCandidateNameRef = useRef('');
  const visitorName = localName || (session?.user_name && session.user_name !== 'Unknown' ? session.user_name : 'Guest');
  const isReturning = session?.is_returning || false;
  const visitCount = session?.visit_count || 1;

  // The backend composes the greeting (it knows resume-vs-new and the
  // institute intro line); these local strings are only a fallback.
  const greeting = session?.greeting || (isReturning
    ? (visitCount > 2
      ? 'Welcome back, ' + visitorName + '! Great to see you again. How may I assist you today?'
      : 'Welcome back, ' + visitorName + '! How may I assist you today?')
    : 'Welcome to R N S Institute of Technology. I am Nova, your digital receptionist. '
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
  const activePromptResolverRef = useRef(null);
  const lastProcessedTextRef = useRef({ text: '', time: 0 });

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const handleUtterance = useCallback(async (float32Audio) => {
    if (!isMounted.current) return;
    if (greetingPlayingRef.current) {
      return;
    }
    if (isSpeaking.current) {
      interruptSpeaking();
    } else if (!activePromptResolverRef.current && statusRef.current === 'processing') {
      return;
    }
    isListening.current = false;
    setListening(false);
    statusRef.current = 'processing';
    setStatus('processing');

    try {
      const i16 = float32ToInt16(float32Audio);
      const response = await fetch(BACKEND + '/stt/pcm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: i16.buffer
      });

      const result = await response.json();
      console.log('[STT WHISPER]', result);
      const heard = (result.text || '').trim();

      // If a specific conversation prompt (e.g. name prompt, Yes/No confirm) is waiting for speech:
      if (activePromptResolverRef.current) {
        if (heard && heard.length > 0) {
          const resolver = activePromptResolverRef.current;
          activePromptResolverRef.current = null;
          if (isMounted.current) setLiveText(heard);
          statusRef.current = 'ready';
          setStatus('ready');
          resolver(heard);
          return;
        } else {
          console.log('[STT] Empty transcript during prompt wait, continuing to wait for speech');
          statusRef.current = 'ready';
          setStatus('ready');
          return;
        }
      }

      const now = Date.now();
      if (heard && heard.length > 1 && !isSpeaking.current) {
        // Prevent duplicate input processing within 2.5 seconds
        if (lastProcessedTextRef.current.text.toLowerCase() === heard.toLowerCase() && (now - lastProcessedTextRef.current.time) < 2500) {
          console.log('[STT] Dropped duplicate heard text within 2.5s:', heard);
          setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
          return;
        }
        lastProcessedTextRef.current = { text: heard, time: now };
        if (isMounted.current) setLiveText(heard);
        sendToBackend(heard);
      } else {
        setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
      }
    } catch (err) {
      console.error('[STT] Error:', err);
      if (activePromptResolverRef.current) {
        const resolver = activePromptResolverRef.current;
        activePromptResolverRef.current = null;
        resolver('');
      }
      setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
    }
  }, []);

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
    if (!isMounted.current) return;

    // Mic already initialized — check if the underlying stream is still alive
    if (micRef.current) {
      const trackEnded = streamRef.current
        && streamRef.current.getTracks().some(t => t.readyState === 'ended');
      if (trackEnded) {
        console.info('[MIC] Track ended (OS mic toggled) — reinitializing VAD');
        micRef.current.destroy();
        micRef.current = null;
        streamRef.current = null;
        stopWaveform();
      } else if (!isSpeaking.current) {
        micRef.current.resume();
        setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
        return;
      } else {
        return;
      }
    }

    try {
      const mic = await createKioskMic({
        onStream: (stream) => { streamRef.current = stream; },
        onSpeechStart: () => {
          if (!isMounted.current) return;
          if (isSpeaking.current) {
            if (greetingPlayingRef.current) return;
            interruptSpeaking();
          }
          isListening.current = true;
          setListening(true);
          setLiveText('');
          setStatus('listening');
          if (streamRef.current) startWaveform(streamRef.current);
        },
        onSpeechEnd: (audio) => {
          handleUtterance(audio);
        },
        onMisfire: () => {
          if (!isMounted.current) return;
          if (isSpeaking.current) {
            restoreSpeaking();
          } else {
            isListening.current = false;
            setListening(false);
            setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
          }
        },
      });
      micRef.current = mic;
      if (!isSpeaking.current) setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
    } catch (err) {
      console.error('[MIC] Error:', err);
      if (!isSpeaking.current) setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
    }
  }, [startWaveform, handleUtterance, duckSpeaking, restoreSpeaking]);

  // release mic + VAD on unmount (session end)
  useEffect(() => () => {
    micRef.current?.destroy();
    micRef.current = null;
  }, []);

  // auto-boot mic when component mounts
  useEffect(() => {
    startListening();
  }, [startListening]);

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
      if (onDone) { try { onDone(); } catch (e) { } onDone = null; }
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
      if (onSentence) { try { onSentence(text, 0); } catch (e) { } }
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
          if (onSentence) { try { onSentence(sentenceText, sentenceIndex); } catch (e) { } }
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
  // sync (ack bubble, farewell, greeting, error fallback, name flow) — same
  // (onStart, onDone) signature as before.
  const speak = useCallback((text, onStart, onDone) => (
    speakStream(text, { onStart, onDone })
  ), [speakStream]);

  // Promise-returning wrapper: resolves once THIS utterance has fully
  // finished playing. Pauses the mic during prompts to prevent speaker echo.
  const speakAndWait = useCallback((text, onStart) => (
    new Promise((resolve) => {
      micRef.current?.pause();
      speak(text, onStart, () => {
        if (isMounted.current && micRef.current) {
          micRef.current.resume();
        }
        resolve();
      });
    })
  ), [speak]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const sendToBackend = useCallback(async (text) => {
    if (!text) return;
    setLiveText('');
    setProcessingHint('');
    const sid = session?.session_id || 'guest';
    const myReqSeq = ++requestSeqRef.current;   // this question's sequence number
    addMessage(text, 'user');

    // ── Check if visitor is affirming a pending name confirmation ──────
    const isAffirmation = /^(?:yes|yeah|yep|yup|sure|ok|okay|correct|right|true|thats right|that is right|thats me|that is me|yes please|i am|it is|confirm|confirmed)[.!?]*$/i.test(text.trim())
      || /\b(?:yes that is my name|yes that is correct|yes that is me|thats my name|that's my name)\b/i.test(text.trim());
    if (pendingCandidateNameRef.current && isAffirmation) {
      const confirmedName = pendingCandidateNameRef.current;
      pendingCandidateNameRef.current = '';
      setLocalName(confirmedName);
      await submitVoiceName(confirmedName, true);
      const greetNamed = `Great to meet you, ${confirmedName}! How may I assist you today?`;
      addMessage(greetNamed, 'kiosk');
      speak(greetNamed);
      return;
    }

    // ── Mid-session bare "change my name" prompt (no name given yet) ──────
    // Explicit "change my name to X" / "call me X" patterns are handled by
    // the backend _deterministic_route via /ask below — do NOT early-return
    // for those, or save_interaction will be skipped and the DB won't log it.
    const bareNameChange = /\b(?:change|update|reset|rename)\s+(?:my\s+|the\s+)?name\b/i.test(text)
      || /\b(?:i want to|can i|can you|please|how do i)\s+(?:change|update|reset|rename)\s+(?:my\s+|the\s+)?name\b/i.test(text);

    // Only fire the interactive prompt when NO name was provided inline
    const hasInlineName = /\b(?:change|update|set|rename)\s+(?:my\s+|the\s+)?name\s+to\s+\w/i.test(text)
      || /\b(?:call me|my name is|actually my name is|its actually|it's actually|no my name is|i am called|this is)\s+\w/i.test(text);

    const isQuestionText = text.includes('?') || /\b(where|what|how|when|who|which|can|tell|fees|admission|hostel|placement|library|department|principal|hod|contact|address|course|branch|branches|syllabus|exam|seat|cutoff|rnsit|college|campus|building|block|canteen|sports)\b/i.test(text);

    // If the visitor directly stated their name or spelled it (e.g. "Akshata", "My name is Akshata", "I am Akshata"):
    if (!isQuestionText && !bareNameChange) {
      const candidateName = extractVisitorName(text);
      if (candidateName && candidateName.split(' ').length <= 3 && !/^(yes|no|guest|skip|continue|ok|okay|bye|thanks|thank you)$/i.test(candidateName)) {
        setLocalName(candidateName);
        fetch(BACKEND + '/visitor/rename?name=' + encodeURIComponent(candidateName), { method: 'POST' }).catch(() => {});
        const doneMsg = `Done! I have changed your name to ${candidateName}. How may I assist you today?`;
        addMessage(doneMsg, 'kiosk');
        speak(doneMsg);
        return;
      }
    }

    if (bareNameChange && !hasInlineName) {
      // ── Step 1: Ask for the new name ────────────────────────────────────
      const promptChange = 'Sure! What should I change your name to?';
      addMessage(promptChange, 'kiosk');
      await speakAndWait(promptChange);
      const heardNewName = await captureUtteranceText(25000);

      if (!heardNewName) {
        const cancelMsg = 'No problem. Let me know if you would like to change your name or ask a question.';
        addMessage(cancelMsg, 'kiosk');
        speak(cancelMsg);
        return;
      }

      addMessage(heardNewName, 'user');
      let extracted = extractVisitorName(heardNewName) || heardNewName.trim();
      extracted = extracted.replace(/[.!?]+$/, '').trim();
      const cleanWords = extracted.split(/\s+/).filter(w =>
        !/^(what|who|where|how|why|which|nova|kiosk|please|my|name|is|to|the)$/i.test(w));

      if (cleanWords.length === 0) {
        const cancelMsg = 'No problem. Let me know if you would like to change your name or ask a question.';
        addMessage(cancelMsg, 'kiosk');
        speak(cancelMsg);
        return;
      }

      extracted = cleanWords.map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');

      // ── Step 2: Confirm with voice OR double-blink ───────────────────────
      const confirmMsg = `Got it — should I call you ${extracted}? Say yes or blink twice to confirm, or say no to spell it out.`;
      addMessage(confirmMsg, 'kiosk');
      await speakAndWait(confirmMsg);
      const confirmed = await captureYesNo(25000);

      // Helper: apply the final name to DB + session
      const applyName = async (finalName) => {
        setLocalName(finalName);
        await fetch(BACKEND + '/visitor/rename?name=' + encodeURIComponent(finalName), { method: 'POST' }).catch(() => {});
        const doneMsg = `Done! I have changed your name to ${finalName}. How may I assist you today?`;
        addMessage(doneMsg, 'kiosk');
        speak(doneMsg);
      };

      // ── Helper: letter-by-letter spelling mode ────────────────────────────
      // Nova echoes each letter as it is heard so the visitor can track
      // progress. Phonetic alphabet (alpha/bravo/charlie…) is also accepted.
      const runSpellingMode = async () => {
        const PHONETIC = {
          alpha:'a', bravo:'b', charlie:'c', delta:'d', echo:'e', foxtrot:'f',
          golf:'g', hotel:'h', india:'i', juliet:'j', kilo:'k', lima:'l',
          mike:'m', november:'n', oscar:'o', papa:'p', quebec:'q', romeo:'r',
          sierra:'s', tango:'t', uniform:'u', victor:'v', whiskey:'w',
          xray:'x', 'x-ray':'x', yankee:'y', zulu:'z',
        };
        const spellPrompt = 'Sure! Please spell out your name — say each letter one at a time. Say "done" when you are finished.';
        addMessage(spellPrompt, 'kiosk');
        await speakAndWait(spellPrompt);

        let spelled = '';
        let attempts = 0;
        while (attempts < 25) {
          const letter = await captureUtteranceText(7000);
          if (!letter) break;

          const t = letter.trim().toLowerCase();
          // Finish words
          if (/^(done|finish|finished|that.?s it|stop|end|complete|that.?s all|ok done)$/i.test(t)) break;

          let ch = '';
          if (t.length === 1 && /[a-z]/.test(t)) {
            ch = t.toUpperCase();
          } else if (PHONETIC[t]) {
            ch = PHONETIC[t].toUpperCase();
          } else if (/^[a-z]\s/i.test(t)) {
            // e.g. STT returns "P." or "P " for a single letter
            ch = t[0].toUpperCase();
          }

          if (ch) {
            spelled += ch;
            const soFar = spelled.split('').join('-');
            const echoMsg = `${ch}. So far: ${soFar}`;
            addMessage(echoMsg, 'kiosk');
            speak(echoMsg);
          } else {
            // Unrecognised syllable — ask them to repeat
            const retryMsg = "Sorry, I did not catch that letter. Please say it again.";
            addMessage(retryMsg, 'kiosk');
            speak(retryMsg);
          }
          attempts++;
        }

        if (spelled.length === 0) return;

        // Capitalise first letter, rest lowercase
        const spelledName = spelled.charAt(0).toUpperCase() + spelled.slice(1).toLowerCase();

        // Final confirmation after spelling
        const spelledConfirmMsg = `I have ${spelledName}. Is that correct? Say yes or blink twice.`;
        addMessage(spelledConfirmMsg, 'kiosk');
        await speakAndWait(spelledConfirmMsg);
        const spelledOk = await captureYesNo(12000);

        if (spelledOk !== false) {
          // Accept on yes, double-blink, or timeout (visitor stayed silent)
          await applyName(spelledName);
        } else {
          const giveUpMsg = 'No problem — I will keep your name as it is for now. You can try again anytime.';
          addMessage(giveUpMsg, 'kiosk');
          speak(giveUpMsg);
        }
      };

      if (confirmed === true) {
        // Voice "yes" or double-blink confirmed
        await applyName(extracted);
      } else if (
        confirmed === false ||
        (typeof confirmed === 'string' && /\b(no|nope|wrong|spell|spelling|incorrect|not right)\b/i.test(confirmed))
      ) {
        // User said no / "spell it" — enter spelling mode
        await runSpellingMode();
      } else {
        // captureYesNo timed out (null) — accept the heard name
        await applyName(extracted);
      }
      return;
    }

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
      fetch(BACKEND + '/session/end?session_id=' + sid, { method: 'POST' }).catch(() => { });
      window.dispatchEvent(new CustomEvent('vrk-session-ended', { detail: { farewell, userName: localName || session?.user_name } }));   // goodbye screen appears now
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
    const isInstantCmd = hasInlineName || bareNameChange ||
      /^(hi|hello|hey|good morning|good afternoon|good evening|bye|thank you|thanks)/i.test(text.trim());

    if (!isInstantCmd && text.split(' ').length >= 3) {
      const ack = acks[Math.floor(Math.random() * acks.length)];
      speak(ack, () => setProcessingHint(ack));
    } else {
      setProcessingHint('Thinking...');
      statusRef.current = 'processing';
      setStatus('processing');
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

      // STALE-ANSWER GUARD: only drop if the backend explicitly says so, or if
      // the visitor has since asked ANOTHER question that superseded this one.
      // We intentionally do NOT compare session_id here anymore — the backend
      // can end/resume a session (face-detection hiccups, re-engagement
      // lookups) while a slow /ask call (e.g. local-LLM timeout -> Gemini
      // fallback) is still in flight for the SAME visitor, which used to make
      // this guard discard a perfectly valid, on-topic answer and leave the
      // UI stuck on the "just a second" filler forever.
      if (data.dropped || myReqSeq !== requestSeqRef.current) {
        console.info('[sendToBackend] dropped stale answer for', sid);
        awaitingAnswerRef.current = false;
        isSpeaking.current = false;
        setProcessingHint('');
        setStatus('ready');
        return;
      }

      const answer = data.answer || 'Sorry, I do not have that information. Please visit the Admin Block.';
      fetch(BACKEND + '/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, text: answer, speaker: 'kiosk' })
      });

      // Clear processing hint when answer arrives
      setProcessingHint('');

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

  // ── Helper parsing for name & guest choices ──
  // Words that must NEVER be treated as a visitor name regardless of context.
  const _REJECTION_WORDS = /^(no|nope|nah|nah|wrong|incorrect|change|not|different|cancel|stop|skip|guest|unknown|friend|none|null|undefined|yes|yeah|yep|yup|sure|ok|okay)$/i;

  const extractVisitorName = useCallback((raw) => {
    if (!raw) return '';
    let s = raw.trim();
    s = s.replace(/^(?:hi|hello|hey|nova|please|ok|okay)?[\s,.]*(?:my name is|i am called|call me|myself|i am|im|it's|its|this is)\s+/i, '');
    s = s.replace(/^(?:hi|hello|hey|nova|please)[\s,.]+/i, '');
    s = s.replace(/[.!?]+$/, '').trim();
    if (!s) return '';
    if (s.includes(',')) {
      const firstPart = s.split(',')[0].trim();
      if (firstPart && /^[a-zA-Z\s]+$/.test(firstPart)) {
        s = firstPart;
      }
    }
    const words = s.split(/\s+/).filter(w => !/^(what|who|where|how|why|which|nova|kiosk|please|my|name|is|to|the)$/i.test(w));
    if (words.length === 0) return '';
    // Guard: if the entire result is a single rejection/control word, return empty
    const result = words.map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
    if (words.length === 1 && _REJECTION_WORDS.test(words[0])) return '';
    return result;
  }, []);

  // ── Double Blink Listener for Yes/Confirm ──────────────────────────────
  const prevDoubleBlinkRef = useRef(0);
  // Latches a double-blink that fired while no prompt was active (e.g. while
  // Nova is speaking). captureUtteranceText/captureYesNo consume it instantly
  // on their next call so the blink is never silently lost.
  const pendingBlinkRef = useRef(false);

  useEffect(() => {
    if (doubleBlink && doubleBlink !== prevDoubleBlinkRef.current) {
      prevDoubleBlinkRef.current = doubleBlink;
      console.log('[BLINK] Double blink detected!');
      if (activePromptResolverRef.current) {
        // A prompt is already waiting — resolve it immediately
        const resolver = activePromptResolverRef.current;
        activePromptResolverRef.current = null;
        statusRef.current = 'ready';
        setStatus('ready');
        resolver('👁️ [Blinked twice — Yes]');
      } else {
        // No prompt active yet (Nova still speaking) — latch it so the
        // NEXT captureUtteranceText/captureYesNo call picks it up instantly
        console.log('[BLINK] No resolver active — latching blink for next prompt');
        pendingBlinkRef.current = true;
      }
    }
  }, [doubleBlink]);

  const wantsToGiveName = useCallback((text) => {
    if (!text) return false;
    return /\b(yes|yeah|yep|yup|sure|ok|okay|why not|of course|certainly|definitely|i do|i would|i want|give name|give my name|my name|tell name|tell my name|provide name|share name|enter name|yes please|i will|blink|blinked)\b/i.test(text)
      || text.includes('👁️') || text.toLowerCase().includes('blink');
  }, []);

  const isGuestOption = useCallback((text) => {
    if (!text) return false;
    return /\b(guest|guest mode|continue as guest|as guest|no name|anonymous|just guest)\b/i.test(text);
  }, []);

  const isContinueOption = useCallback((text) => {
    if (!text) return false;
    return /\b(skip|dont want|neither|no thanks|continue|just continue|start|just start|proceed|dont give)\b/i.test(text);
  }, []);

  // ── Voice prompt capture helpers (uses single persistent mic) ──────────
  const captureUtteranceText = useCallback((timeoutMs = 25000) => {
    return new Promise((resolve) => {
      // If a double-blink was latched while Nova was speaking, consume it now
      if (pendingBlinkRef.current) {
        pendingBlinkRef.current = false;
        resolve('👁️ [Blinked twice — Yes]');
        return;
      }
      let timer = null;
      const resolver = (text) => {
        if (timer) clearTimeout(timer);
        resolve((text || '').trim());
      };
      const checkTimeout = () => {
        // If user is currently speaking or audio is being transcribed (Whisper STT), keep waiting!
        if (isListening.current || statusRef.current === 'processing' || statusRef.current === 'listening') {
          timer = setTimeout(checkTimeout, 3000);
          return;
        }
        if (activePromptResolverRef.current === resolver) {
          activePromptResolverRef.current = null;
        }
        resolve('');
      };
      timer = setTimeout(checkTimeout, timeoutMs);

      activePromptResolverRef.current = resolver;
      if (micRef.current) {
        micRef.current.resume();
      }
      setStatus('ready');
    });
  }, []);


  // Waits for a spoken "yes"/"no" response, double blink, or direct correction
  const captureYesNo = useCallback((timeoutMs = 25000) => {
    return new Promise((resolve) => {
      // If a double-blink was latched while Nova was speaking, consume it now
      if (pendingBlinkRef.current) {
        pendingBlinkRef.current = false;
        resolve(true);
        return;
      }
      let timer = null;
      const resolver = (rawText) => {
        if (timer) clearTimeout(timer);
        const heard = (rawText || '').trim().toLowerCase();
        if (/\b(yes|yeah|yep|yup|sure|ok|okay|please|correct|right|true|thats right|that is right|thats me|that is me|yes please|i am|it is|blink|blinked)\b/i.test(heard) || heard.includes('👁️')) {
          resolve(true);
        } else if (/\b(no|nope|nah|wrong|incorrect|not right|not that|different|change)\b/i.test(heard) || /don.?t/i.test(heard)) {
          resolve(false);
        } else if (rawText && rawText.trim().length > 0) {
          resolve(rawText.trim());
        } else {
          resolve(null);
        }
      };
      const checkTimeout = () => {
        // If user is currently speaking or audio is being transcribed, keep waiting!
        if (isListening.current || statusRef.current === 'processing' || statusRef.current === 'listening') {
          timer = setTimeout(checkTimeout, 3000);
          return;
        }
        if (activePromptResolverRef.current === resolver) {
          activePromptResolverRef.current = null;
        }
        resolve(null);
      };
      timer = setTimeout(checkTimeout, timeoutMs);

      activePromptResolverRef.current = resolver;
      if (micRef.current) {
        micRef.current.resume();
      }
      setStatus('ready');
    });
  }, []);


  const submitVoiceName = useCallback(async (finalName, save) => {
    try {
      await fetch(BACKEND + '/visitor/submit_name?name=' + encodeURIComponent(finalName) +
        '&save=' + save, { method: 'POST' });
    } catch (e) { console.error('[NAME FLOW] submit failed', e); }
    if (isMounted.current) setNameStage('done');
  }, []);

  // ── Integrated Conversation Start Flow (Voice + Double-Blink: Yes / No / Guest / Name) ──
  const greetedRef = useRef(null);
  const flowRunningRef = useRef(false);
  const [celebrate, setCelebrate] = useState(false);

  const runSessionStartFlow = useCallback(async () => {
    const sid = session?.session_id || 'active_session';
    if (greetedRef.current === sid || flowRunningRef.current) return;
    greetedRef.current = sid;
    flowRunningRef.current = true;

    const myRun = ++nameFlowIdRef.current;
    const stillCurrent = () => nameFlowIdRef.current === myRun && isMounted.current;

    try {
      const isKnownNamedVisitor = (visitorName && visitorName !== 'Guest' && visitorName !== 'Unknown' && visitorName !== 'Friend')
        || (session?.user_name && session.user_name !== 'Guest' && session.user_name !== 'Unknown' && session.user_name !== 'Friend');
      if (isKnownNamedVisitor) {
        // Visitor already has a known/changed name: greet immediately and start listening for questions
        const nameToUse = (visitorName && visitorName !== 'Guest' && visitorName !== 'Unknown') ? visitorName : session?.user_name;
        const greetMsg = greeting || `Welcome back, ${nameToUse}! How may I assist you today?`;
        addMessage(greetMsg, 'kiosk');
        await speakAndWait(greetMsg);
        if (stillCurrent()) startListening();
        return;
      }

      // First time or guest visitor: celebrate burst + single unified welcome & name prompt
      setCelebrate(true);
      setTimeout(() => setCelebrate(false), 2400);

      while (stillCurrent()) {
        setNameStage('asking');
        const askChatMsg = 'Welcome to RNS Institute of Technology! I am Nova, your digital receptionist.\n\nWould you like to give your name or continue as guest?\n\n• 🗣️ Say "Yes" or 👁️ Blink twice to give your name\n• 🗣️ Say "Guest" to continue as Guest';
        const askSpokenMsg = 'Welcome to R N S Institute of Technology! I am Nova, your digital receptionist. Would you like to give your name, or continue as guest? You can say yes or blink twice to give your name, or say guest to continue as guest.';
        addMessage(askChatMsg, 'kiosk');
        await speakAndWait(askSpokenMsg);
        if (!stillCurrent()) return;

        setNameStage('listening_name');
        const heard = await captureUtteranceText(6000);
        if (!stillCurrent()) return;

        let choseGiveName = false;
        let directNameProvided = null;

        if (heard) {
          addMessage(heard, 'user');

          if (wantsToGiveName(heard)) {
            // User affirmed verbally or with double-blink: e.g. "Yes", "👁️ [Blinked twice — Yes]"
            choseGiveName = true;
          } else if (isGuestOption(heard)) {
            // User chose Guest verbally
            const guestMsg = 'Continuing as Guest! How may I assist you today?';
            addMessage(guestMsg, 'kiosk');
            setLocalName('Guest');
            await submitVoiceName('Guest', false);
            await speakAndWait(guestMsg);
            if (stillCurrent()) startListening();
            break;
          } else if (isContinueOption(heard)) {
            // User chose skip/continue
            const contMsg = "Sure, let's continue! How may I assist you today?";
            addMessage(contMsg, 'kiosk');
            setLocalName('Guest');
            await submitVoiceName('Guest', false);
            await speakAndWait(contMsg);
            if (stillCurrent()) startListening();
            break;
          } else {
            // Check if user spoke a campus question directly
            const cleanNameCandidate = extractVisitorName(heard);
            const isQuestion = heard.includes('?') || heard.split(' ').length >= 4 ||
              /\b(where|what|how|when|who|which|can|tell|fees|admission|hostel|placement|library|department|principal|hod)\b/i.test(heard);

            if (isQuestion && !cleanNameCandidate) {
              setLocalName('Guest');
              await submitVoiceName('Guest', false);
              sendToBackend(heard);
              break;
            }

            // User directly provided their name: e.g. "Rahul", "Akshatha", "My name is John"
            directNameProvided = cleanNameCandidate || heard.trim();
          }
        } else {
          // Timeout — check if anyone is still in front of the camera before
          // defaulting to Guest. If the person walked away while we were
          // waiting, silently abort rather than creating a phantom session.
          const stateNow = detStateRef.current;
          if (stateNow === 'IDLE' || stateNow === 'COOLDOWN') {
            // Nobody there — abort the flow entirely
            break;
          }
          const noAnsMsg = 'Continuing as Guest! How may I assist you today?';
          addMessage(noAnsMsg, 'kiosk');
          setLocalName('Guest');
          await submitVoiceName('Guest', false);
          await speakAndWait(noAnsMsg);
          if (stillCurrent()) startListening();
          break;
        }

        // If user indicated they want to give their name (or direct name not provided yet)
        let finalName = directNameProvided;
        if (choseGiveName && !finalName) {
          setNameStage('asking');
          const askNamePrompt = 'Great! What is your name?';
          addMessage(askNamePrompt, 'kiosk');
          await speakAndWait(askNamePrompt);
          if (!stillCurrent()) return;

          setNameStage('listening_name');
          const heardSpokenName = await captureUtteranceText(25000);
          if (!stillCurrent()) return;

          if (heardSpokenName) {
            addMessage(heardSpokenName, 'user');
            // Guard: never treat a single rejection/negation word as a visitor name
            const _NAME_REJECTION = /^(no|nope|nah|wrong|incorrect|change|not|different|cancel|stop|skip|guest|unknown|friend)$/i;
            const extractedName = extractVisitorName(heardSpokenName);
            if (extractedName && !_NAME_REJECTION.test(extractedName.trim())) {
              finalName = extractedName;
            } else if (!_NAME_REJECTION.test(heardSpokenName.trim())) {
              finalName = heardSpokenName.trim();
            } else {
              finalName = null; // will trigger the retry below
            }
          } else {
            // Ask once more if missed
            const askRetry = "Could you please say your name?";
            addMessage(askRetry, 'kiosk');
            await speakAndWait(askRetry);
            if (!stillCurrent()) return;

            const retrySpoken = await captureUtteranceText(25000);
            if (retrySpoken) {
              addMessage(retrySpoken, 'user');
              finalName = extractVisitorName(retrySpoken) || retrySpoken.trim();
            } else {
              finalName = 'Friend';
            }
          }
        }

        if (!finalName) finalName = 'Friend';

        // 3. Confirm name with voice ("Yes" / "No") or double blink ("Yes")
        setNameStage('confirming');
        pendingCandidateNameRef.current = finalName;
        const confirmChatMsg = `I heard ${finalName}. Is that correct?\n\n• 🗣️ Say "Yes" or 👁️ Blink twice to confirm\n• 🗣️ Say "No" to change it`;
        const confirmSpokenMsg = `I heard ${finalName}. Is that correct? Say yes or blink twice to confirm, or say no to change it.`;
        addMessage(confirmChatMsg, 'kiosk');
        await speakAndWait(confirmSpokenMsg);
        if (!stillCurrent()) return;

        setNameStage('listening_confirm');
        const confirmed = await captureYesNo(45000);
        if (!stillCurrent()) return;

        if (confirmed === true) {
          pendingCandidateNameRef.current = '';
          setNameStage('saving');
          const greetNamed = `Great to meet you, ${finalName}! How may I assist you today?`;
          addMessage(greetNamed, 'kiosk');
          setLocalName(finalName);
          await submitVoiceName(finalName, true);
          await speakAndWait(greetNamed);
          if (stillCurrent()) startListening();
          break;
        } else if (confirmed === false || (typeof confirmed === 'string' && confirmed.length > 0)) {
          pendingCandidateNameRef.current = '';
          let correctedName = '';
          if (typeof confirmed === 'string' && confirmed.length > 0 && !/\b(no|nope|nah|wrong|change|not)\b/i.test(confirmed)) {
            correctedName = extractVisitorName(confirmed) || confirmed.trim();
          } else {
            setNameStage('asking');
            const retryMsg = "My apologies! Could you please spell out your name?";
            addMessage(retryMsg, 'kiosk');
            await speakAndWait(retryMsg);
            if (!stillCurrent()) return;

            setNameStage('listening_name');
            const retrySpokenName = await captureUtteranceText(25000);
            if (!stillCurrent()) return;

            // Guard: never accept rejection/negation words as a name
            const _REJECTION = /^(no|nope|nah|wrong|incorrect|change|not|different|cancel|stop|skip|guest|unknown|friend)$/i;
            const extracted = extractVisitorName(retrySpokenName);
            if (extracted && !_REJECTION.test(extracted.trim())) {
              correctedName = extracted;
            } else if (retrySpokenName && retrySpokenName.trim().split(/\s+/).length > 1) {
              // Multi-word response not matching rejection — treat as spelled name
              correctedName = retrySpokenName.trim();
            } else {
              // Still got a rejection word or silence — fall back to Guest rather than saving garbage
              correctedName = 'Guest';
            }
          }

          if (!correctedName || correctedName === 'Guest') {
            // Couldn't get a valid name — proceed as Guest
            setNameStage('saving');
            const guestFallback = 'No problem! Continuing as Guest. How may I assist you today?';
            addMessage(guestFallback, 'kiosk');
            await submitVoiceName('Guest', false);
            await speakAndWait(guestFallback);
            if (stillCurrent()) startListening();
            break;
          }

          setNameStage('saving');
          const changedMsg = `Done! Your name has been changed to ${correctedName}. How may I assist you today?`;
          addMessage(changedMsg, 'kiosk');
          setLocalName(correctedName);
          await submitVoiceName(correctedName, true);
          await speakAndWait(changedMsg);
          if (stillCurrent()) startListening();
          break;
        }

        // If captureYesNo timed out — check face presence before assuming name is confirmed
        const stateAtTimeout = detStateRef.current;
        if (stateAtTimeout === 'IDLE' || stateAtTimeout === 'COOLDOWN') {
          // Nobody in front anymore — abort silently
          break;
        }
        setNameStage('saving');
        const greetNamed = `Great to meet you, ${finalName}! How may I assist you today?`;
        addMessage(greetNamed, 'kiosk');
        setLocalName(finalName);
        await submitVoiceName(finalName, true);
        await speakAndWait(greetNamed);
        if (stillCurrent()) startListening();
        break;
      }
    } finally {
      flowRunningRef.current = false;
    }
  }, [session, isReturning, greeting, extractVisitorName, wantsToGiveName, isGuestOption, isContinueOption, captureUtteranceText, captureYesNo, submitVoiceName, speakAndWait, addMessage, sendToBackend, startListening]);

  useEffect(() => {
    runSessionStartFlow();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.session_id]);

  // ── Backend event WebSocket — server-pushed session_end & are_you_there ──
  const handleDepartureCheck = useCallback(async (targetName) => {
    if (handlingDepartureRef.current || farewellPlayingRef.current) return;
    handlingDepartureRef.current = true;
    const name = targetName || localName || session?.user_name || 'there';
    const promptText = (name !== 'there' && name !== 'Guest' && name !== 'Unknown' && name !== '')
      ? `Are you there, ${name}?`
      : 'Are you there?';

    try { interruptSpeakingRef.current && interruptSpeakingRef.current(); } catch (_) { }

    // Show the prompt in the chat UI too
    addMessage(promptText, 'kiosk');
    await speakAndWait(promptText);

    // Listen for up to 9 seconds for a response
    const heardAnswer = await captureUtteranceText(9000);

    // Face is genuinely BACK only if detection says ACTIVE — DEPARTING means
    // they are STILL gone (camera hasn't seen them yet). DWELLING/RECOGNIZING
    // means someone stepped in front but we don't yet know who — also count that.
    const faceBack = detStateRef.current === 'ACTIVE'
      || detStateRef.current === 'DWELLING'
      || detStateRef.current === 'RECOGNIZING';

    // Require an EXPLICIT confirmation word — do NOT treat random noise / empty
    // transcription as "yes". Background noise often produces short garbage text
    // (1-2 chars) which was incorrectly triggering "Glad you're still here"
    // even when the visitor had already left.
    const heardYes = /\b(yes|yeah|yep|yup|here|i'm here|im here|i am here|present|hi|hello|hey|stay|i am|nova|what)\b/i.test(heardAnswer || '');

    if (faceBack || heardYes) {
      handlingDepartureRef.current = false;
      // Only say "glad you're still here" if the face is ACTUALLY back, not
      // just because we heard something ambiguous.
      if (faceBack) {
        const gladText = "Great! Glad you're still here.";
        addMessage(gladText, 'kiosk');
        speak(gladText);
      }
      startListening();
    } else {
      // Face is still gone AND no clear verbal response — say goodbye
      const farewellText = (name !== 'there' && name !== 'Guest' && name !== 'Unknown' && name !== '')
        ? `Goodbye, ${name}! Have a wonderful day.`
        : 'Goodbye! Have a wonderful day.';
      addMessage(farewellText, 'kiosk');
      farewellPlayingRef.current = true;
      speak(farewellText, null, () => { farewellPlayingRef.current = false; });
      fetch(BACKEND + '/session/end?session_id=' + (session?.session_id || ''), { method: 'POST' }).catch(() => { });
      window.dispatchEvent(new CustomEvent('vrk-session-ended', { detail: { farewell: farewellText, userName: localName || session?.user_name } }));
      handlingDepartureRef.current = false;
    }
  }, [session, localName, speak, speakAndWait, captureUtteranceText, addMessage, startListening]);


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
            if (!farewellPlayingRef.current) {
              try { interruptSpeakingRef.current && interruptSpeakingRef.current(); } catch (_) { }
              pendingUtteranceRef.current = null;
            }
            window.dispatchEvent(new CustomEvent('vrk-session-ended', { detail: { farewell: '', userName: localName || session?.user_name } }));
          } else if (msg.type === 'are_you_there') {
            handleDepartureCheck(msg.user_name);
          } else if (msg.type === 'session_update' && msg.user_name) {
            // Backend confirmed a name change (e.g. via _deterministic_route in /ask).
            // Sync localName so farewell + departure messages use the real name.
            const updatedName = msg.user_name;
            if (updatedName && updatedName !== 'Guest' && updatedName !== 'Unknown') {
              setLocalName(updatedName);
            }
          }
        } catch (_) { }
      };
      ws.onclose = () => { if (!dead) setTimeout(connect, 3000); };
    }
    connect();
    return () => { dead = true; ws?.close(); };
  }, [handleDepartureCheck]);

  const btnPrimary = { padding: '11px 24px', border: 'none', borderRadius: '8px', background: '#1a237e', color: '#fff', cursor: 'pointer', fontSize: '14px', fontWeight: '600' };

  /* ── ANIMATED NOVA CHARACTER ─────────────────────────────────────────── */
  const NovaCharacter = ({ st }) => (
    <svg className={`nova-svg nova-${st}`} viewBox="0 0 320 500"
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
      {privacyOpen && (
        <div onClick={e => e.target === e.currentTarget && setPrivacyOpen(false)}
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#fff', borderRadius: '20px', padding: '36px', width: '460px', maxHeight: '80vh', overflowY: 'auto', boxShadow: '0 24px 64px rgba(0,0,0,0.22)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '18px' }}>
              <div style={{ width: '44px', height: '44px', borderRadius: '50%', background: '#e8eaf6', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1a237e" strokeWidth="2">
                  <path d="M12 2 4 6v6c0 5 3.5 9 8 10 4.5-1 8-5 8-10V6l-8-4z" />
                  <path d="M9 12l2 2 4-4" />
                </svg>
              </div>
              <div>
                <div style={{ fontSize: '18px', fontWeight: '700', color: '#1a237e' }}>Your Privacy at this Kiosk</div>
                <div style={{ fontSize: '12px', color: '#999' }}>How Nova sees and remembers you</div>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {[
                {
                  icon: <path d="M23 7l-7 5 7 5V7zM1 5h15v14H1z" />,
                  title: 'The camera is only used to greet you',
                  body: 'The kiosk camera looks for a face so Nova knows a visitor has arrived and can recognise returning visitors. It is not recorded or streamed anywhere.'
                },
                {
                  icon: <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z" />,
                  title: 'Face data is saved only if you say yes',
                  body: 'When Nova asks for your name, she also asks — out loud — whether you\'d like to be remembered for next time. Say yes and your name and face are stored so Nova can greet you by name next time. Say no, or continue as a guest, and nothing is saved.'
                },
                {
                  icon: <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />,
                  title: 'Conversations are used only to help you',
                  body: 'What you say is used to answer your questions during this visit and briefly shown on screen. It isn\'t used for advertising or shared outside the institute.'
                },
                {
                  icon: <path d="M12 8v4l3 3M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z" />,
                  title: 'Your session ends automatically',
                  body: 'After you say goodbye or step away, the session closes and live conversation data is cleared from the screen.'
                },
              ].map((item, i) => (
                <div key={i} style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#5c6bc0" strokeWidth="2" style={{ flexShrink: 0, marginTop: '2px' }}>
                    {item.icon}
                  </svg>
                  <div>
                    <div style={{ fontSize: '13.5px', fontWeight: '700', color: '#333' }}>{item.title}</div>
                    <div style={{ fontSize: '12.5px', color: '#777', lineHeight: '1.55', marginTop: '2px' }}>{item.body}</div>
                  </div>
                </div>
              ))}
            </div>

            <p style={{ fontSize: '11.5px', color: '#aaa', marginTop: '18px', lineHeight: '1.6' }}>
              Questions about your data? Speak to a staff member at the Admin Block.
            </p>

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px' }}>
              <button onClick={() => setPrivacyOpen(false)} style={btnPrimary}>Got it</button>
            </div>
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
          <img src="/rnslogo.png" alt="RNSIT"
            style={{ height: '40px', width: '40px', objectFit: 'contain' }} />
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
        </div>
      </header>

      {/* ── MAIN BODY ── */}
      <div style={{ flex: '1 1 0', display: 'flex', overflow: 'hidden' }}>

        {/* ══════════ LEFT: ANIMATED Nova CHARACTER ══════════ */}
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

          {/* ── happy-moment sparkle burst: first-time-visitor greeting only ── */}
          {celebrate && (
            <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 2 }}>
              {['10%', '25%', '75%', '88%', '45%', '60%'].map((left, i) => (
                <span key={i} className="sparkle-burst" style={{
                  position: 'absolute', left, top: `${30 + (i % 3) * 12}%`,
                  fontSize: `${14 + (i % 3) * 6}px`, animationDelay: `${i * 0.12}s`,
                }}>✨</span>
              ))}
            </div>
          )}

          {/* ── Nova SVG character ── */}
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', position: 'relative', zIndex: 1 }}>
            <NovaCharacter st={status} />
          </div>

          {/* ── Name + status badge ── */}
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px',
            zIndex: 1, marginTop: '8px'
          }}>
            <div style={{ fontSize: '20px', fontWeight: '800', color: '#1a237e', letterSpacing: '0.3px' }}>Nova</div>
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
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px', padding: '30px 12px', textAlign: 'center' }}>
                <img src="/rnslogo.png" onError={e => { e.currentTarget.style.display = 'none'; }} alt="RNSIT Logo"
                  style={{ width: '120px', height: 'auto', objectFit: 'contain', opacity: 0.95, filter: 'drop-shadow(0 10px 20px rgba(26,35,126,0.18))' }} />
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#9fa8da' }}>Your conversation with Nova will appear here</div>
                <div style={{ fontSize: '12px', color: '#c5cae9' }}>Just speak — she&apos;s ready</div>
              </div>
            )}

            {messages.map((msg, i) => {
              const isNova = msg.speaker === 'kiosk';
              const prevSame = i > 0 && messages[i - 1].speaker === msg.speaker;
              return (
                <div key={i} style={{
                  display: 'flex', flexDirection: 'column',
                  alignItems: isNova ? 'flex-start' : 'flex-end',
                  marginTop: prevSame ? '2px' : '8px'
                }}>
                  {!prevSame && (
                    <span style={{
                      fontSize: '10px', color: '#bbb', marginBottom: '2px',
                      paddingLeft: isNova ? '6px' : 0, paddingRight: !isNova ? '6px' : 0, fontWeight: '600'
                    }}>
                      {isNova ? 'Nova' : visitorName}
                    </span>
                  )}
                  <div className="msg-in" style={{
                    maxWidth: '88%', padding: '8px 12px',
                    borderRadius: isNova
                      ? (prevSame ? '4px 14px 14px 14px' : '14px 14px 14px 4px')
                      : (prevSame ? '14px 4px 14px 14px' : '14px 14px 4px 14px'),
                    background: isNova ? '#ffffff' : '#1a237e',
                    color: isNova ? '#1a1a1a' : '#ffffff',
                    fontSize: '13.5px', lineHeight: '1.5',
                    boxShadow: isNova ? '0 1px 3px rgba(0,0,0,0.08)' : '0 1px 4px rgba(26,35,126,0.25)',
                    wordBreak: 'break-word'
                  }}>
                    {msg.text}
                    <span style={{ fontSize: '9px', color: isNova ? '#ccc' : 'rgba(255,255,255,0.5)', marginLeft: '6px', float: 'right', marginTop: '3px', whiteSpace: 'nowrap' }}>
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
            {/* Hands-free voice prompt hints (0% clicking required) */}
            {status === 'ready' && !processingHint && !liveText &&
              messages.length > 0 && messages[messages.length - 1].speaker === 'kiosk' && (
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', paddingLeft: '4px', marginTop: '4px' }}>
                  <div style={{
                    padding: '6px 14px', background: '#eef2ff', border: '1px solid #c7cbe8',
                    borderRadius: '16px', fontSize: '12px', fontWeight: '600', color: '#3c4370'
                  }}>
                    💬 Try saying: "Ask something else"
                  </div>
                  <div style={{
                    padding: '6px 14px', background: '#eef2ff', border: '1px solid #c7cbe8',
                    borderRadius: '16px', fontSize: '12px', fontWeight: '600', color: '#3c4370'
                  }}>
                    💬 Or say: "That's all, thanks"
                  </div>
                </div>
              )}

            {processingHint && !liveText && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                <div style={{ fontSize: '11px', color: '#bbb', marginBottom: '4px', paddingLeft: '4px', fontWeight: '500' }}>
                  RNSIT Kiosk &nbsp;·&nbsp; thinking
                </div>
                <div style={{
                  maxWidth: '60%', padding: '13px 18px', borderRadius: '4px 18px 18px 18px',
                  background: '#f3f2fb', color: '#6a6f8c', fontSize: '15.5px', fontStyle: 'italic',
                  lineHeight: '1.6', border: '1.5px dashed #d8d6ea'
                }}>
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
            <span style={{ fontSize: '10px', color: '#ddd', marginLeft: 'auto' }}>RNSIT · Nova AI</span>
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

        /* ── name-flow listening pulse (small dot in the voice-name card) ── */
        .name-flow-pulse { animation: nfPulse 1.3s ease-in-out infinite; }
        @keyframes nfPulse { 0%,100%{box-shadow:0 0 0 0 rgba(67,160,71,0.45)} 50%{box-shadow:0 0 0 6px rgba(67,160,71,0)} }

        /* ── happy-moment sparkle burst (first-visit greeting only) ── */
        .sparkle-burst { animation: sparkleBurst 1.8s ease-out both; }
        @keyframes sparkleBurst {
          0% { opacity:0; transform: translateY(10px) scale(0.5) rotate(0deg); }
          25% { opacity:1; }
          100% { opacity:0; transform: translateY(-60px) scale(1.15) rotate(25deg); }
        }

        /* ══════════════════════════════
             Nova CHARACTER ANIMATIONS
        ══════════════════════════════ */

        /* BODY — gentle breathing (always on) */
        .nova-svg .body-grp { animation: charBreathe 5s ease-in-out infinite; }
        @keyframes charBreathe { 0%,100%{transform:scaleY(1)} 50%{transform:scaleY(1.016)} }

        /* HEAD — base: idle micro-float */
        .nova-svg .head-grp { animation: idleFloat 6s ease-in-out infinite; }
        @keyframes idleFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }

        /* ── LISTENING ── */
        .nova-listening .head-grp { animation: listenLean 0.7s ease-out forwards, idleFloat 0s; }
        @keyframes listenLean { to{transform:translateX(10px) rotate(6deg)} }

        /* pulse ring */
        .listen-r1 { animation:lRing 2s ease-out infinite; }
        .listen-r2 { animation:lRing 2s 0.7s ease-out infinite; }
        @keyframes lRing { 0%{r:92;opacity:0.55} 100%{r:140;opacity:0} }

        /* ── PROCESSING ── */
        .nova-processing .head-grp { animation: thinkTilt 0.6s ease-out forwards, idleFloat 0s; }
        @keyframes thinkTilt { to{transform:translateX(-12px) rotate(-7deg)} }

        /* thinking arm rise */
        .nova-processing .arm-think { animation: armRise 0.6s ease-out both; transform-origin:265px 228px; }
        @keyframes armRise { from{transform:translateY(30px);opacity:0} to{transform:none;opacity:1} }

        /* thinking bubble float */
        .tbub { animation:tBubFloat 1.6s ease-in-out infinite; }
        @keyframes tBubFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }

        /* furrowed brow when thinking */
        .brow-think { animation: browFurrow 0.5s ease-out forwards; transform-origin:133px 86px; }
        @keyframes browFurrow { to{transform:translateY(4px)} }

        /* ── SPEAKING ── */
        .nova-speaking .head-grp { animation: headBob 0.55s ease-in-out infinite; }
        @keyframes headBob { 0%,100%{transform:translateY(0) rotate(0)} 30%{transform:translateY(-5px) rotate(1.5deg)} 70%{transform:translateY(2px) rotate(-1deg)} }

        /* mouth alternates: a visible ↔ b visible */
        .nova-speaking .mouth-a { animation: mA 0.38s ease-in-out infinite; }
        .nova-speaking .mouth-b { animation: mB 0.38s ease-in-out infinite; }
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


// ── Nova's avatar: used in message rows and header ───────────────────────
const NovaAvatar = ({ size = 38, speaking = false }) => (
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