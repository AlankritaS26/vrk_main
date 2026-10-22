import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createKioskMic, float32ToInt16 } from './kioskMic';
import { normalizeSpelledName, resolveSingleLetter, isSpellingSkipOrGuest } from './utils/spellingNormalizer';
import Nova3DAvatar from './Nova3DAvatar';

const BACKEND = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:8001';

export const CONV_STATE = {
  IDLE: 'IDLE',
  GREETING: 'GREETING',
  ASKING_NAME: 'ASKING_NAME',
  CAPTURING_NAME: 'CAPTURING_NAME',
  ASKING_SPELLING: 'ASKING_SPELLING',
  CAPTURING_SPELLING: 'CAPTURING_SPELLING',
  CONFIRMING_SPELLING: 'CONFIRMING_SPELLING',
  RETRY_SPELLING: 'RETRY_SPELLING',
  CONFIRMED: 'CONFIRMED',
  GUEST: 'GUEST',
  LISTENING: 'LISTENING',
  PROCESSING: 'PROCESSING',
  SPEAKING: 'SPEAKING',
  INTERRUPTED: 'INTERRUPTED',
  GOODBYE: 'GOODBYE'
};
// Splits an answer into the same TTS sentence/clause chunks speakStream()
// plays, sequentially. Pulled out to module scope and shared with
// sendToBackend's "first chunk" prefetch below — they used to each keep
// their own copy of this splitting logic, and only speakStream's copy had
// the "merge a too-short first fragment into the next one" step. That
// meant a title like "Dr." (a complete "sentence" to the [.!?] splitter)
// got prefetched and synthesized on its own by the first-chunk request —
// a wasted ~2s Kokoro call for 3 characters, e.g. "Dr. Ramesh Babu H S is
// the current Principal..." — while speakStream's OWN sentences[0] (after
// its merge step) was actually "Dr. Ramesh Babu H S is the current
// Principal of RNSIT. Contact: ...", the mismatch also meaning the
// prefetched "Dr."-only clip did not match what got played for chunk 0.
// One shared function makes both places agree by construction.
function buildTtsSentenceChunks(text) {
  const raw = (text.match(/[^.!?]+[.!?]+["']?\s*|[^.!?]+$/g) || [text])
    .map(s => s.trim()).filter(Boolean);

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

  // Merge a too-short first fragment (an abbreviation like "Dr." caught by
  // the sentence-boundary regex, not a real standalone clause) into the
  // next chunk so it's never synthesized/spoken on its own.
  if (sentences.length > 1 && sentences[0].length < 25) {
    sentences[1] = sentences[0] + ' ' + sentences[1];
    sentences.shift();
  }

  return sentences;
}

export default function WelcomeScreen({ session, messages, setMessages, askingName, detState, doubleBlink, blink }) {
  const scrollRef = useRef(null);
  const escalationScrollRef = useRef(null);
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
  const speakStreamRef = useRef(null);
  const ttsGainRef = useRef(null);               // shared gain node — lets us duck/restore TTS volume smoothly
  const lastAnswerRef = useRef('');              // stores the most recent full answer text for resume-on-interrupt
  const wasInterruptedRef = useRef(false);       // true if TTS was barged-in before it finished — triggers "want to continue?" offer
  const ttsAnalyserRef = useRef(null);           // live TTS level for the 3D avatar mouth

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
  const [convState, setConvState] = useState(CONV_STATE.IDLE);
  const convStateRef = useRef(CONV_STATE.IDLE);
  useEffect(() => { convStateRef.current = convState; }, [convState]);
  const sttModeRef = useRef('normal');
  const candidateNameRef = useRef('');
  const lastQuestionHadAckRef = useRef(false);
  const nameFlowIdRef = useRef(0);          // bumped to invalidate an in-flight run

  const [localName, setLocalName] = useState('');
  const visitorName = localName || (session?.user_name && session.user_name !== 'Unknown' ? session.user_name : 'Guest');
  const isReturning = session?.is_returning || false;
  const visitCount = session?.visit_count || 1;

  // ── QR Companion & Escalation States ──
  const [companionToken, setCompanionToken] = useState(null);
  const [companionUrl, setCompanionUrl] = useState('');
  const [showCompanionModal, setShowCompanionModal] = useState(false);
  const [escalationState, setEscalationState] = useState(null); // null | 'pending' | 'connected' | 'timeout'
  const [escalationMsg, setEscalationMsg] = useState('');
  const [escalationMessages, setEscalationMessages] = useState([]);
  const escalationStateRef = useRef(null);
  useEffect(() => { escalationStateRef.current = escalationState; }, [escalationState]);

  useEffect(() => {
    if (!session?.session_id) {
      setCompanionToken(null);
      setCompanionUrl('');
      return;
    }
    fetch(BACKEND + '/companion/token', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        if (d?.token) {
          setCompanionToken(d.token);
          setCompanionUrl(d.url);
        }
      })
      .catch(() => {});
  }, [session?.session_id]);

  const handleRequestEscalation = async () => {
    try {
      setEscalationState('pending');
      setEscalationMessages([]);
      setEscalationMsg('Connecting you to front desk staff…');
      await fetch(BACKEND + '/escalation/request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: 'user_request' }),
      });
    } catch (err) {
      console.error('[ESCALATION] Request failed:', err);
    }
  };

  const handleCancelEscalation = async () => {
    setEscalationState(null);
    const sid = sessionRef.current?.session_id;
    if (sid) {
      try {
        await fetch(`${BACKEND}/escalation/cancel/${sid}`, { method: 'POST' });
      } catch (e) { /* silent */ }
    }
  };

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
    const panel = escalationScrollRef.current;
    if (!panel) return undefined;
    const frame = requestAnimationFrame(() => {
      panel.scrollTo({ top: panel.scrollHeight, behavior: 'smooth' });
    });
    return () => cancelAnimationFrame(frame);
  }, [escalationMessages]);

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
    if (!text || text === '__BLINK__' || /👁|\[Blinked/i.test(text)) return;
    text = cleanText(text);
    setMessages(prev => [...prev, {
      text, speaker,
      timestamp: new Date().toLocaleTimeString()
    }]);
  }, [setMessages]);

  const playEscalationAudio = useCallback(async (payload) => {
    if (!payload?.audio) return;
    try {
      interruptSpeakingRef.current?.();
      if (!playCtxRef.current) {
        playCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = playCtxRef.current;
      if (ctx.state === 'suspended') await ctx.resume();
      const binary = atob(payload.audio);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      let buffer;
      if (payload.mime_type?.startsWith('audio/pcm')) {
        const samples = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
        buffer = ctx.createBuffer(1, samples.length, 16000);
        const channel = buffer.getChannelData(0);
        for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;
      } else {
        buffer = await ctx.decodeAudioData(bytes.buffer);
      }
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const startAt = Math.max(ctx.currentTime, playCursorRef.current);
      source.start(startAt);
      playCursorRef.current = startAt + buffer.duration;
      activeNodesRef.current.push(source);
      source.onended = () => {
        activeNodesRef.current = activeNodesRef.current.filter(node => node !== source);
      };
    } catch (error) {
      console.warn('[ESC] Audio playback failed:', error);
      if (payload.speaker === 'staff' && payload.text) {
        speakStreamRef.current?.(payload.text);
      }
    }
  }, []);

  // Creates a new chat bubble for `speaker` and returns a function that
  // appends sentence-by-sentence text into it — keeps the on-screen text
  // in sync with what the TTS is actually reading aloud.
  const startProgressiveMessage = useCallback((speaker) => {
    const msgId = Date.now() + '-' + Math.random().toString(36).slice(2, 8);
    setMessages(prev => [...prev, {
      _id: msgId,
      text: '',
      speaker,
      timestamp: new Date().toLocaleTimeString()
    }]);
    return (sentence) => {
      if (!sentence) return;
      const cleaned = cleanText(sentence);
      setMessages(prev => prev.map(m =>
        m._id === msgId
          ? { ...m, text: m.text ? m.text + ' ' + cleaned : cleaned }
          : m
      ));
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
      const tSttStart = performance.now();
      const currentMode = sttModeRef.current || 'normal';
      const i16 = float32ToInt16(float32Audio);
      const response = await fetch(BACKEND + '/stt/pcm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'X-STT-Mode': currentMode,
          'X-STT-Candidate': candidateNameRef.current || ''
        },
        body: i16.buffer
      });

      const result = await response.json();
      const tSttEnd = performance.now();
      console.log(`[LATENCY-FLOW] STT_RESULT_RECEIVED in ${(tSttEnd - tSttStart).toFixed(0)}ms (mode: ${currentMode}, conf: ${result.confidence ?? 0}, whisper: ${result.latency_ms ?? 0}ms)`, result);
      const heard = (result.text || '').trim();

      // If a specific conversation prompt (e.g. name prompt, Yes/No confirm) is waiting for speech:
      if (activePromptResolverRef.current) {
        if (heard && heard.length > 0) {
          const resolver = activePromptResolverRef.current;
          activePromptResolverRef.current = null;
          if (isMounted.current) setLiveText(heard);
          statusRef.current = 'ready';
          setStatus('ready');
          resolver(heard, result);
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
        wasInterruptedRef.current = false;   // visitor spoke something — clear interrupt flag
        if (isMounted.current) setLiveText(heard);
        if (escalationStateRef.current === 'connected') {
          const sid = session?.session_id;
          if (sid) {
            fetch(BACKEND + '/escalation/visitor-audio/' + sid, {
              method: 'POST',
              headers: { 'Content-Type': 'application/octet-stream' },
              body: i16.buffer
            }).catch(error => console.warn('[ESC] Visitor audio relay failed:', error));
            fetch(BACKEND + '/escalation/message', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ session_id: sid, text: heard }),
            }).then(async response => {
              if (!response.ok) throw new Error(await response.text());
              const data = await response.json();
              if (data.message) {
                setEscalationMessages(prev => prev.some(m =>
                  m.timestamp === data.message.timestamp && m.text === data.message.text
                ) ? prev : [...prev, data.message].slice(-50));
              }
            }).catch(err => {
              console.error('[ESCALATION] Spoken message failed:', err);
              setEscalationMsg('Your message could not be delivered. Please speak again.');
            });
          }
          setLiveText('');
          setStatus('ready');
        } else {
          sendToBackend(heard);
        }
      } else {
        // Empty/too-short STT result after a barge-in interruption:
        // offer to resume the answer rather than silently going to 'ready'.
        if (wasInterruptedRef.current && lastAnswerRef.current) {
          wasInterruptedRef.current = false;
          const continuePrompt = 'It seems like you wanted to say something — would you like me to continue with the full answer?';
          addMessage(continuePrompt, 'kiosk');
          speak(continuePrompt);
          // The visitor can then say "yes" / "continue" which will be caught
          // by the resume intent check in sendToBackend on their next utterance.
          // We pre-set wasInterruptedRef so the next "yes" also works:
          wasInterruptedRef.current = true;
        } else {
          setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
        }
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
    window.__novaTtsActive = false;         // kill lipsync immediately on barge-in
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
            // Duck volume gently on sound onset — do NOT kill TTS audio immediately.
            // If it's a misfire (echo/noise spike), onMisfire will restore volume.
            // If it's real speech, onSpeechEnd below will hard-stop TTS.
            duckSpeaking();
            return;
          }
          isListening.current = true;
          setListening(true);
          setLiveText('');
          setStatus('listening');
          if (streamRef.current) startWaveform(streamRef.current);
        },
        onSpeechEnd: (audio) => {
          if (isSpeaking.current) {
            if (lastAnswerRef.current && !farewellPlayingRef.current && !greetingPlayingRef.current) {
              wasInterruptedRef.current = true;
            }
            interruptSpeaking();
          }
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
  const speakStream = useCallback(async (text, { onStart, onSentence, onDone, initialClipPromise, timingBase } = {}) => {
    // CRITICAL: stop any audio still playing from a PREVIOUS speak() call
    if (isSpeaking.current) interruptSpeaking();

    window.speechSynthesis.cancel();
    const myId = Symbol('speak');
    activeSpeakIdRef.current = myId;
    isSpeaking.current = true;

    let hasStartedAvatar = false;
    let firstAudioPlayStart = 0;

    const fireStart = () => {
      if (hasStartedAvatar) return;
      hasStartedAvatar = true;
      window.__novaTtsActive = true;   // tell the avatar animation loop that lipsync is live
      setStatus('speaking');    // avatar flips to "speaking" ONLY when first audio buffer starts
      if (onStart) { onStart(); onStart = null; }
    };

    const finish = () => {
      if (activeSpeakIdRef.current !== myId) return;
      window.__novaTtsActive = false;  // lipsync off — mouth returns to rest
      isSpeaking.current = false;
      if (isMounted.current) startListening();
      setStatus(awaitingAnswerRef.current ? 'processing' : 'ready');
      if (onDone) { try { onDone(); } catch (e) { } onDone = null; }
    };

    // Web Audio setup
    if (!playCtxRef.current) {
      playCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    const pctx = playCtxRef.current;
    if (pctx.state === 'suspended') { try { await pctx.resume(); } catch (e) { } }

    if (!ttsGainRef.current) {
      ttsGainRef.current = pctx.createGain();
      ttsGainRef.current.connect(pctx.destination);
    }
    if (!ttsAnalyserRef.current) {
      ttsAnalyserRef.current = pctx.createAnalyser();
      ttsAnalyserRef.current.fftSize = 1024;          // more frequency bins → smoother envelope
      ttsAnalyserRef.current.smoothingTimeConstant = 0.65;
      // Wiring: source nodes → analyser → gain → destination
      // (source nodes connect to analyser, not directly to gain)
      ttsAnalyserRef.current.connect(ttsGainRef.current);
      window.__novaTtsAnalyser = ttsAnalyserRef.current;
    }
    ttsGainRef.current.gain.cancelScheduledValues(pctx.currentTime);
    ttsGainRef.current.gain.setValueAtTime(1, pctx.currentTime);

    // Sentence splitting with natural chunking
    const raw = (text.match(/[^.!?]+[.!?]+["']?\s*|[^.!?]+$/g) || [text])
      .map(s => s.trim()).filter(Boolean);

    const sentences = [];
    if (raw.length) {
      let first = raw[0];
      if (first.length > 55) {
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
        if (buf.length >= 80) { sentences.push(buf); buf = ''; }
      }
      if (buf) sentences.push(buf);
    }

    if (sentences.length > 1 && sentences[0].length < 20) {
      sentences[1] = sentences[0] + ' ' + sentences[1];
      sentences.shift();
    }

    console.log('TTS QUEUE:', sentences);

    // Fallback: browser speech synthesis
    const browserSpeak = async () => {
      fireStart();
      for (let i = 0; i < sentences.length; i++) {
        if (activeSpeakIdRef.current !== myId) break;
        if (onSentence) { try { onSentence(sentences[i], i); } catch (e) { } }
        await new Promise((resolveEnd) => {
          const utter = new SpeechSynthesisUtterance(sentences[i]);
          utter.lang = 'en-US';
          utter.rate = 1.05;
          utter.volume = 1;
          utter.onstart = () => fireStart();
          utter.onend = () => resolveEnd();
          utter.onerror = () => resolveEnd();
          window.speechSynthesis.speak(utter);
        });
      }
      finish();
    };

    if (pctx.state === 'suspended') {
      console.warn('[TTS] AudioContext suspended — using browser voice.');
      await browserSpeak();
      return;
    }

    // Helper: fetch base64 AND pre-decode into AudioBuffer in parallel
    const fetchAndDecodeClip = async (sentenceText) => {
      if (!sentenceText) return null;
      const tReq = performance.now();
      try {
        const b64 = await fetch(BACKEND + '/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: sentenceText })
        }).then(r => r.json()).then(d => d.audio || null).catch(() => null);

        const tAudioRecv = performance.now();
        if (!b64) return { buf: null, tReq, tAudioRecv, tDecoded: tAudioRecv };

        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let b = 0; b < bin.length; b++) bytes[b] = bin.charCodeAt(b);
        const buf = await pctx.decodeAudioData(bytes.buffer);
        const tDecoded = performance.now();
        return { buf, tReq, tAudioRecv, tDecoded };
        if (activeSpeakIdRef.current !== myId) return resolveStarted();  // interrupted while decoding
        const node = pctx.createBufferSource();
        node.buffer = buf;
        node.connect(ttsAnalyserRef.current || ttsGainRef.current);
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
        console.warn('[TTS] Fetch or decode error for sentence:', sentenceText, e);
        return { buf: null, tReq, tAudioRecv: performance.now(), tDecoded: performance.now() };
      }
    };

    // Helper: decode an already-in-flight initialClipPromise
    const decodeInitialClipPromise = async (promise) => {
      const tReq = performance.now();
      try {
        const b64 = await promise;
        const tAudioRecv = performance.now();
        if (!b64) return { buf: null, tReq, tAudioRecv, tDecoded: tAudioRecv };

        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let b = 0; b < bin.length; b++) bytes[b] = bin.charCodeAt(b);
        const buf = await pctx.decodeAudioData(bytes.buffer);
        const tDecoded = performance.now();
        return { buf, tReq, tAudioRecv, tDecoded };
      } catch (e) {
        return { buf: null, tReq, tAudioRecv: performance.now(), tDecoded: performance.now() };
      }
    };

    try {
      console.log('TTS START:', text);

      // Pre-pipeline: Chunk 0 is preloaded immediately; Chunk 1 prefetch starts while Chunk 0 plays
      let prefetchNextPromise = (initialClipPromise && sentences.length > 0)
        ? decodeInitialClipPromise(initialClipPromise)
        : (sentences.length > 0 ? fetchAndDecodeClip(sentences[0]) : null);

      for (let i = 0; i < sentences.length; i++) {
        if (activeSpeakIdRef.current !== myId) break;

        const currentSentence = sentences[i];
        if (onSentence) { try { onSentence(currentSentence, i); } catch (e) { } }

        // Trigger pre-fetch & pre-decode of chunk N+1 WHILE chunk N is prepared/played
        const nextPromise = (i + 1 < sentences.length) ? fetchAndDecodeClip(sentences[i + 1]) : null;
        const clipData = await prefetchNextPromise;
        prefetchNextPromise = nextPromise;

        if (activeSpeakIdRef.current !== myId) break;

        let playedSuccessfully = false;

        if (clipData && clipData.buf) {
          try {
            if (activeSpeakIdRef.current === myId) {
              const node = pctx.createBufferSource();
              node.buffer = clipData.buf;
              // Route through analyser so the avatar's lipsync can read the live audio level.
              // Chain: source → analyser → gain → destination
              node.connect(ttsAnalyserRef.current || ttsGainRef.current);
              activeNodesRef.current.push(node);

              await new Promise((resolveEnd) => {
                node.onended = () => {
                  activeNodesRef.current = activeNodesRef.current.filter((n) => n !== node);
                  resolveEnd();
                };

                const tPlayStart = performance.now();
                if (i === 0) {
                  firstAudioPlayStart = tPlayStart;
                  fireStart(); // Avatar flips to speaking EXACTLY when buffer starts playing!

                  const tAns = timingBase != null ? timingBase : clipData.tReq;
                  const ttsSynthNet = clipData.tAudioRecv - clipData.tReq;
                  const decodeMs = clipData.tDecoded - clipData.tAudioRecv;
                  const timeToFirstAudio = tPlayStart - tAns;

                  console.log(`[LATENCY-FLOW] ANSWER_RECEIVED: t=${tAns.toFixed(1)}ms`);
                  console.log(`[LATENCY-FLOW] TTS_REQUEST_START: t=${clipData.tReq.toFixed(1)}ms (+${(clipData.tReq - tAns).toFixed(1)}ms)`);
                  console.log(`[LATENCY-FLOW] TTS_AUDIO_RECEIVED: t=${clipData.tAudioRecv.toFixed(1)}ms (TTS synth+net: ${ttsSynthNet.toFixed(0)}ms)`);
                  console.log(`[LATENCY-FLOW] AUDIO_DECODED: t=${clipData.tDecoded.toFixed(1)}ms (decode: ${decodeMs.toFixed(0)}ms)`);
                  console.log(`[LATENCY-FLOW] AUDIO_PLAY_START: t=${tPlayStart.toFixed(1)}ms (time-to-first-audio: ${timeToFirstAudio.toFixed(0)}ms)`);
                }
                node.start(0);
              });
              playedSuccessfully = true;
            }
          } catch (playbackErr) {
            console.warn('[TTS] Web Audio playback error, falling back to browser voice:', playbackErr);
          }
        }

        // Fallback recovery if backend TTS returned null or decode failed
        if (!playedSuccessfully && activeSpeakIdRef.current === myId) {
          console.log('[TTS] Browser voice fallback for:', currentSentence);
          fireStart();
          await new Promise((resolveEnd) => {
            const utter = new SpeechSynthesisUtterance(currentSentence);
            utter.lang = 'en-US';
            utter.rate = 1.05;
            utter.volume = 1;
            utter.onend = () => resolveEnd();
            utter.onerror = () => resolveEnd();
            window.speechSynthesis.speak(utter);
          });
        }
      }

      if (activeSpeakIdRef.current !== myId) return;
      const tAudioPlayEnd = performance.now();
      if (firstAudioPlayStart > 0) {
        console.log(`[LATENCY-FLOW] AUDIO_PLAY_END: t=${tAudioPlayEnd.toFixed(1)}ms (Total playback: ${(tAudioPlayEnd - firstAudioPlayStart).toFixed(0)}ms)`);
      }
      finish();
    } catch (e) {
      if (activeSpeakIdRef.current !== myId) return;
      console.error('[TTS] Sequential queue error, falling back to browser voice:', e);
      await browserSpeak();
    }
  }, [startListening, interruptSpeaking]);
  speakStreamRef.current = speakStream;

  // Thin wrapper over speakStream for callers that don't need per-sentence
  // sync (ack bubble, farewell, greeting, error fallback, name flow) — same
  // (onStart, onDone) signature as before.
  const speak = useCallback((text, onStart, onDone) => (
    speakStream(text, { onStart, onDone })
  ), [speakStream]);

  // Promise-returning wrapper: resolves once THIS utterance has fully
  // finished playing. Pauses the mic during prompts to prevent speaker echo.
  // Safety: a 15-second hard timeout always resumes the mic even if TTS
  // hangs or SpeechSynthesis never fires onend (e.g. Kokoro slow to load,
  // browser voice unavailable). Without this the mic stays paused forever.
  const speakAndWait = useCallback((text, onStart) => (
    new Promise((resolve) => {
      let settled = false;
      const done = () => {
        if (settled) return;
        settled = true;
        if (isMounted.current && micRef.current) {
          micRef.current.resume();
        }
        resolve();
      };
      micRef.current?.pause();
      // Hard timeout — always resume listening even if TTS never finishes
      const safetyTimer = setTimeout(done, 15000);
      speak(text, onStart, () => {
        clearTimeout(safetyTimer);
        done();
      });
    })
  ), [speak]);

  // Legacy callback kept temporarily to avoid the merge artifact; the active flow uses the later callback below.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const sendToBackendLegacy = useCallback(async (text) => {
    if (!text) return;
    setLiveText('');
    setProcessingHint('');
    const sid = session?.session_id || 'guest';
    addMessage(text, 'user');

    if (escalationState && /\b(?:stop|cancel|never mind|nevermind|back|continue|dont connect|no staff)\b/i.test(text)) {
      setEscalationState(null);
    }

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

    // NOTE: We do NOT auto-accept single or multi-word blurts as a name change.
    // Names are only updated when the visitor uses an explicit phrase like
    // "my name is X", "call me X", "change my name to X", or goes through the
    // bareNameChange interactive flow below. This prevents random words like
    // "okay", "yes", "Akshay Dao" (said mid-conversation) from being silently
    // treated as a name change.

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

      const extracted = extractVisitorName(heardNewName);
      if (!extracted) {
        const cancelMsg = 'No problem. Let me know if you would like to change your name or ask a question.';
        addMessage(cancelMsg, 'kiosk');
        speak(cancelMsg);
        return;
      }

      addMessage(heardNewName, 'user');

      // ── Step 2: Confirm with voice OR double-blink ───────────────────────
      const confirmMsg = `Got it — should I call you ${extracted}? Say yes or blink twice to confirm, or say no to spell it out.`;
      addMessage(confirmMsg, 'kiosk');
      await speakAndWait(confirmMsg);
      const confirmed = await captureYesNo(25000);

      // Helper: apply the final name to DB + session
      const applyName = async (finalName) => {
        setLocalName(finalName);
        window.dispatchEvent(new CustomEvent('vrk_user_name_update', { detail: { userName: finalName } }));
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

      // STALE-ANSWER GUARD: only drop if sessions are distinct and neither is guest
      const liveSid = sessionRef.current?.session_id || 'guest';
      if (data.dropped || (sid !== 'guest' && liveSid !== 'guest' && liveSid !== sid)) {
        console.info('[sendToBackend] dropped stale answer for', sid);
        awaitingAnswerRef.current = false;
        isSpeaking.current = false;
        setProcessingHint('');
        setStatus('ready');
        return;
      }

      const answer = data.answer || 'Sorry, I do not have that information. Please visit the Admin Block.';
      const tAnswerReceived = performance.now();
      console.log(`[LATENCY] ANSWER_RECEIVED t=${tAnswerReceived.toFixed(1)}ms`, answer);

      // Extracts just the first chunk — this is all the first TTS request
      // needs to send; the rest is chunked+prefetched inside speakStream.
      // Uses the SAME buildTtsSentenceChunks() speakStream uses for its
      // own sentences[0], so this prefetch can never diverge from (or be
      // a too-short throwaway fragment ahead of) what actually gets played.
      const firstChunkText = (ans) => buildTtsSentenceChunks(ans)[0] || ans;

      // Store full answer so resume intent can replay it on interruption
      lastAnswerRef.current = answer;
      wasInterruptedRef.current = false;
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
  const _REJECTION_WORDS = /^(no|nope|nah|wrong|incorrect|change|not|different|cancel|stop|skip|guest|unknown|friend|none|null|undefined|yes|yeah|yep|yup|sure|ok|okay)$/i;

  const extractVisitorName = useCallback((raw) => {
    if (!raw) return '';
    // Disqualify any string that contains eye emoji, blink, yes, no
    if (/👁|\[Blinked|blink|twice/i.test(raw)) return '';
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
    if (words.length === 0 || words.length > 3) return '';
    // Only accept strictly alphabetic words
    if (!words.every(w => /^[a-zA-Z]+$/.test(w))) return '';
    const result = words.map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');
    if (words.length === 1 && _REJECTION_WORDS.test(words[0])) return '';
    return result;
  }, []);

  // ── Double Blink Listener for Yes/Confirm ──────────────────────────────
  const prevDoubleBlinkRef = useRef(0);
  // Latches a double-blink that fired while no prompt was active (e.g. while
  // Nova is speaking). captureYesNo consumes it instantly so the blink is never lost.
  const pendingBlinkRef = useRef(false);

  useEffect(() => {
    if (doubleBlink && doubleBlink !== prevDoubleBlinkRef.current) {
      prevDoubleBlinkRef.current = doubleBlink;
      console.log('[BLINK] Double blink detected!');
      if (activePromptResolverRef.current) {
        const resolver = activePromptResolverRef.current;
        if (resolver.isYesNo) {
          activePromptResolverRef.current = null;
          statusRef.current = 'ready';
          setStatus('ready');
          resolver(true);
        } else if (resolver.allowBlink) {
          activePromptResolverRef.current = null;
          statusRef.current = 'ready';
          setStatus('ready');
          resolver('__BLINK__');
        } else {
          console.log('[BLINK] Prompt is waiting for spoken name/text; latching blink for next prompt');
          pendingBlinkRef.current = true;
        }
      } else {
        console.log('[BLINK] No resolver active — latching blink for next prompt');
        pendingBlinkRef.current = true;
      }
    }
  }, [doubleBlink]);

  const wantsToGiveName = useCallback((text) => {
    if (!text) return false;
    return text === '__BLINK__'
      || /\b(yes|yeah|yep|yup|sure|ok|okay|why not|of course|certainly|definitely|i do|i would|i want|give name|give my name|my name|tell name|tell my name|provide name|share name|enter name|yes please|i will|blink|blinked)\b/i.test(text);
  }, []);

  const isGuestOption = useCallback((text) => {
    if (!text || text === '__BLINK__') return false;
    return /\b(guest|guest mode|continue as guest|as guest|no name|anonymous|just guest)\b/i.test(text);
  }, []);

  const isContinueOption = useCallback((text) => {
    if (!text || text === '__BLINK__') return false;
    return /\b(skip|dont want|neither|no thanks|continue|just continue|start|just start|proceed|dont give)\b/i.test(text);
  }, []);

  // ── Voice prompt capture helpers (uses single persistent mic) ──────────
  const captureUtteranceText = useCallback((timeoutMs = 25000, allowBlink = false) => {
    return new Promise((resolve) => {
      // Only consume latched blink if caller explicitly permits blinks
      if (allowBlink && pendingBlinkRef.current) {
        pendingBlinkRef.current = false;
        resolve('__BLINK__');
        return;
      }
      let timer = null;
      const resolver = (text) => {
        if (timer) clearTimeout(timer);
        resolve((text || '').trim());
      };
      resolver.allowBlink = allowBlink;

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
      // If a double-blink was latched, consume it immediately as affirmative
      if (pendingBlinkRef.current) {
        pendingBlinkRef.current = false;
        resolve(true);
        return;
      }
      let timer = null;
      const resolver = (rawText) => {
        if (timer) clearTimeout(timer);
        if (typeof rawText === 'boolean') {
          resolve(rawText);
          return;
        }
        const heard = (rawText || '').trim().toLowerCase();
        if (rawText === '__BLINK__' || /\b(yes|yeah|yep|yup|sure|ok|okay|please|correct|right|true|thats right|that is right|thats me|that is me|yes please|i am|it is|blink|blinked)\b/i.test(heard)) {
          resolve(true);
        } else if (/\b(no|nope|nah|wrong|incorrect|not right|not that|different|change)\b/i.test(heard) || /don.?t/i.test(heard)) {
          resolve(false);
        } else if (rawText && rawText.trim().length > 0) {
          resolve(rawText.trim());
        } else {
          resolve(null);
        }
      };
      resolver.isYesNo = true;

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
  }, []);

  // ── Unified Reusable Name Spelling & Confirmation State Machine ───────────
  // Requirements:
  //   Step 2: ALWAYS ask for spelling ("Thanks, [Name]. Could you spell your name for me, one letter at a time?")
  //   Step 3: Capture spelling — conversational letter-by-letter or natural full utterance (NATO, phonetics)
  //   Step 4: Spelling confirmation ("I have A-L-A-N-K-R-I-T-A. Is that correct?")
  //   Step 5: NEVER save unconfirmed names. Only save upon explicit confirmation.
  const runSpellingCaptureAndConfirm = useCallback(async (candidate, stillCurrent) => {
    candidateNameRef.current = candidate || '';
    setConvState(CONV_STATE.ASKING_SPELLING);
    sttModeRef.current = 'spelling';

    const cleanCandidate = (candidate || '').trim();
    const spellPrompt = (cleanCandidate && !['Friend', 'Guest', 'there', 'Unknown'].includes(cleanCandidate))
      ? `Thanks, ${cleanCandidate}. Could you please spell your name?`
      : 'Could you please spell your name?';
    const spellChatMsg = (cleanCandidate && !['Friend', 'Guest', 'there', 'Unknown'].includes(cleanCandidate))
      ? `Thanks, ${cleanCandidate}! Could you please spell your name?\n\n• 🗣️ Say the letters (e.g. "A L A N K R I T A")\n• 🗣️ Say "Guest" to skip`
      : 'Could you please spell your name?\n\n• 🗣️ Say the letters (e.g. "A L A N K R I T A")\n• 🗣️ Say "Guest" to skip';

    addMessage(spellChatMsg, 'kiosk');
    await speakAndWait(spellPrompt);
    if (!stillCurrent()) return;

    let accumulatedLetters = [];
    let retryCount = 0;

    while (stillCurrent() && retryCount < 3) {
      setConvState(CONV_STATE.CAPTURING_SPELLING);
      sttModeRef.current = 'spelling';
      const heardSpelling = await captureUtteranceText(18000);
      if (!stillCurrent()) return;

      // Visitor chose to skip or continue as guest
      if (isSpellingSkipOrGuest(heardSpelling)) {
        setConvState(CONV_STATE.GUEST);
        setLocalName('Guest');
        await submitVoiceName('Guest', false);
        const guestMsg = "That's okay. We can continue as Guest. How can I help you?";
        addMessage(guestMsg, 'kiosk');
        await speakAndWait(guestMsg);
        if (stillCurrent()) startListening();
        return;
      }

      if (!heardSpelling) {
        retryCount++;
        if (retryCount >= 3) break;
        const promptRep = "Please spell your name, like A L A N K R I T A. Or say Guest to continue.";
        addMessage(promptRep, 'kiosk');
        await speakAndWait(promptRep);
        continue;
      }

      addMessage(heardSpelling, 'user');

      // Check if visitor said "done" or "finished"
      const isDoneKeyword = /^(done|finish|finished|that.?s it|stop|end|complete|that.?s all|ok done)$/i.test(heardSpelling.trim());
      if (isDoneKeyword) {
        if (accumulatedLetters.length < 2) {
          const needMoreMsg = "Please tell me the letters in your name before saying done.";
          addMessage(needMoreMsg, 'kiosk');
          await speakAndWait(needMoreMsg);
          continue;
        }
      } else {
        // 1. Check if user spoke a full multi-letter sequence in one utterance (e.g. "A L A N K R I T A", "A, L, A...", "A-L-A...")
        const fullNorm = normalizeSpelledName(heardSpelling);
        if (fullNorm && fullNorm.letters && fullNorm.letters.length >= 2) {
          accumulatedLetters = fullNorm.letters;
          console.log('[NAME-STATE] Full utterance spelling captured in one text:', fullNorm);
        } else {
          // If cleanCandidate exists and user's utterance has all letters together (e.g. "ALANKRITA")
          const cleanLetters = heardSpelling.toUpperCase().replace(/[^A-Z]/g, '');
          if (cleanLetters.length >= 2) {
            accumulatedLetters = cleanLetters.split('');
            console.log('[NAME-STATE] Clean letter string captured:', accumulatedLetters);
          } else {
            // 2. Check for single letter conversational utterance
            const singleLetter = resolveSingleLetter(heardSpelling);
            if (singleLetter) {
              accumulatedLetters.push(singleLetter);
              const soFarDashed = accumulatedLetters.join('-');
              console.log(`[NAME-STATE] Captured letter: ${singleLetter}, so far: ${soFarDashed}`);

              // If candidateName was provided and we have collected all its letters:
              const expectedLen = cleanCandidate.replace(/[^a-zA-Z]/g, '').length;
              if (expectedLen > 1 && accumulatedLetters.length >= expectedLen) {
                // Collected all expected letters, move to confirmation
              } else {
                const nextPrompt = `${singleLetter}. Next letter?`;
                addMessage(`${singleLetter}. So far: ${soFarDashed}`, 'kiosk');
                await speakAndWait(nextPrompt);
                continue;
              }
            } else {
              // STT low confidence or unrecognised syllable
              const retryLetterMsg = "Sorry, I missed that. Could you please spell your name, like A L A N K R I T A?";
              addMessage(retryLetterMsg, 'kiosk');
              await speakAndWait(retryLetterMsg);
              continue;
            }
          }
        }
      }

      // Step 4: Spelling Confirmation
      const spelledDashed = accumulatedLetters.join('-');
      const normalizedFinalName = accumulatedLetters.join('').charAt(0).toUpperCase() + accumulatedLetters.join('').slice(1).toLowerCase();

      setConvState(CONV_STATE.CONFIRMING_SPELLING);
      sttModeRef.current = 'normal';
      const confirmSpokenMsg = `I have ${spelledDashed}. Is that correct?`;
      const confirmChatMsg = `I have ${spelledDashed}. Is that correct?\n\n• 🗣️ Say "Yes" or 👁️ Blink twice to confirm\n• 🗣️ Say "No" to retry`;
      addMessage(confirmChatMsg, 'kiosk');
      await speakAndWait(confirmSpokenMsg);
      if (!stillCurrent()) return;

      const confirmed = await captureYesNo(25000);
      if (!stillCurrent()) return;

      if (confirmed === true) {
        // YES or double-blink: SAVE confirmed name
        setConvState(CONV_STATE.CONFIRMED);
        setLocalName(normalizedFinalName);
        await submitVoiceName(normalizedFinalName, true);
        const meetMsg = `Lovely to meet you, ${normalizedFinalName}. How can I help you?`;
        addMessage(meetMsg, 'kiosk');
        await speakAndWait(meetMsg);
        if (stillCurrent()) startListening();
        return;
      } else if (confirmed === false || (typeof confirmed === 'string' && /\b(no|nope|wrong|change|not|different|retry)\b/i.test(confirmed))) {
        // NO: Do NOT save name! Prompt retry
        retryCount++;
        if (retryCount < 3) {
          setConvState(CONV_STATE.RETRY_SPELLING);
          const retryMsg = "No problem. Let's try that again. Please spell your name for me.";
          addMessage(retryMsg, 'kiosk');
          await speakAndWait(retryMsg);
          accumulatedLetters = [];
          continue;
        } else {
          break;
        }
      } else {
        // Unclear or timeout: NEVER save unconfirmed name
        break;
      }
    }

    // Retries exhausted or abandoned: Fall back to Guest mode
    setConvState(CONV_STATE.GUEST);
    setLocalName('Guest');
    await submitVoiceName('Guest', false);
    const guestFallbackMsg = "That's okay. We can continue as Guest. How can I help you?";
    addMessage(guestFallbackMsg, 'kiosk');
    await speakAndWait(guestFallbackMsg);
    if (stillCurrent()) startListening();
  }, [addMessage, speakAndWait, captureUtteranceText, captureYesNo, submitVoiceName, startListening]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const sendToBackend = useCallback(async (text) => {
    if (!text) return;
    setLiveText('');
    setProcessingHint('');
    const sid = session?.session_id || 'guest';
    const myReqSeq = ++requestSeqRef.current;   // this question's sequence number
    addMessage(text, 'user');

    // ── Resume-interrupted-answer intent ───────────────────────────────
    const resumeKeywords = /\b(continue|go on|full answer|complete|finish|what else|rest of|repeat that|say again|resume|give me the full|tell me more|carry on)\b/i.test(text.trim());
    const resumeYes = wasInterruptedRef.current && /^(yes|yeah|yep|sure|ok|okay|please|go ahead|sure please)[.!?]*$/i.test(text.trim());
    if ((resumeKeywords || resumeYes) && lastAnswerRef.current) {
      wasInterruptedRef.current = false;
      const storedAnswer = lastAnswerRef.current;
      addMessage(storedAnswer, 'kiosk');
      speakStream(storedAnswer, {
        onDone: () => { wasInterruptedRef.current = false; },
      });
      return;
    }

    // ── Mid-session name change or introduction ─────────────────────────
    const bareNameChange = /\b(?:change|update|reset|rename)\s+(?:my\s+|the\s+)?name\b/i.test(text)
      || /\b(?:i want to|can i|can you|please|how do i)\s+(?:change|update|reset|rename)\s+(?:my\s+|the\s+)?name\b/i.test(text);

    const hasInlineName = /\b(?:change|update|set|rename)\s+(?:my\s+|the\s+)?name\s+to\s+\w/i.test(text)
      || /\b(?:call me|my name is|actually my name is|its actually|it's actually|no my name is|i am called|this is)\s+\w/i.test(text);

    const isExplicitNameIntro = /\b(my name is|call me|i am|i'm|this is)\s+\w/i.test(text);

    if ((isExplicitNameIntro || hasInlineName) && !bareNameChange) {
      const candidateName = extractVisitorName(text);
      if (candidateName && candidateName.length >= 2 && candidateName.split(' ').length <= 3
          && !/^(yes|no|guest|skip|continue|ok|okay|bye|thanks|thank you|done|then|well|so|and|but|or|the|a|an)$/i.test(candidateName)) {
        // ALWAYS route through spelling capture before saving!
        await runSpellingCaptureAndConfirm(candidateName, () => isMounted.current);
        return;
      }
    }

    if (bareNameChange && !hasInlineName) {
      const promptChange = 'Sure! What should I change your name to?';
      addMessage(promptChange, 'kiosk');
      await speakAndWait(promptChange);
      const heardNewName = await captureUtteranceText(20000);
      if (!heardNewName) {
        const cancelMsg = 'No problem. Let me know if you would like to change your name or ask a question.';
        addMessage(cancelMsg, 'kiosk');
        speak(cancelMsg);
        return;
      }
      addMessage(heardNewName, 'user');
      const candidateName = extractVisitorName(heardNewName) || heardNewName.trim().replace(/[.!?]+$/, '');
      await runSpellingCaptureAndConfirm(candidateName, () => isMounted.current);
      return;
    }

    // ── Farewell / Goodbye ─────────────────────────────────────────────
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
      farewellPlayingRef.current = true;
      speak(farewell, null, () => { farewellPlayingRef.current = false; });
      fetch(BACKEND + '/session/end?session_id=' + sid, { method: 'POST' }).catch(() => { });
      window.dispatchEvent(new CustomEvent('vrk-session-ended', { detail: { farewell, userName: localName || session?.user_name } }));
      return;
    }

    awaitingAnswerRef.current = true;
    setConvState(CONV_STATE.PROCESSING);
    sttModeRef.current = 'normal';

    // ── Coordinated instant acknowledgment ────────────────────────────
    const acks = [
      'Sure, let me check that for you.',
      'Good question - one moment.',
      'Let me look that up for you.',
      'Of course, just a second.',
      'Right, let me find that.',
    ];
    const isInstantCmd = hasInlineName || bareNameChange ||
      /^(hi|hello|hey|good morning|good afternoon|good evening|bye|thank you|thanks)/i.test(text.trim());

    let ackTimer = null;

    if (!isInstantCmd && text.split(' ').length >= 3 && !lastQuestionHadAckRef.current) {
      ackTimer = setTimeout(() => {
        if (!awaitingAnswerRef.current) return;
        const ack = acks[Math.floor(Math.random() * acks.length)];
        lastQuestionHadAckRef.current = true;
        console.log('THINKING AUDIO START:', ack);
        speak(ack, () => setProcessingHint(ack), () => {
          console.log('THINKING AUDIO COMPLETED:', ack);
        });
      }, 750);
    } else {
      lastQuestionHadAckRef.current = false;
      setProcessingHint('Thinking...');
      statusRef.current = 'processing';
      setStatus('processing');
    }

    const tAskStart = performance.now();
    const askController = new AbortController();
    const askTimeout = setTimeout(() => askController.abort(), 35000);
    try {
      const [, askRes] = await Promise.all([
        fetch(BACKEND + '/message', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sid, text, speaker: 'user' })
        }),
        fetch(BACKEND + '/ask?question=' + encodeURIComponent(text) + '&session_id=' + encodeURIComponent(sid),
          { signal: askController.signal })
      ]);
      clearTimeout(askTimeout);
      if (ackTimer) clearTimeout(ackTimer);

      const data = await askRes.json();
      const tAnswerReceived = performance.now();
      const backendAnsLatency = tAnswerReceived - tAskStart;
      console.log(`[LATENCY-FLOW] ANSWER_RECEIVED: at t=${tAnswerReceived.toFixed(1)}ms (backend answer latency: ${backendAnsLatency.toFixed(0)}ms)`, data);

      if (data.dropped || myReqSeq !== requestSeqRef.current) {
        awaitingAnswerRef.current = false;
        isSpeaking.current = false;
        setProcessingHint('');
        setStatus('ready');
        return;
      }

      // If backend returned a name spelling prompt (e.g. from _deterministic_route):
      if (data.source === 'name_change_ask_spelling' || data.source === 'name_spelling_prompt') {
        awaitingAnswerRef.current = false;
        setProcessingHint('');
        const cand = data.candidate_name || extractVisitorName(text);
        await runSpellingCaptureAndConfirm(cand, () => isMounted.current);
        return;
      }

      const answer = data.answer || 'Sorry, I do not have that information. Please visit the Admin Block.';

      const firstChunkText = (ans) => {
        const raw = (ans.match(/[^.!?]+[.!?]+["']?\s*|[^.!?]+$/g) || [ans]).map(s => s.trim()).filter(Boolean);
        let first = raw.length ? raw[0] : ans;
        if (first.length > 55) {
          const cut = first.indexOf(',');
          if (cut > 15) first = first.slice(0, cut + 1);
        }
        return first;
      };

      lastAnswerRef.current = answer;
      wasInterruptedRef.current = false;

      fetch(BACKEND + '/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, text: answer, speaker: 'kiosk' })
      });

      setProcessingHint('');
      awaitingAnswerRef.current = false;

      // Render FULL answer text into the response box
      addMessage(answer, 'kiosk');

      // Start prefetching chunk 0 immediately — zero artificial wait!
      const tTtsReqStart = performance.now();
      console.log(`[LATENCY-FLOW] TTS_REQUEST_START: chunk 0 at t=${tTtsReqStart.toFixed(1)}ms`);
      const firstClipPromise = fetch(BACKEND + '/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: firstChunkText(answer) })
      }).then(r => r.json()).then(d => {
        const tTtsResponse = performance.now();
        console.log(`[LATENCY-FLOW] TTS_AUDIO_RECEIVED: at t=${tTtsResponse.toFixed(1)}ms (+${(tTtsResponse - tTtsReqStart).toFixed(0)}ms)`);
        return d.audio || null;
      }).catch(() => null);

      await speakStream(answer, { initialClipPromise: firstClipPromise, timingBase: tAnswerReceived });

    } catch (e) {
      clearTimeout(askTimeout);
      if (ackTimer) clearTimeout(ackTimer);
      awaitingAnswerRef.current = false;
      setProcessingHint('');
      console.error('[sendToBackend]', e);
      const fallback = e.name === 'AbortError'
        ? "I'm sorry, that's taking longer than expected. Please try asking again."
        : 'Sorry, something went wrong. Please try again.';
      addMessage(fallback, 'kiosk');
      speak(fallback);
    }
  }, [session, addMessage, speakStream, runSpellingCaptureAndConfirm, extractVisitorName, speakAndWait, captureUtteranceText, speak]);

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
        setConvState(CONV_STATE.ASKING_NAME);
        sttModeRef.current = 'name';

        // Step 1: Nova asks "Hi! May I know your name?"
        const askChatMsg = 'Welcome to RNS Institute of Technology! I am Nova, your digital receptionist.\n\nHi! May I know your name?\n\n• 🗣️ Speak your name\n• 🗣️ Say "Guest" to continue as Guest';
        const askSpokenMsg = 'Welcome to R N S Institute of Technology! I am Nova, your digital receptionist. Hi! May I know your name?';
        addMessage(askChatMsg, 'kiosk');
        await speakAndWait(askSpokenMsg);
        if (!stillCurrent()) return;

        setConvState(CONV_STATE.CAPTURING_NAME);
        sttModeRef.current = 'name';
        const heard = await captureUtteranceText(12000);
        if (!stillCurrent()) return;

        if (heard) {
          if (heard !== '__BLINK__') {
            addMessage(heard, 'user');
          }

          if (wantsToGiveName(heard)) {
            choseGiveName = true;
          } else if (isGuestOption(heard) || isContinueOption(heard)) {
            setConvState(CONV_STATE.GUEST);
            const guestMsg = 'Continuing as Guest! How may I assist you today?';
            addMessage(guestMsg, 'kiosk');
            setLocalName('Guest');
            await submitVoiceName('Guest', false);
            await speakAndWait(guestMsg);
            if (stillCurrent()) startListening();
            break;
          }

          // Check if user spoke a campus question directly (e.g. "Where is the admission office?")
          const cleanNameCandidate = extractVisitorName(heard);
          const isQuestion = heard.includes('?') || heard.split(' ').length >= 4 ||
            /\b(where|what|how|when|who|which|can|tell|fees|admission|hostel|placement|library|department|principal|hod)\b/i.test(heard);

          if (isQuestion && !cleanNameCandidate) {
            setConvState(CONV_STATE.GUEST);
            setLocalName('Guest');
            await submitVoiceName('Guest', false);
            sendToBackend(heard);
            break;
          }

          // Spoken candidate name: treat as candidate ONLY
          const candidate = cleanNameCandidate || heard.trim().replace(/[.!?]+$/, '');
          const isInvalidName = /^(yes|yeah|yep|no|nope|nah|guest|skip|continue|ok|okay|bye|thanks|thank you|friend|unknown)$/i.test(candidate);

          if (candidate && candidate.length >= 2 && !isInvalidName) {
            // Step 2, 3, 4: ALWAYS route through mandatory spelling capture & confirmation!
            await runSpellingCaptureAndConfirm(candidate, stillCurrent);
            break;
          } else {
            // Ambiguous response or rejection word: ask once more politely
            const askRetry = "I want to make sure I get your name right. Could you say it once more?";
            addMessage(askRetry, 'kiosk');
            await speakAndWait(askRetry);
            if (!stillCurrent()) return;

            sttModeRef.current = 'name';
            const retrySpoken = await captureUtteranceText(12000);
            if (retrySpoken) {
              addMessage(retrySpoken, 'user');
              if (isGuestOption(retrySpoken) || isContinueOption(retrySpoken)) {
                setConvState(CONV_STATE.GUEST);
                setLocalName('Guest');
                await submitVoiceName('Guest', false);
                const gMsg = "No problem! Continuing as Guest. How can I help you today?";
                addMessage(gMsg, 'kiosk');
                await speakAndWait(gMsg);
                if (stillCurrent()) startListening();
                break;
              }
              const retryCand = extractVisitorName(retrySpoken) || retrySpoken.trim().replace(/[.!?]+$/, '');
              if (retryCand && retryCand.length >= 2 && !/^(yes|no|guest|skip|continue|ok|okay)$/i.test(retryCand)) {
                await runSpellingCaptureAndConfirm(retryCand, stillCurrent);
                break;
              }
            }
            // Fall back to Guest
            setConvState(CONV_STATE.GUEST);
            setLocalName('Guest');
            await submitVoiceName('Guest', false);
            const guestFallback = "That's okay. We can continue as Guest. How can I help you?";
            addMessage(guestFallback, 'kiosk');
            await speakAndWait(guestFallback);
            if (stillCurrent()) startListening();
            break;
          }
        } else {
          // Timeout — check if anyone is still in front of the camera
          const stateNow = detStateRef.current;
          if (stateNow === 'IDLE' || stateNow === 'COOLDOWN') {
            // Nobody there — abort silently
            break;
          }
          setConvState(CONV_STATE.GUEST);
          const noAnsMsg = 'Continuing as Guest! How may I assist you today?';
          addMessage(noAnsMsg, 'kiosk');
          setLocalName('Guest');
          await submitVoiceName('Guest', false);
          await speakAndWait(noAnsMsg);
          if (stillCurrent()) startListening();
          break;
        }
      }
    } finally {
      flowRunningRef.current = false;
    }
  }, [session, isReturning, greeting, extractVisitorName, isGuestOption, isContinueOption, captureUtteranceText, submitVoiceName, speakAndWait, addMessage, sendToBackend, startListening, runSpellingCaptureAndConfirm]);

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
          } else if (msg.type === 'companion_qr') {
            if (msg.token) setCompanionToken(msg.token);
            if (msg.url) setCompanionUrl(msg.url);
          } else if (msg.type === 'escalation_pending' &&
                     msg.session_id === sessionRef.current?.session_id) {
            setEscalationState('pending');
            setEscalationMsg(msg.message || 'Connecting you to front desk staff…');
          } else if (msg.type === 'escalation_connected' &&
                     msg.session_id === sessionRef.current?.session_id) {
            setEscalationState('connected');
            setEscalationMsg(msg.message || 'A staff member has connected!');
          } else if (msg.type === 'escalation_message' &&
                     msg.session_id === sessionRef.current?.session_id) {
            setEscalationMessages(prev => [...prev, msg.message].slice(-50));
            if (msg.message?.speaker === 'staff' && msg.message.text && !msg.message.has_audio) {
              speakStreamRef.current?.(msg.message.text);
            }
          } else if (msg.type === 'escalation_audio' &&
                     msg.session_id === sessionRef.current?.session_id &&
                     msg.speaker === 'staff') {
            playEscalationAudio(msg);
          } else if (msg.type === 'escalation_timeout' &&
                     msg.session_id === sessionRef.current?.session_id) {
            setEscalationState('timeout');
            setEscalationMsg(msg.message || 'No staff available right now. Nova will continue helping you.');
            setTimeout(() => setEscalationState(null), 6000);
          } else if ((msg.type === 'escalation_cancelled' || msg.type === 'escalation_resolved') &&
                     msg.session_id === sessionRef.current?.session_id) {
            setEscalationState(null);
          }
        } catch (_) { }
      };
      ws.onclose = () => { if (!dead) setTimeout(connect, 3000); };
    }
    connect();
    return () => { dead = true; ws?.close(); };
  }, [handleDepartureCheck, playEscalationAudio]);

  const btnPrimary = { padding: '11px 24px', border: 'none', borderRadius: '8px', background: '#1a237e', color: '#fff', cursor: 'pointer', fontSize: '14px', fontWeight: '600' };

  /* ── 3D NOVA AVATAR ACTIVE (replaces legacy SVG) ── */


  const statusLabel = { ready: 'Ready', listening: 'Listening…', processing: 'Thinking…', speaking: 'Speaking…' }[status] || 'Ready';
  const statusColor = { ready: '#1a237e', listening: '#2e7d32', processing: '#6a1b9a', speaking: '#bf360c' }[status] || '#1a237e';
  const statusBg = { ready: '#e8eaf6', listening: '#e8f5e9', processing: '#f3e5f5', speaking: '#fff3e0' }[status] || '#e8eaf6';

  return (
    <div style={{
      height: '100vh', overflow: 'hidden', display: 'flex', flexDirection: 'column',
      fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif", background: '#dbeafe'
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
          background: {
            ready: '#dbeafe',
            listening: '#dbeafe',
            processing: '#dbeafe',
            speaking: '#dbeafe',
          }[status] || '#dbeafe', transition: 'background 0.8s ease', position: 'relative',
          overflow: 'hidden'
        }}>

          {/* subtle radial glow behind character */}
          <div style={{
            position: 'absolute', bottom: '60px', left: '50%', transform: 'translateX(-50%)',
            width: '320px', height: '320px', borderRadius: '50%',
            background: 'rgba(56,189,248,0.18)', filter: 'blur(40px)', pointerEvents: 'none'
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

          {/* ── Always-Visible Hands-Free QR Companion Card ── */}
          {companionToken && (
            <div style={{
              position: 'absolute', top: '16px', left: '16px', zIndex: 10,
              background: 'rgba(255, 255, 255, 0.92)', backdropFilter: 'blur(12px)',
              borderRadius: '16px', padding: '10px 14px', display: 'flex', alignItems: 'center', gap: '12px',
              boxShadow: '0 8px 24px rgba(26, 35, 126, 0.12)', border: '1.5px solid rgba(224, 228, 255, 0.95)',
              animation: 'fadeIn 0.5s ease'
            }}>
              <img
                src={`${BACKEND}/companion/qr/${companionToken}`}
                alt="Companion QR"
                style={{ width: '70px', height: '70px', borderRadius: '10px', border: '1px solid #c7cbe8', background: '#fff', padding: '3px', objectFit: 'contain' }}
              />
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <span style={{ fontSize: '13px' }}>📱</span>
                  <span style={{ fontSize: '11.5px', fontWeight: '800', color: '#1a237e', textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                    Phone Companion
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: '#1e293b', fontWeight: '700', marginTop: '3px' }}>
                  Scan with smartphone
                </div>
                <div style={{ fontSize: '10px', color: '#64748b', marginTop: '2px', lineHeight: '1.3', maxWidth: '140px' }}>
                  Take <strong>Visit Summary</strong> &amp; <strong>PDF Brochures</strong> with you
                </div>
              </div>
            </div>
          )}

          {/* ── Nova 3D Character ── */}
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', position: 'relative', zIndex: 1 }}>
            <Nova3DAvatar st={status} />
          </div>

          {/* ── Name + status badge ── */}
          <div style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px',
            zIndex: 1, marginTop: '8px'
          }}>
            <div style={{ fontSize: '22px', fontWeight: '800', color: '#ffffff', letterSpacing: '0.3px', textShadow: '0 2px 12px rgba(0,0,0,0.35)' }}>Nova</div>
            <div style={{ fontSize: '12px', color: 'rgba(219,234,254,0.82)', fontWeight: '600', letterSpacing: '0.5px' }}>RNSIT Digital Receptionist</div>
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
              background: 'rgba(6,18,54,0.52)', backdropFilter: 'blur(8px)',
              border: '1px solid rgba(125,211,252,0.32)', fontSize: '12px',
              color: 'rgba(224,242,254,0.88)', fontStyle: 'italic', textAlign: 'center',
              maxWidth: '280px', zIndex: 1
            }}>
              {hints[hintIndex]}
            </div>
          )}
        </div>

        {/* ══════════ RIGHT: COMPACT CHAT ══════════ */}
        <div style={{
          flex: 1, display: 'flex', flexDirection: 'column', background: 'rgba(219,234,254,0.72)',
          border: '1px solid rgba(255,255,255,0.62)', borderRadius: '28px', margin: '14px',
          boxShadow: '0 16px 36px rgba(30,64,175,0.14)', backdropFilter: 'blur(8px)',
          overflow: 'hidden'
        }}>

          {/* chat header */}
          <div style={{
            padding: '12px 18px', background: 'rgba(255,255,255,0.22)',
            borderBottom: '1px solid rgba(255,255,255,0.45)', flexShrink: 0,
            display: 'flex', alignItems: 'center', gap: '8px'
          }}>
            <div style={{
              width: '8px', height: '8px', borderRadius: '50%',
              background: { ready: '#43a047', listening: '#43a047', processing: '#7e57c2', speaking: '#e53935' }[status] || '#43a047',
              transition: 'background 0.3s', boxShadow: '0 0 0 3px rgba(67,160,71,0.15)'
            }} />
            <span style={{ fontSize: '13px', fontWeight: '800', color: '#1e3a8a' }}>Conversation</span>
            <span style={{ fontSize: '11px', color: '#64748b', marginLeft: 'auto' }}>
              {messages.length > 0 ? `${messages.length} message${messages.length > 1 ? 's' : ''}` : 'Just started'}
            </span>
          </div>

          {/* messages */}
          <div ref={scrollRef} style={{ flex: '1 1 0', overflowY: 'auto', padding: '14px 18px', display: 'flex', flexDirection: 'column', gap: '6px' }}>

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
                      fontSize: '10px', color: '#64748b', marginBottom: '4px',
                      paddingLeft: isNova ? '6px' : 0, paddingRight: !isNova ? '6px' : 0, fontWeight: '600'
                    }}>
                      {isNova ? 'Nova' : visitorName}
                    </span>
                  )}
                  <div className="msg-in" style={{
                    maxWidth: '86%', padding: '10px 14px',
                    borderRadius: isNova
                      ? (prevSame ? '4px 14px 14px 14px' : '14px 14px 14px 4px')
                      : (prevSame ? '14px 4px 14px 14px' : '14px 14px 4px 14px'),
                    background: isNova ? 'rgba(255,255,255,0.72)' : 'rgba(30,64,175,0.88)',
                    color: isNova ? '#1e293b' : '#ffffff',
                    fontSize: '13.5px', lineHeight: '1.55',
                    border: isNova ? '1px solid rgba(255,255,255,0.8)' : '1px solid rgba(147,197,253,0.35)',
                    boxShadow: isNova ? '0 5px 14px rgba(30,64,175,0.08)' : '0 5px 14px rgba(30,64,175,0.18)',
                    backdropFilter: 'blur(8px)',
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
            padding: '10px 14px', background: 'rgba(255,255,255,0.22)', borderTop: '1px solid rgba(255,255,255,0.45)',
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

      {/* ── ESCALATION MODAL / OVERLAY ── */}
      {escalationState && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 100,
          background: 'rgba(15, 23, 42, 0.75)', backdropFilter: 'blur(8px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px',
        }}>
          <div style={{
            background: '#ffffff', borderRadius: '20px', maxWidth: '440px', width: '100%',
            padding: '32px 28px', textAlign: 'center', boxShadow: '0 20px 50px rgba(0,0,0,0.3)',
            animation: 'msgIn 0.3s cubic-bezier(0.18,0.89,0.32,1.28) both',
          }}>
            {escalationState === 'pending' && (
              <>
                <div style={{
                  width: '64px', height: '64px', margin: '0 auto 18px', borderRadius: '50%',
                  border: '4px solid #e0e7ff', borderTopColor: '#4338ca',
                  animation: 'spin 1s linear infinite',
                }} />
                <h2 style={{ fontSize: '20px', fontWeight: '800', color: '#1e1b4b', marginBottom: '8px' }}>
                  Connecting to Front Desk
                </h2>
                <p style={{ fontSize: '14px', color: '#475569', lineHeight: '1.6', marginBottom: '16px' }}>
                  {escalationMsg || 'A front desk reception team member has been alerted to assist you directly at this kiosk.'}
                </p>
                <div style={{
                  background: 'linear-gradient(135deg, #eef2ff, #f0fdf4)',
                  borderRadius: '14px', padding: '14px 16px', marginBottom: '16px',
                  border: '1px solid #c7d2fe', textAlign: 'left',
                }}>
                  <div style={{ fontSize: '11px', fontWeight: '700', color: '#4338ca', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
                    📞 Front Desk &amp; Campus Helpline
                  </div>
                  <div style={{ fontSize: '15px', fontWeight: '800', color: '#1e1b4b', marginBottom: '3px' }}>
                    +91 80286 11880 / 81 / 82
                  </div>
                  <div style={{ fontSize: '12px', color: '#3730a3' }}>
                    Admissions Helpline: <strong>+91 81472 86667</strong> · Admin Block Ground Floor
                  </div>
                </div>
                <div style={{
                  background: '#f0f9ff', borderRadius: '12px', padding: '12px 16px',
                  fontSize: '13px', color: '#0369a1', border: '1px solid #bae6fd',
                  fontWeight: '600',
                }}>
                  🎤 Say <strong>"Stop"</strong> anytime to return to Nova
                </div>
              </>
            )}

            {escalationState === 'connected' && (
              <>
                <div style={{ fontSize: '56px', marginBottom: '16px' }}>🤝</div>
                <h2 style={{ fontSize: '20px', fontWeight: '800', color: '#065f46', marginBottom: '8px' }}>
                  Staff Member Connected!
                </h2>
                <p style={{ fontSize: '14px', color: '#334155', lineHeight: '1.6', marginBottom: '20px' }}>
                  {escalationMsg || 'A front desk team member is now assisting you.'}
                </p>
                <div ref={escalationScrollRef} style={{
                  maxHeight: '180px', overflowY: 'auto', textAlign: 'left',
                  background: '#f8fafc', border: '1px solid #e2e8f0',
                  borderRadius: '14px', padding: '10px', marginBottom: '12px'
                }}>
                  {escalationMessages.length === 0 && (
                    <div style={{ fontSize: '12px', color: '#94a3b8', textAlign: 'center', padding: '12px' }}>
                      Speak normally. Your voice is sent to the connected staff member.
                    </div>
                  )}
                  {escalationMessages.map((message, index) => (
                    <div key={`${message.timestamp}-${index}`} style={{
                      display: 'flex', justifyContent: message.speaker === 'visitor' ? 'flex-end' : 'flex-start',
                      marginBottom: '6px'
                    }}>
                      <div style={{
                        maxWidth: '82%', padding: '8px 10px', borderRadius: '10px',
                        background: message.speaker === 'visitor' ? '#dbeafe' : '#dcfce7',
                        color: '#334155', fontSize: '12px', lineHeight: '1.4'
                      }}>
                        <strong style={{ display: 'block', fontSize: '10px', color: '#64748b', marginBottom: '2px' }}>
                          {message.speaker === 'visitor' ? 'You' : 'Staff'}
                        </strong>
                        {message.text}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{
                  background: '#f0fdf4', borderRadius: '12px', padding: '12px 16px',
                  fontSize: '13px', color: '#15803d', border: '1px solid #bbf7d0',
                  fontWeight: '600',
                }}>
                  🎤 Say <strong>&ldquo;Stop&rdquo;</strong> to return to Nova
                </div>
              </>
            )}

            {escalationState === 'timeout' && (
              <>
                <div style={{ fontSize: '56px', marginBottom: '16px' }}>⏳</div>
                <h2 style={{ fontSize: '20px', fontWeight: '800', color: '#92400e', marginBottom: '8px' }}>
                  Staff Currently Occupied
                </h2>
                <p style={{ fontSize: '14px', color: '#475569', lineHeight: '1.6', marginBottom: '16px' }}>
                  {escalationMsg || 'Front desk staff members are currently assisting other visitors. Nova will gladly continue answering your questions!'}
                </p>
                <div style={{
                  background: 'linear-gradient(135deg, #fef3c7, #fffbeb)',
                  borderRadius: '14px', padding: '14px 16px', marginBottom: '18px',
                  border: '1px solid #fde68a', textAlign: 'left',
                }}>
                  <div style={{ fontSize: '11px', fontWeight: '700', color: '#92400e', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
                    📞 Direct Contact Numbers
                  </div>
                  <div style={{ fontSize: '15px', fontWeight: '800', color: '#78350f', marginBottom: '3px' }}>
                    +91 80286 11880 / 81 / 82
                  </div>
                  <div style={{ fontSize: '12px', color: '#92400e' }}>
                    Admissions Desk: <strong>+91 81472 86667</strong> · Admin Block Ground Floor
                  </div>
                </div>
                <div style={{
                  background: '#fffbeb', borderRadius: '12px', padding: '10px 16px',
                  fontSize: '13px', color: '#92400e', border: '1px solid #fde68a',
                  fontWeight: '600',
                }}>
                  🎤 Say anything to continue with Nova
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── QR COMPANION MODAL ── */}
      {showCompanionModal && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 90,
          background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(8px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px',
        }}>
          <div style={{
            background: '#ffffff', borderRadius: '24px', maxWidth: '440px', width: '100%',
            padding: '30px', textAlign: 'center', boxShadow: '0 25px 60px rgba(0,0,0,0.3)',
            animation: 'msgIn 0.3s cubic-bezier(0.18,0.89,0.32,1.28) both',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '24px' }}>📱</span>
                <span style={{ fontSize: '17px', fontWeight: '800', color: '#1a237e' }}>Phone Companion</span>
              </div>
              <button
                onClick={() => setShowCompanionModal(false)}
                style={{
                  border: 'none', background: '#f1f5f9', borderRadius: '50%', width: '32px', height: '32px',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
                  color: '#64748b', fontSize: '16px', fontWeight: '700',
                }}
              >
                ✕
              </button>
            </div>

            <p style={{ fontSize: '13px', color: '#64748b', lineHeight: '1.5', marginBottom: '20px' }}>
              Scan this QR code with your phone camera to take your conversation recap, useful links, and official brochures with you.
            </p>

            {companionToken ? (
              <div style={{
                display: 'inline-block', padding: '12px', background: '#f8fafc',
                borderRadius: '16px', border: '1.5px solid #e2e8f0', marginBottom: '16px',
              }}>
                <img
                  src={`${BACKEND}/companion/qr/${companionToken}`}
                  alt="Companion QR Code"
                  style={{ width: '190px', height: '190px', display: 'block', borderRadius: '8px' }}
                />
              </div>
            ) : (
              <div style={{
                height: '190px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: '#f8fafc', borderRadius: '16px', color: '#94a3b8', fontSize: '14px', marginBottom: '16px',
              }}>
                Generating QR code…
              </div>
            )}

            <div style={{
              display: 'flex', flexDirection: 'column', gap: '6px', background: '#f1f5f9',
              borderRadius: '12px', padding: '12px', textAlign: 'left', fontSize: '11px', color: '#475569',
              marginBottom: '20px',
            }}>
              <div>✅ <strong>Session Summary:</strong> Instant recap of topics discussed</div>
              <div>📄 <strong>Official Brochure:</strong> Direct PDF download on your device</div>
              <div>⏱️ <strong>Session Window:</strong> Valid for 20 minutes</div>
            </div>

            {companionUrl && (
              <a
                href={companionUrl}
                target="_blank"
                rel="noreferrer"
                style={{
                  display: 'block', width: '100%', padding: '11px 0', borderRadius: '12px',
                  background: '#1a237e', color: '#fff', textDecoration: 'none',
                  fontSize: '13px', fontWeight: '700', textAlign: 'center',
                }}
              >
                Open Companion Link Directly
              </a>
            )}
          </div>
        </div>
      )}

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
        @keyframes spin { to { transform: rotate(360deg); } }

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