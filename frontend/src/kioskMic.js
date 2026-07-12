/**
 * kioskMic.js — browser-side voice capture for the VRK Kiosk
 *
 * Replaces the MediaRecorder + homemade silence-checker approach.
 * Silero VAD runs IN THE BROWSER (tiny neural VAD, negligible CPU) and
 * hands us one Float32 PCM utterance @ 16 kHz every time the visitor
 * finishes speaking. We convert to Int16 and POST it to the GPU backend.
 *
 * Why this is faster + more accurate than before:
 *  - neural VAD end-of-speech (~300 ms) vs 1200 ms fixed silence timer
 *  - raw PCM straight to Whisper — no WebM encode, no ffmpeg decode
 *  - positiveSpeechThreshold 0.8 rejects distant crowd chatter
 *
 * Assets: run `npm run copy-vad` once (auto-runs before npm start) so the
 * VAD worklet/onnx/wasm files are served from public/ — this is the fix
 * for CRA refusing to serve the .mjs module files.
 */

import { MicVAD } from '@ricky0123/vad-web';

export function float32ToInt16(f32) {
  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return i16;
}

export async function createKioskMic({ onSpeechStart, onSpeechEnd, onMisfire, onStream }) {
  // One persistent stream for the whole session (no per-turn getUserMedia).
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      noiseSuppression: true,
      echoCancellation: true,   // stops kiosk TTS re-triggering the mic
      autoGainControl: false,   // AGC amplifies crowd noise during silence —
                                // it defeats the backend energy gate. Keep OFF.
    },
  });
  onStream?.(stream);

  const vad = await MicVAD.new({
    stream,
    baseAssetPath: '/',         // served from public/ (see scripts/copyVadAssets.js)
    onnxWASMBasePath: '/',

    // Kiosk tuning — accept the person in front, reject background voices
    positiveSpeechThreshold: 0.8,
    negativeSpeechThreshold: 0.55,
    minSpeechFrames: 4,         // ignore coughs / screen taps (<~130 ms)
    redemptionFrames: 8,        // ~800 ms pause = end of utterance

    onSpeechStart,
    onSpeechEnd,                // receives Float32Array @ 16 kHz
    onVADMisfire: onMisfire,
  });

  vad.start();

  return {
    pause: () => vad.pause(),   // call while TTS speaks / name modal open
    resume: () => vad.start(),
    destroy: () => {
      try { vad.destroy(); } catch (e) { /* already destroyed */ }
      stream.getTracks().forEach((t) => t.stop());
    },
  };
}
