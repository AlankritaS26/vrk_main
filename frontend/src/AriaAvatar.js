import React, { useEffect, useState } from 'react';
import portrait from './aria-portrait.jpg';

/**
 * AriaAvatar — real photo with pixel-verified eyelid + mouth overlays.
 * Coordinates below were measured directly against the bundled photo
 * (see /avatar_work notes) so the eyelids sit exactly over the eyes and
 * the mouth mask sits exactly over the lips — not guessed percentages.
 *
 * Eyelids: opaque skin-tone ellipses sampled from this photo's own
 * eyelid/brow pixels, scaled 0→1 to blink (no pixel warping — a real
 * eyelid closing over the eye reads convincingly; distorting the eye
 * itself would not).
 *
 * Mouth: a duplicate of the same photo, masked to just the lip area
 * (feathered edge via mask-image so there's no visible seam), gently
 * translated/stretched while speaking — plus a soft warm shadow behind
 * it that peeks through as it moves, hinting at the mouth opening.
 *
 * Driven only by `status` ('ready' | 'listening' | 'processing' | 'speaking').
 * No audio taps, no GPU, no new backend calls — presentational only.
 */
const STATE_COLOR = {
  ready: '#ffb300',
  listening: '#43a047',
  processing: '#7e57c2',
  speaking: '#ef5350',
};

// Measured against src/assets/aria-portrait.jpg (640×755)
const LEFT_EYE = { x: 39.6, y: 46.8, rx: 3.6, ry: 1.6 };
const RIGHT_EYE = { x: 59.8, y: 46.8, rx: 3.6, ry: 1.6 };
const MOUTH = { x: 49.8, y: 59.7, rx: 7, ry: 2.6 };

export default function AriaAvatar({ status = 'ready', size = 220 }) {
  const [blink, setBlink] = useState(false);

  // Natural, irregular blinking.
  useEffect(() => {
    let alive = true;
    const cycle = () => {
      const delay = 2400 + Math.random() * 2800;
      const t = setTimeout(() => {
        if (!alive) return;
        setBlink(true);
        setTimeout(() => { if (alive) setBlink(false); }, 120);
        cycle();
      }, delay);
      return t;
    };
    const t = cycle();
    return () => { alive = false; clearTimeout(t); };
  }, []);

  const color = STATE_COLOR[status] || STATE_COLOR.ready;
  const maskCss = `radial-gradient(ellipse ${MOUTH.rx}% ${MOUTH.ry}% at ${MOUTH.x}% ${MOUTH.y}%, #000 55%, transparent 100%)`;

  return (
    <div className={`human-avatar human-avatar--${status}`} style={{ width: size, height: size * 1.18 }}>
      <span className="human-glow" style={{ background: `radial-gradient(circle, ${color}40, transparent 68%)` }} />

      <div className="human-frame">
        <img src={portrait} alt="Nova — RNSIT Digital Receptionist" className="human-photo" />

        {/* warm shadow that peeks through as the mouth layer shifts — reads as "opening" */}
        <span className="human-mouth-shadow" style={{ left: `${MOUTH.x}%`, top: `${MOUTH.y}%`, width: `${MOUTH.rx * 2}%`, height: `${MOUTH.ry * 2.2}%` }} />

        {/* duplicated + masked mouth region, animates during speech */}
        <div className="human-mouth-layer" style={{ WebkitMaskImage: maskCss, maskImage: maskCss, transformOrigin: `${MOUTH.x}% ${MOUTH.y}%` }}>
          <img src={portrait} alt="" className="human-photo" />
        </div>

        {/* eyelids */}
        <span className={`human-eyelid ${blink ? 'is-blinking' : ''}`} style={{ left: `${LEFT_EYE.x}%`, top: `${LEFT_EYE.y}%`, width: `${LEFT_EYE.rx * 2}%`, height: `${LEFT_EYE.ry * 2}%` }} />
        <span className={`human-eyelid ${blink ? 'is-blinking' : ''}`} style={{ left: `${RIGHT_EYE.x}%`, top: `${RIGHT_EYE.y}%`, width: `${RIGHT_EYE.rx * 2}%`, height: `${RIGHT_EYE.ry * 2}%` }} />

        <span className="human-ring" style={{ borderColor: color }} />
      </div>

      <span className="human-status-dot" style={{ background: color, boxShadow: `0 0 0 4px ${color}22` }} />

      <style>{`
        .human-avatar { position: relative; flex-shrink: 0; }

        .human-glow {
          position: absolute; inset: -10%; border-radius: 50%;
          filter: blur(8px); transition: background 0.4s ease;
          animation: human-glow-pulse 3.6s ease-in-out infinite;
        }
        .human-avatar--listening .human-glow { animation-duration: 1.6s; }
        .human-avatar--processing .human-glow,
        .human-avatar--speaking .human-glow { animation-duration: 2s; }
        @keyframes human-glow-pulse { 0%,100%{opacity:0.5;transform:scale(0.96)} 50%{opacity:1;transform:scale(1.04)} }

        .human-frame {
          position: relative; width: 100%; height: 100%; border-radius: 50% 50% 46% 46%;
          overflow: hidden; box-shadow: 0 10px 26px rgba(26,35,126,0.28);
          transition: transform 0.35s ease;
        }
        .human-avatar--ready .human-frame     { animation: human-breathe 4s ease-in-out infinite; }
        .human-avatar--listening .human-frame { transform: scale(1.035); }
        .human-avatar--processing .human-frame{ animation: human-tilt 3.2s ease-in-out infinite; }
        .human-avatar--speaking .human-frame  { animation: human-nod 0.8s ease-in-out infinite; }
        @keyframes human-breathe { 0%,100%{transform:scale(1) translateY(0)} 50%{transform:scale(1.015) translateY(-1.5px)} }
        @keyframes human-tilt    { 0%,100%{transform:rotate(0deg)} 30%{transform:rotate(2.6deg)} 70%{transform:rotate(-1.2deg)} }
        @keyframes human-nod     { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-1.6px)} }

        .human-photo {
          width: 100%; height: 100%; object-fit: cover; object-position: 50% 50%;
          display: block; -webkit-user-drag: none; user-select: none;
        }

        .human-ring {
          position: absolute; inset: 0; border-radius: inherit;
          border: 3px solid; opacity: 0.55; pointer-events: none;
          transition: border-color 0.3s ease;
        }

        /* ── eyelids ── */
        .human-eyelid {
          position: absolute; border-radius: 50%; pointer-events: none;
          background: linear-gradient(180deg, #825f4e 0%, #966955 100%);
          transform: translate(-50%, -50%) scaleY(0);
          transform-origin: center;
          transition: transform 0.09s ease-in-out;
        }
        .human-eyelid.is-blinking { transform: translate(-50%, -50%) scaleY(1); }
        /* natural resting blink even outside deliberate "is-blinking" toggles is handled in JS */

        /* ── mouth ── */
        .human-mouth-shadow {
          position: absolute; transform: translate(-50%, -50%);
          border-radius: 50%; pointer-events: none;
          background: radial-gradient(ellipse, #6b2f2a 0%, rgba(107,47,42,0) 75%);
          opacity: 0;
        }
        .human-avatar--speaking .human-mouth-shadow { animation: human-mouth-shadow-pulse 0.48s ease-in-out infinite; }
        @keyframes human-mouth-shadow-pulse { 0%,100%{opacity:0} 45%{opacity:0.15} 50%{opacity:0.85} 55%{opacity:0.15} }

        .human-mouth-layer { position: absolute; inset: 0; pointer-events: none; }
        .human-avatar--speaking .human-mouth-layer { animation: human-mouth-move 0.48s ease-in-out infinite; }
        @keyframes human-mouth-move {
          0%, 100% { transform: translateY(0) scaleY(1); }
          50%      { transform: translateY(1.6%) scaleY(1.22); }
        }

        .human-status-dot {
          position: absolute; bottom: 5%; right: 20%;
          width: 13px; height: 13px; border-radius: 50%;
          border: 2px solid #fff; transition: background 0.3s ease;
        }
      `}</style>
    </div>
  );
}