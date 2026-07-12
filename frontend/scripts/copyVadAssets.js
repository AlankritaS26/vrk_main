/**
 * Copies the Silero VAD + onnxruntime-web assets into public/ so CRA serves
 * them as plain static files. THIS is the fix for "CRA won't serve the .mjs
 * module files" — the worklet, onnx model, and wasm runtime just become
 * regular files under / instead of webpack-processed modules.
 *
 * Runs automatically before `npm start` / `npm run build` (see package.json).
 */
const fs = require('fs');
const path = require('path');

const PUBLIC = path.join(__dirname, '..', 'public');

const SOURCES = [
  {
    dir: path.join(__dirname, '..', 'node_modules', '@ricky0123', 'vad-web', 'dist'),
    match: (f) =>
      f.endsWith('.onnx') || f === 'vad.worklet.bundle.min.js',
  },
  {
    dir: path.join(__dirname, '..', 'node_modules', 'onnxruntime-web', 'dist'),
    match: (f) => f.endsWith('.wasm') || f.endsWith('.mjs'),
  },
];

let copied = 0;
for (const { dir, match } of SOURCES) {
  if (!fs.existsSync(dir)) {
    console.warn(`[copy-vad] missing ${dir} — run npm install first`);
    continue;
  }
  for (const f of fs.readdirSync(dir)) {
    if (!match(f)) continue;
    fs.copyFileSync(path.join(dir, f), path.join(PUBLIC, f));
    copied++;
  }
}
console.log(`[copy-vad] copied ${copied} VAD/ORT assets into public/`);
