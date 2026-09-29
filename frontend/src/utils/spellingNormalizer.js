/*
  Robust spoken-name spelling parser.

  Supported examples:
  - "A L A N K R I T A" -> Alankrita
  - "A-L-A-N-K-R-I-T-A" -> Alankrita
  - "Alfa Lima Alfa November Kilo Romeo India Tango Alfa" -> Alankrita
  - "A as in Apple L as in Lion A as in Apple" -> Ala
  - "J O double N Y" -> Jonny
  - "M I double S I S S I P P I" -> Mississippi
*/

const LETTER_MAP = {
  a: "A",
  b: "B",
  c: "C",
  d: "D",
  e: "E",
  f: "F",
  g: "G",
  h: "H",
  i: "I",
  j: "J",
  k: "K",
  l: "L",
  m: "M",
  n: "N",
  o: "O",
  p: "P",
  q: "Q",
  r: "R",
  s: "S",
  t: "T",
  u: "U",
  v: "V",
  w: "W",
  x: "X",
  y: "Y",
  z: "Z",

  bee: "B",
  be: "B",
  sea: "C",
  see: "C",
  cee: "C",
  dee: "D",
  gee: "G",
  aitch: "H",
  jay: "J",
  kay: "K",
  el: "L",
  ell: "L",
  em: "M",
  en: "N",
  oh: "O",
  owe: "O",
  pee: "P",
  cue: "Q",
  queue: "Q",
  ar: "R",
  are: "R",
  ess: "S",
  es: "S",
  tee: "T",
  tea: "T",
  you: "U",
  vee: "V",
  doubleyou: "W",
  doubleu: "W",
  ex: "X",
  why: "Y",
  zed: "Z",
  zee: "Z",

  alfa: "A",
  alpha: "A",
  bravo: "B",
  charlie: "C",
  delta: "D",
  echo: "E",
  foxtrot: "F",
  golf: "G",
  hotel: "H",
  india: "I",
  juliet: "J",
  juliett: "J",
  kilo: "K",
  lima: "L",
  mike: "M",
  november: "N",
  oscar: "O",
  papa: "P",
  quebec: "Q",
  romeo: "R",
  sierra: "S",
  tango: "T",
  uniform: "U",
  victor: "V",
  whiskey: "W",
  whisky: "W",
  xray: "X",
  yankee: "Y",
  zulu: "Z",
};

const MULTIPLIERS = {
  double: 2,
  twin: 2,
  two: 2,
  triple: 3,
  three: 3,
};

const SKIP_PATTERN =
  /\b(skip|guest|continue|continue as guest|no name|never mind|nevermind|cancel|stop|pass|later|does not matter|doesn't matter|dont care|don't care)\b/i;

export function resolveLetter(token) {
  if (!token) return "";

  const clean = token.toLowerCase().replace(/[^a-z]/g, "");

  if (!clean) return "";

  return LETTER_MAP[clean] || "";
}

export function resolveSingleLetter(rawText = "") {
  const original = String(rawText || "").trim();
  if (!original || isSpellingSkipOrGuest(original)) return "";

  let text = original.toLowerCase()
    .replace(/\b(letter|next letter is|next letter|it is|it's|its)\b/g, " ")
    .replace(/[—–]/g, " ")
    .replace(/[-,./\\|:;!?()[\]{}]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const tokens = text.split(" ").filter(Boolean);
  if (!tokens.length) return "";

  // Check for "A as in Apple" or "A for Apple"
  if (tokens.length >= 2 && ["as", "for", "like"].includes(tokens[1])) {
    const direct = resolveLetter(tokens[0]);
    if (direct) return direct;
  }

  // Check first token (e.g. "A", "ay", "Alpha")
  const first = resolveLetter(tokens[0]);
  if (first) return first;

  // Check last token (e.g. "the letter L")
  if (tokens.length > 1) {
    const last = resolveLetter(tokens[tokens.length - 1]);
    if (last) return last;
  }

  return "";
}

export function isSpellingSkipOrGuest(text = "") {
  return SKIP_PATTERN.test(String(text));
}

export function normalizeSpelledName(rawText = "") {
  const original = String(rawText || "").trim();

  if (!original || isSpellingSkipOrGuest(original)) {
    return {
      name: "",
      spelledOut: "",
      letters: [],
    };
  }

  let text = original.toLowerCase();

  text = text
    .replace(/\b(my name is|my name|it is|it's|its|the spelling is|spelled|spell it)\b/g, " ")
    .replace(/[—–]/g, " ")
    .replace(/[-,./\\|:;!?()[\]{}]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const tokens = text.split(" ").filter(Boolean);
  const letters = [];

  for (let index = 0; index < tokens.length; index += 1) {
    const current = tokens[index];

    /*
      Handles:
      "A as in apple" -> A
      "L for lion" -> L
      "A like apple" -> A
    */
    if (
      index + 2 < tokens.length &&
      ["as", "for", "like"].includes(tokens[index + 1])
    ) {
      const firstLetter = resolveLetter(current);

      if (firstLetter) {
        letters.push(firstLetter);

        if (tokens[index + 1] === "as" && tokens[index + 2] === "in") {
          index += 3;
        } else {
          index += 2;
        }

        continue;
      }
    }

    /*
      Handles:
      "double L" -> LL
      "triple A" -> AAA
    */
    if (MULTIPLIERS[current] && index + 1 < tokens.length) {
      const repeatedLetter = resolveLetter(tokens[index + 1]);

      if (repeatedLetter) {
        for (let count = 0; count < MULTIPLIERS[current]; count += 1) {
          letters.push(repeatedLetter);
        }

        index += 1;
        continue;
      }
    }

    const letter = resolveLetter(current);

    if (letter) {
      letters.push(letter);
      continue;
    }

    /*
      If Whisper returns a continuous spelling such as "alankrita" rather than
      individual letters, it is safer to treat it as an ordinary name candidate.
      This is useful only when it is the sole token.
    */
    if (tokens.length === 1 && /^[a-z]+$/i.test(current)) {
      for (const character of current.toUpperCase()) {
        letters.push(character);
      }
    }
  }

  const joined = letters.join("");

  if (joined.length < 2) {
    return {
      name: "",
      spelledOut: "",
      letters: [],
      isValid: false,
    };
  }

  return {
    name: joined.charAt(0) + joined.slice(1).toLowerCase(),
    spelledOut: letters.join("-"),
    letters,
    isValid: true,
  };
}