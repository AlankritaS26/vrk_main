const LETTER_MAP = {
  a: "A", b: "B", c: "C", d: "D", e: "E", f: "F", g: "G",
  h: "H", i: "I", j: "J", k: "K", l: "L", m: "M", n: "N",
  o: "O", p: "P", q: "Q", r: "R", s: "S", t: "T", u: "U",
  v: "V", w: "W", x: "X", y: "Y", z: "Z",

  bee: "B", be: "B",
  sea: "C", see: "C", cee: "C",
  dee: "D",
  gee: "G",
  aitch: "H",
  jay: "J",
  kay: "K",
  el: "L", ell: "L",
  em: "M",
  en: "N",
  oh: "O", owe: "O",
  pee: "P",
  cue: "Q", queue: "Q",
  ar: "R", are: "R",
  ess: "S", es: "S",
  tee: "T", tea: "T",
  you: "U",
  vee: "V",
  doubleyou: "W", doubleu: "W",
  ex: "X",
  why: "Y",
  zed: "Z", zee: "Z",

  alfa: "A", alpha: "A",
  bravo: "B",
  charlie: "C",
  delta: "D",
  echo: "E",
  foxtrot: "F",
  golf: "G",
  hotel: "H",
  india: "I",
  juliet: "J", juliett: "J",
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
  whiskey: "W", whisky: "W",
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

const SKIP_OR_GUEST =
  /\b(skip|guest|continue|continue as guest|no name|never mind|nevermind|cancel|stop|pass|later|does not matter|doesn't matter|dont care|don't care)\b/i;

function resolveLetter(token) {
  if (!token) return "";

  const cleaned = String(token).toLowerCase().replace(/[^a-z]/g, "");

  if (!cleaned) return "";

  return LETTER_MAP[cleaned] || "";
}

export function isSpellingSkipOrGuest(text = "") {
  return SKIP_OR_GUEST.test(String(text));
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
    const token = tokens[index];

    if (
      index + 2 < tokens.length &&
      ["as", "for", "like"].includes(tokens[index + 1])
    ) {
      const leadingLetter = resolveLetter(token);

      if (leadingLetter) {
        letters.push(leadingLetter);

        if (tokens[index + 1] === "as" && tokens[index + 2] === "in") {
          index += 3;
        } else {
          index += 2;
        }

        continue;
      }
    }

    if (MULTIPLIERS[token] && index + 1 < tokens.length) {
      const repeated = resolveLetter(tokens[index + 1]);

      if (repeated) {
        for (let repeat = 0; repeat < MULTIPLIERS[token]; repeat += 1) {
          letters.push(repeated);
        }

        index += 1;
        continue;
      }
    }

    const resolved = resolveLetter(token);

    if (resolved) {
      letters.push(resolved);
      continue;
    }

    if (tokens.length === 1 && /^[a-z]+$/i.test(token)) {
      for (const character of token.toUpperCase()) {
        letters.push(character);
      }
    }
  }

  const combined = letters.join("");

  if (combined.length < 2) {
    return {
      name: "",
      spelledOut: "",
      letters: [],
    };
  }

  return {
    name: combined.charAt(0) + combined.slice(1).toLowerCase(),
    spelledOut: letters.join("-"),
    letters,
  };
}