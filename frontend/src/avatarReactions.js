/**
 * Avatar Reactions & Personality System
 * 
 * Detects question intent from user messages and triggers appropriate avatar reactions.
 * Also provides time-based greetings and contextual personality touches.
 *
 * NOTE: Reaction states are mostly head/body based (tilt, bob, sway,
 * expression), but the 'thinking' processing state and 'thanking' state
 * DO use hand/arm gestures (see NovaCharacter's arm-think/hand-think in
 * WelcomeScreen.js) — these are intentional and should be kept.
 */

// ─── INTENT DETECTION ─────────────────────────────────────────────────────────

/**
 * Detect question intent from user input
 * Returns: 'curious' | 'happy' | 'confused' | 'thanks' | 'greeting' | 'standard'
 */
export function detectQuestionIntent(text) {
  if (!text) return 'standard';
  
  const lower = text.toLowerCase().trim();
  
  // Gratitude/Thanks
  if (lower.match(/\b(thanks?|thank you|appreciated|grateful|grateful|thankyou|ty)\b/)) {
    return 'thanks';
  }
  
  // Happy/Positive questions
  if (lower.match(/\b(placements?|best|great|excellent|amazing|wonderful|beautiful|favorite)\b/)) {
    return 'happy';
  }
  
  // Curious questions (how, why, what, tell me about)
  if (lower.match(/^(how|why|what|tell me about|explain|describe|show|list)\b/)) {
    return 'curious';
  }
  
  // Greetings
  if (lower.match(/\b(hello|hi|hey|good morning|good afternoon|good evening|namaste)\b/)) {
    return 'greeting';
  }
  
  // Confused/Typo (very short or random words)
  if (lower.length < 3 || lower.split(' ').length > 12) {
    return 'confused';
  }
  
  return 'standard';
}

// ─── TIME-BASED GREETINGS ─────────────────────────────────────────────────────

/**
 * Get greeting based on time of day
 */
export function getTimeBasedGreeting() {
  const hour = new Date().getHours();
  
  if (hour >= 6 && hour < 12) {
    return { greeting: "Good morning! ☀️", emoji: '🌅' };
  } else if (hour >= 12 && hour < 18) {
    return { greeting: "Good afternoon! 🌤️", emoji: '☀️' };
  } else if (hour >= 18 && hour < 21) {
    return { greeting: "Good evening! 🌙", emoji: '🌙' };
  } else {
    return { greeting: "It's late! Still here? 🌙", emoji: '⭐' };
  }
}

// ─── PERSONALITY RESPONSES ────────────────────────────────────────────────────

/**
 * Get personality-enhanced response prefix based on question intent
 */
export function getPersonalityPrefix(intent) {
  const prefixes = {
    curious: [
      "Great question! Let me look into that for you.",
      "Interesting! Here's what I found...",
      "Good thinking! Let me explain...",
      "Love the curiosity! Here's the scoop...",
    ],
    happy: [
      "I love your enthusiasm! Yes, our placements are...",
      "Wonderful to hear your interest! Here's the great news...",
      "That's fantastic that you're excited! Let me share...",
      "Your energy is amazing! Here's what makes it great...",
    ],
    thanks: [
      "You're welcome! I'm always happy to help.",
      "My pleasure! That's what I'm here for.",
      "Glad I could assist! Anything else?",
      "Anytime! Feel free to ask more questions.",
    ],
    greeting: [
      "Welcome! Great to see you. How can I help?",
      "Hello there! What brings you here?",
      "Hey! Nice to meet you. What's on your mind?",
      "Glad you're here! What would you like to know?",
    ],
    confused: [
      "I'm not quite sure I understood that. Could you rephrase?",
      "Hmm, that's an interesting question. Let me think...",
      "I didn't quite catch that. Can you say it again?",
      "That's a unique question! Let me see if I can help...",
    ],
    standard: [
      "Here's what I found...",
      "Let me help you with that...",
      "Sure! Here's the information...",
      "Here you go...",
    ],
  };
  
  const list = prefixes[intent] || prefixes.standard;
  return list[Math.floor(Math.random() * list.length)];
}

/**
 * Get personality-enhanced response suffix based on interaction
 */
export function getPersonalitySuffix(visitCount, isReturning) {
  if (isReturning && visitCount > 3) {
    return [
      " You're a regular! I like that. 😊",
      " Great to see you again! Any other questions?",
      " You're becoming a campus expert! Anything else?",
      " Welcome back, explorer! Need more info?",
    ][Math.floor(Math.random() * 4)];
  }
  
  return [
    " Does that help?",
    " Anything else I can clarify?",
    " Need more information?",
    " Let me know if you have more questions!",
  ][Math.floor(Math.random() * 4)];
}

// ─── IDLE SCREEN MESSAGES ─────────────────────────────────────────────────────

/**
 * Get personality-driven idle screen messages
 */
export function getIdleScreenMessage(distanceM) {
  const messages = [
    "Waiting for someone brilliant... ✨",
    "Ready to help! Come on over 👋",
    "Bored of staring at walls... hello? 👀",
    "I'm here to answer all your questions!",
    "Welcome to RNSIT! What would you like to know?",
    "Curious about campus? Let's chat!",
    "I can tell you anything about RNSIT...",
    "Walk up and say hi! I don't bite 😊",
  ];
  
  // Distance-based encouragement
  if (distanceM !== null && distanceM < 2) {
    return "Oh hi there! Come closer, I won't bite! 😄";
  }
  
  return messages[Math.floor(Math.random() * messages.length)];
}

// ─── AVATAR STATE MAPPING ─────────────────────────────────────────────────────

/**
 * Map intent to avatar animation state
 * Standard states: ready, listening, processing, speaking
 * Reaction states: curious, delighted, confused, enthusiastic, thanking
 * (all head/body based — see file header note; no gesture states exist)
 */
export function getAvatarStateForIntent(intent) {
  const stateMap = {
    curious: 'curious',      // Head tilt, focused
    happy: 'delighted',      // Bright bob, happy expression
    thanks: 'thanking',      // Warm head/body bow (no gesture)
    greeting: 'enthusiastic', // Head wiggle, big smile
    confused: 'confused',    // Head shake
    standard: 'processing',  // Default thinking state
  };
  
  return stateMap[intent] || 'processing';
}

// ─── REACTION TIMING ──────────────────────────────────────────────────────────

/**
 * How long to hold the reaction state before switching to speaking
 * (allows brief moment of emotional expression before answering)
 */
export function getReactionDuration(intent) {
  const durations = {
    curious: 800,       // Quick tilt
    happy: 600,         // Quick smile
    thanks: 1200,       // Longer bow
    greeting: 1000,     // Head wiggle
    confused: 800,      // Shrug
    standard: 200,      // Minimal
  };
  
  return durations[intent] || 200;
}