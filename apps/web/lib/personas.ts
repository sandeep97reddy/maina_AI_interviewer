/**
 * Avatar persona catalog.
 *
 * No image/video assets: each persona is identified by an emoji + accent color
 * rendered by `<AvatarStage>` and the setup picker. Each persona also owns an
 * Edge TTS voice, sent to the agent via LiveKit token metadata (`voice`) so
 * the interviewer actually sounds different per persona.
 *
 * Pure data — no `server-only`, no env reads — safe to import from any server
 * or client component.
 */

/** Stable persona ids used across the pipeline. Never rename these. */
export type PersonaId = "anime" | "superhero" | "recruiter" | "professor";

export interface Persona {
  /** Stable id sent through the interview pipeline. */
  id: PersonaId;
  /** Display name shown on the picker card. */
  name: string;
  /** One-line interviewer style/tone, surfaced under the name. */
  style: string;
  /** Emoji shown on the avatar tile — the visual identity. */
  emoji: string;
  /** Accent color (hex) for the tile gradient/glow. */
  color: string;
  /** Edge TTS voice for this persona (must be a `*Neural` voice id). */
  edgeVoice: string;
}

export const PERSONAS: Persona[] = [
  {
    id: "anime",
    name: "Mika",
    style: "Bright, encouraging anime mentor who keeps the energy up.",
    emoji: "✨",
    color: "#b65a78",
    edgeVoice: "en-GB-SoniaNeural",
  },
  {
    id: "superhero",
    name: "Vanguard",
    style: "Bold superhero coach who pushes you to your best answer.",
    emoji: "🔥",
    color: "#4338ca",
    edgeVoice: "en-US-GuyNeural",
  },
  {
    id: "recruiter",
    name: "Dana",
    style: "Calm, professional recruiter — true-to-life screening tone.",
    emoji: "🎙️",
    color: "#4a6b5d",
    edgeVoice: "en-US-JennyNeural",
  },
  {
    id: "professor",
    name: "Dr. Chen",
    style: "Calm, thoughtful academic who probes deeply and values precision.",
    emoji: "🎓",
    color: "#7a6b42",
    edgeVoice: "en-GB-RyanNeural",
  },
];

/** Default persona when the user hasn't picked one yet. */
export const DEFAULT_PERSONA_ID = "recruiter";

/** Look up a persona by id, falling back to the default. */
export function getPersona(id: string | undefined): Persona {
  const fallback =
    PERSONAS.find((p) => p.id === DEFAULT_PERSONA_ID) ?? PERSONAS[0];
  // The catalog is always non-empty, so a fallback exists.
  return PERSONAS.find((p) => p.id === id) ?? (fallback as Persona);
}
