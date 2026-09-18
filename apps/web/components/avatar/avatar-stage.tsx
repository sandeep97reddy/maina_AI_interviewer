"use client";

/**
 * <AvatarStage> — the avatar centerpiece.
 *
 * Asset-free by design: each persona is an emoji on a per-persona gradient
 * stage (`emoji`/`color` in `lib/personas.ts`), with a state-aware breathing
 * pulse while speaking. Driven entirely by props (`persona`, `state`) —
 * fully decoupled from LiveKit.
 */

import { cn } from "@/lib/cn";
import type { Persona } from "@/lib/personas";

export type AvatarState = "idle" | "listening" | "thinking" | "speaking";

export interface AvatarStageProps {
  persona: Persona;
  state: AvatarState;
  className?: string;
}

/**
 * Per-persona fallback hues — kept inside the editorial palette (light, paper,
 * one indigo accent). Each is a subtle two-stop gradient + a soft accent glow,
 * so the three personas read as distinct without leaving the design language.
 */
const FALLBACK_STYLE: Record<
  Persona["id"],
  { gradient: string; glow: string; accent: string }
> = {
  // Anime — warm pastel rose/peach, soft and friendly.
  anime: {
    gradient: "linear-gradient(165deg, #fdf6f3 0%, #f7eef0 55%, #f0ecf6 100%)",
    glow: "radial-gradient(120% 90% at 50% 18%, rgba(225,150,170,0.22), transparent 60%)",
    accent: "#b65a78",
  },
  // Superhero — cool steel/indigo, calm and heroic.
  superhero: {
    gradient: "linear-gradient(165deg, #f3f5fb 0%, #eceef8 55%, #e8eaf4 100%)",
    glow: "radial-gradient(120% 90% at 50% 18%, rgba(67,56,202,0.20), transparent 60%)",
    accent: "#4338ca",
  },
  // Recruiter — neutral office grey-green, professional and warm.
  recruiter: {
    gradient: "linear-gradient(165deg, #faf9f6 0%, #f2f1ec 55%, #edefee 100%)",
    glow: "radial-gradient(120% 90% at 50% 18%, rgba(120,140,130,0.20), transparent 60%)",
    accent: "#4a6b5d",
  },
  professor: {
    gradient: "linear-gradient(165deg, #f9f6f0 0%, #f0ece2 55%, #e8e4da 100%)",
    glow: "radial-gradient(120% 90% at 50% 18%, rgba(160,140,100,0.20), transparent 60%)",
    accent: "#7a6b42",
  },
};
const STATE_LABEL: Record<AvatarState, string> = {
  idle: "IDLE",
  listening: "LISTENING",
  thinking: "THINKING",
  speaking: "SPEAKING",
};

/**
 * Scoped keyframes for the fallback breathing/pulse. Injected as a plain <style>
 * (not styled-jsx) so it needs no globals.css edit and no extra dep. Both
 * animations freeze under `prefers-reduced-motion` per the a11y requirement.
 */
const STAGE_KEYFRAMES = `
@keyframes di-avatar-breathe {
  0%, 100% { opacity: 0.55; transform: scale(1); }
  50%      { opacity: 0.8;  transform: scale(1.03); }
}
@keyframes di-avatar-speak {
  0%, 100% { opacity: 0.6;  transform: scale(1); }
  50%      { opacity: 1;    transform: scale(1.08); }
}
.di-avatar-pulse { animation: di-avatar-breathe 4.5s ease-in-out infinite; }
.di-avatar-pulse[data-speaking="true"] { animation: di-avatar-speak 1.6s ease-in-out infinite; }
@media (prefers-reduced-motion: reduce) {
  .di-avatar-pulse { animation: none !important; }
}
`;

export function AvatarStage({ persona, state, className }: AvatarStageProps) {
  const speaking = state === "speaking";

  const look = FALLBACK_STYLE[persona.id];

  return (
    <div
      className={cn(
        "relative aspect-[4/5] w-full overflow-hidden rounded-card",
        "border border-line bg-panel select-none",
        className,
      )}
      role="img"
      aria-label={`${persona.name} avatar, ${STATE_LABEL[state].toLowerCase()}`}
    >
      <style>{STAGE_KEYFRAMES}</style>

      {/* Stage — gradient + glow + breathing orb + emoji identity. */}
      <div className="absolute inset-0" style={{ background: look.gradient }}>
        {/* Soft accent glow + breathing orb behind the emoji. */}
        <div className="absolute inset-0" style={{ background: look.glow }} />
        <div
          className="di-avatar-pulse absolute left-1/2 top-[34%] h-40 w-40 -translate-x-1/2 -translate-y-1/2 rounded-full blur-2xl"
          data-speaking={speaking}
          style={{ backgroundColor: look.accent, opacity: 0.5 }}
        />

        {/* Persona identity. Sits above the bottom audio visualizer (h-20 in
            voice-stage), so pad clear of that 5rem band to avoid overlap. */}
        <div className="absolute inset-x-0 bottom-0 flex flex-col items-center gap-1 px-5 pb-24 text-center">
          <span aria-hidden="true" className="text-6xl leading-none">
            {persona.emoji}
          </span>
          <span className="serif mt-2 text-2xl text-ink">{persona.name}</span>
          <span className="max-w-[26ch] text-xs leading-snug text-muted">
            {persona.style}
          </span>
        </div>
      </div>

      {/* State pill — small, mono, for at-a-glance clarity. */}
      <div className="absolute left-3 top-3 z-10">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md px-2 py-0.5",
            "bg-paper/85 font-mono text-[10px] tracking-[0.12em] text-ink-soft",
            "border border-line backdrop-blur-sm",
          )}
        >
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              speaking ? "bg-accent" : "bg-faint",
            )}
          />
          {STATE_LABEL[state]}
        </span>
      </div>
    </div>
  );
}
