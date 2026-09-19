"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Play, Square } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { useMessages } from "@/lib/i18n/client";
import { t } from "@/lib/i18n";
import { cn } from "@/lib/cn";

type Status = "idle" | "loading" | "playing";

/**
 * Singleton stop so only one persona preview plays at a time — starting a
 * second card stops the first (server <audio> or browser speech).
 */
let stopActive: (() => void) | null = null;
function claim(stop: () => void) {
  stopActive?.();
  stopActive = stop;
}
function release(stop: () => void) {
  if (stopActive === stop) stopActive = null;
}

function isFemaleVoice(edgeVoice: string): boolean {
  // Catalog today: Jenny/Sonia are female, Guy/Ryan male. Match on the voice
  // id (not the persona name) so future renames don't flip the heuristic.
  return /jenny|sonia|aria|emma|davis/i.test(edgeVoice);
}

/** Pick the closest built-in browser voice for offline preview fallback. */
function pickBrowserVoice(
  voices: SpeechSynthesisVoice[],
  edgeVoice: string,
): SpeechSynthesisVoice | null {
  if (voices.length === 0) return null;
  const langPrefix = edgeVoice.slice(0, 5).toLowerCase(); // "en-us" / "en-gb"
  const langCandidates = voices.filter((v) =>
    v.lang.toLowerCase().startsWith(langPrefix),
  );
  const pool =
    langCandidates.length > 0
      ? langCandidates
      : voices.filter((v) => v.lang.toLowerCase().startsWith("en"));
  const candidates = pool.length > 0 ? pool : voices;
  const female = isFemaleVoice(edgeVoice);
  const gendered = candidates.find((v) =>
    female
      ? /female|jenny|sonia|aria|samantha|zira|google uk english female/i.test(
          v.name,
        )
      : /male|guy|ryan|david|mark|daniel|google uk english male/i.test(v.name),
  );
  return gendered ?? candidates[0] ?? null;
}

export function VoicePreviewButton({
  edgeVoice,
  personaName,
}: {
  /** Edge TTS voice id for this persona (e.g. "en-US-JennyNeural"). */
  edgeVoice: string;
  /** Persona display name, used for the fallback utterance + aria labels. */
  personaName: string;
}) {
  const messages = useMessages();
  const [status, setStatus] = useState<Status>("idle");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const cancelledRef = useRef(false);

  const stop = useCallback(() => {
    cancelledRef.current = true;
    // Server-audio path.
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audioRef.current = null;
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    // Browser-speech path.
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    release(stopRef.current);
    setStatus("idle");
  }, []);
  const stopRef = useRef(stop);
  stopRef.current = stop;

  // Always clean up on unmount (stops audio + frees the blob URL).
  useEffect(() => {
    const current = stopRef.current;
    return () => {
      current();
    };
  }, []);

  function playBrowserFallback(sample: string) {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      setStatus("idle");
      return;
    }
    const synth = window.speechSynthesis;
    synth.cancel();
    const utter = new SpeechSynthesisUtterance(sample);
    const voices = synth.getVoices();
    const match = pickBrowserVoice(voices, edgeVoice);
    if (match) utter.voice = match;
    utter.lang = match?.lang ?? edgeVoice.slice(0, 5);
    utter.rate = 1;
    utter.pitch = isFemaleVoice(edgeVoice) ? 1.1 : 0.9;
    utter.onend = () => {
      if (!cancelledRef.current) {
        release(stopRef.current);
        setStatus("idle");
      }
    };
    utter.onerror = () => {
      if (!cancelledRef.current) {
        release(stopRef.current);
        setStatus("idle");
      }
    };
    claim(stopRef.current);
    setStatus("playing");
    synth.speak(utter);
  }

  async function toggle() {
    if (status !== "idle") {
      stop();
      return;
    }
    cancelledRef.current = false;
    setStatus("loading");
    const sample = `Hi, I'm ${personaName}. I'll be guiding your interview today. How does my voice sound?`;

    // Prefer the EXACT interview voice via the server proxy (Edge bridge when
    // it's running). Anything non-OK — bridge down, offline — falls through
    // to the built-in browser voice so preview still works with zero setup.
    try {
      const res = await fetch(
        `/api/voice-preview?voice=${encodeURIComponent(edgeVoice)}`,
        { cache: "force-cache" },
      );
      if (cancelledRef.current) return;
      if (res.ok) {
        const blob = await res.blob();
        if (cancelledRef.current) return;
        const objectUrl = URL.createObjectURL(blob);
        objectUrlRef.current = objectUrl;
        const audio = new Audio(objectUrl);
        audioRef.current = audio;
        audio.onended = () => {
          if (!cancelledRef.current) {
            URL.revokeObjectURL(objectUrl);
            objectUrlRef.current = null;
            audioRef.current = null;
            release(stopRef.current);
            setStatus("idle");
          }
        };
        audio.onerror = () => playBrowserFallback(sample);
        claim(stopRef.current);
        setStatus("playing");
        await audio.play();
        return;
      }
    } catch {
      // Network failure → browser fallback below.
    }
    if (!cancelledRef.current) playBrowserFallback(sample);
  }

  const label =
    status === "playing"
      ? t(messages, "setup.voiceStop")
      : status === "loading"
        ? t(messages, "setup.voiceLoading")
        : t(messages, "setup.voicePreview");

  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        void toggle();
      }}
      aria-label={
        status === "idle"
          ? `Preview ${personaName} voice`
          : status === "loading"
            ? `Loading ${personaName} voice preview`
            : `Stop ${personaName} voice preview`
      }
      className={cn(
        "inline-flex items-center gap-1.5 rounded-[8px] border px-2.5 py-1.5 text-[12px] font-medium transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2",
        status === "idle"
          ? "border-line text-ink-soft hover:border-ink hover:text-ink"
          : "border-accent bg-accent-soft text-ink",
      )}
    >
      {status === "loading" ? (
        <Spinner className="h-3.5 w-3.5" label={label} />
      ) : status === "playing" ? (
        <Square className="h-3.5 w-3.5" aria-hidden />
      ) : (
        <Play className="h-3.5 w-3.5" aria-hidden />
      )}
      {label}
    </button>
  );
}
