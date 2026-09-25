"use client";

import {
  InterviewContextSchema,
  type InterviewContext,
} from "@deepinterview/shared";

const KEY = "di.cached.context.v1";
// 30-day reuse window + 4.5MB guard (localStorage ~5MB quota).
const MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;
const MAX_BYTES = 4_500_000;

export type CachedContextEntry = {
  savedAt: string;
  context: InterviewContext;
};

function byteLen(s: string): number {
  try {
    return new Blob([s]).size;
  } catch {
    return s.length;
  }
}

export function loadCachedContext(): CachedContextEntry | null {
  try {
    if (typeof window === "undefined" || !window.localStorage) return null;
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    if (byteLen(raw) > MAX_BYTES) {
      localStorage.removeItem(KEY);
      return null;
    }
    const parsed = JSON.parse(raw) as { savedAt?: unknown; context?: unknown };
    if (typeof parsed.savedAt !== "string" || !parsed.context) return null;
    const age = Date.now() - Date.parse(parsed.savedAt);
    if (!Number.isFinite(age) || age < 0 || age > MAX_AGE_MS) {
      localStorage.removeItem(KEY);
      return null;
    }
    const context = InterviewContextSchema.parse(parsed.context);
    return { savedAt: parsed.savedAt, context };
  } catch {
    try {
      if (typeof window !== "undefined" && window.localStorage) {
        localStorage.removeItem(KEY);
      }
    } catch {
      // storage failure must never break setup
    }
    return null;
  }
}

export function saveCachedContext(context: InterviewContext): void {
  try {
    if (typeof window === "undefined" || !window.localStorage) return;
    const payload = JSON.stringify({
      savedAt: new Date().toISOString(),
      context,
    });
    if (byteLen(payload) > MAX_BYTES) return;
    // Validate before persisting so a corrupt context never poisons reuse.
    InterviewContextSchema.parse(context);
    localStorage.setItem(KEY, payload);
  } catch {
    // Best-effort: quota or serialization failure must never block the UI.
  }
}

export function clearCachedContext(): void {
  try {
    if (typeof window === "undefined" || !window.localStorage) return;
    localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}

export function describeCachedContext(entry: CachedContextEntry): string {
  const job = entry.context.job;
  const title = job.title || "Interview";
  const company = job.company_name ? ` at ${job.company_name}` : "";
  const n = entry.context.plan.questions.length;
  return `${title}${company} (${n} questions)`;
}
