"use client";

import { useState } from "react";
import { getMessages, type Locale, type Messages } from "@/lib/i18n";

/**
 * Client-side message resolution. English-only product: always resolve the EN
 * pack, ignoring any stale `locale` cookie a previous version may have left.
 * (Server-side `getMessages` already falls back to EN; this closes the client
 * path, including the `vi` overlay that used to live here.)
 */
function resolve(): Messages {
  return getMessages("en");
}

/**
 * Resolve messages on the client. English-only product: a constant EN pack —
 * no cookie reads, so first paint and post-mount render are identical (no
 * hydration drift by construction).
 */
export function useMessages(): Messages {
  const [messages] = useState<Messages>(() => resolve());
  return messages;
}

/**
 * The active locale. English-only product: always "en" (kept as a hook so
 * consumers like the study coach need no changes).
 */
export function useLocale(): Locale {
  return "en";
}
