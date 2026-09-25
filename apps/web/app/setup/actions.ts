"use server";

import type { InterviewContext, PrepRequest } from "@deepinterview/shared";
import { features } from "@deepinterview/ee";
import { requestNewQuestions, requestPrep, requestSessionFromContext } from "@/lib/api";
import { getUser } from "@/lib/supabase/server";
import { isSupabaseConfigured } from "@/lib/env";

export type StartSessionResult =
  | { ok: true; session_id: string }
  // `error` stays required (existing consumers read `result.error`); `reason`
  // is an additive discriminator the client uses to route to /login
  // (auth_required). OSS is billing-free — there is no interview cap.
  | {
      ok: false;
      error: string;
      reason?: "auth_required";
    };

/**
 * Kick off the prep pipeline and return the new session id. We return
 * `{session_id}` (rather than redirect()) so the client owns navigation via
 * router.push — this also keeps the Next 15 redirect-in-action pitfall out of
 * the picture.
 *
 * OSS is self-host, bring-your-own-keys, and UNCAPPED: there is no billing and
 * no per-tier interview limit (payments/gating live in the private cloud fork).
 * When Supabase is configured we resolve the signed-in user so the agent can
 * stamp `sessions.user_id`; an anonymous user simply proceeds. A distribution
 * that requires auth (the ee `features.auth` seam, off in OSS) fails closed.
 */
export async function startSession(
  input: PrepRequest,
): Promise<StartSessionResult> {
  let userId: string | null = null;

  if (isSupabaseConfigured()) {
    const user = await getUser();
    if (user) userId = user.id;
  }

  if (!userId && features.auth) {
    // Distribution gate (no-op in OSS): server actions are public POST
    // endpoints, so a required-auth distribution must fail closed here even
    // though its proxy already redirects anonymous visitors off /setup.
    // Deliberately independent of Supabase config: a missing/broken env must
    // never let anonymous callers create sessions in such a build.
    return {
      ok: false,
      error: "Sign in to start an interview.",
      reason: "auth_required",
    };
  }

  try {
    // Forward the signed-in user's id so the agent stamps it on the `sessions`
    // row (sessions.user_id). Without this the row is unowned (NULL) and the
    // report's RLS read (auth.uid() = user_id) can never see it, so the page
    // falls back to sample data. Anonymous/dev has no user → omit it.
    const { session_id } = await requestPrep({
      ...input,
      user_id: userId ?? undefined,
    });
    return { ok: true, session_id };
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Could not reach the prep service.";
    return { ok: false, error: message };
  }
}

/**
 * Instant reuse: build a fresh ready session from a cached InterviewContext
 * (0 LLM calls). The agent clones the context under a new session id with
 * cursor/answers/scorecard reset. Same auth gating as startSession.
 */
export async function startSessionFromContext(
  context: InterviewContext,
): Promise<StartSessionResult> {
  let userId: string | null = null;

  if (isSupabaseConfigured()) {
    const user = await getUser();
    if (user) userId = user.id;
  }

  if (!userId && features.auth) {
    return {
      ok: false,
      error: "Sign in to start an interview.",
      reason: "auth_required",
    };
  }

  try {
    const { session_id } = await requestSessionFromContext(context, userId ?? undefined);
    return { ok: true, session_id };
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Could not reuse the cached context.";
    return { ok: false, error: message };
  }
}

/**
 * Fresh questions: new ready session from a past session's saved analysis
 * (1 planner LLM call, no CV/JD/company re-analysis). Same auth gating.
 */
export async function refreshQuestions(
  sessionId: string,
): Promise<StartSessionResult> {
  let userId: string | null = null;

  if (isSupabaseConfigured()) {
    const user = await getUser();
    if (user) userId = user.id;
  }

  if (!userId && features.auth) {
    return {
      ok: false,
      error: "Sign in to start an interview.",
      reason: "auth_required",
    };
  }

  try {
    const { session_id } = await requestNewQuestions(sessionId);
    return { ok: true, session_id };
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Could not generate fresh questions.";
    return { ok: false, error: message };
  }
}
