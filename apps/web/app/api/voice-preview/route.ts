import { PERSONAS } from "@/lib/personas";

/**
 * GET /api/voice-preview?voice=<Edge voice id>&text=<optional>
 *
 * Proxies a short synthesis to the local Edge-TTS bridge
 * (`services/edge_tts_server.py`, same OpenAI-compatible shape the LiveKit
 * worker uses via `KOKORO_BASE_URL`) so /setup can play the EXACT interview
 * voice before the interview. No keys needed — the bridge speaks with free
 * Edge Neural voices.
 *
 * When the bridge isn't running (default for most users), this returns 503
 * `{ error: "preview-unavailable" }` and the client falls back to the
 * browser's built-in `speechSynthesis` so preview still works offline.
 */

export const dynamic = "force-dynamic";

const MAX_TEXT_CHARS = 280;
const UPSTREAM_TIMEOUT_MS = 8000;

const ALLOWED_VOICES = new Set(PERSONAS.map((p) => p.edgeVoice));

function bridgeSpeechUrl(): string {
  const raw = (
    process.env.EDGE_TTS_URL ??
    process.env.KOKORO_BASE_URL ??
    "http://127.0.0.1:8880"
  ).trim().replace(/\/+$/, "");
  // KOKORO_BASE_URL already includes /v1 (…:8880/v1) → append /audio/speech;
  // a bare host (…:8880) → append /v1/audio/speech.
  return raw.endsWith("/v1") ? `${raw}/audio/speech` : `${raw}/v1/audio/speech`;
}

function defaultLine(voice: string): string {
  const persona = PERSONAS.find((p) => p.edgeVoice === voice);
  const name = persona?.name ?? "your interviewer";
  return `Hi, I'm ${name}. I'll be guiding your interview today. How does my voice sound?`;
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const voice = (url.searchParams.get("voice") ?? "").trim();
  if (!ALLOWED_VOICES.has(voice)) {
    return Response.json({ error: "unknown-voice" }, { status: 400 });
  }

  let text = (url.searchParams.get("text") ?? "").trim();
  if (!text) text = defaultLine(voice);
  if (text.length > MAX_TEXT_CHARS) text = text.slice(0, MAX_TEXT_CHARS);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const upstream = await fetch(bridgeSpeechUrl(), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "tts-1",
        input: text,
        voice,
        response_format: "mp3",
      }),
      signal: controller.signal,
    });
    if (!upstream.ok) {
      return Response.json(
        { error: "preview-unavailable" },
        { status: 503 },
      );
    }
    const buf = await upstream.arrayBuffer();
    if (buf.byteLength === 0) {
      return Response.json(
        { error: "preview-unavailable" },
        { status: 503 },
      );
    }
    return new Response(buf, {
      status: 200,
      headers: {
        "content-type":
          upstream.headers.get("content-type") ?? "audio/mpeg",
        // Voice + text determine the bytes — safe to cache for a day.
        "cache-control": "public, max-age=86400",
      },
    });
  } catch {
    return Response.json({ error: "preview-unavailable" }, { status: 503 });
  } finally {
    clearTimeout(timer);
  }
}
