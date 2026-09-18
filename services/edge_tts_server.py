"""Microsoft Edge Neural Voice bridge (TTS-only).

OpenAI-compatible ``/v1/audio/speech`` shim so the LiveKit worker's existing
local-TTS path (``TTS_PROVIDER=kokoro`` -> ``livekit.plugins.openai.TTS`` at
``KOKORO_BASE_URL``) speaks with free Edge voices — no worker code changes,
no Cartesia/ElevenLabs keys, no Docker.

Contracts honored (see apps/agent/src/deepinterview_agent/worker.py):
  * Startup probe: ``GET {KOKORO_BASE_URL}/models`` must be reachable.
    ``_unreachable_local_providers`` treats only transport failures as fatal,
    but a 200 with the ``tts-1`` id keeps the audio-branch guard quiet.
  * Synthesis: ``POST {KOKORO_BASE_URL}/audio/speech`` with the OpenAI speech
    shape ``{model, input, voice, response_format, speed?}``. ``KOKORO_MODEL``
    must stay ``tts-1`` (selects the plugin's raw-audio transport, not SSE).
  * Voice: ``KOKORO_VOICE`` (e.g. ``en-US-JennyNeural``) arrives as ``voice``.
    Kokoro-style ids (``af_heart``, ...) fall back to the default Edge voice
    so a blank ``KOKORO_VOICE`` still produces audio.

Run:  ``python services/edge_tts_server.py``  (listens on 127.0.0.1:8880)
Env:  ``EDGE_TTS_VOICE`` overrides the default voice; ``EDGE_TTS_PORT`` the port.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("edge_tts_bridge")

DEFAULT_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")
PORT = int(os.getenv("EDGE_TTS_PORT", "8880"))

app = FastAPI(title="DeepInterview Edge-TTS Bridge")


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(default="", description="Text to speak (OpenAI speech shape)")
    voice: str = DEFAULT_VOICE
    response_format: str = "mp3"
    speed: float = 1.0


def _resolve_voice(requested: str | None) -> str:
    """Map the worker's ``voice`` to an Edge voice id.

    Edge ids look like ``en-US-JennyNeural`` (locale prefix + ``Neural``).
    Anything else (Kokoro ids such as ``af_heart`` when ``KOKORO_VOICE`` is
    blank) falls back to the configured default so synthesis never 400s.
    """
    voice = (requested or "").strip() or DEFAULT_VOICE
    if "-" not in voice or "Neural" not in voice:
        log.warning("edge-tts: unknown voice %r; using default %r", requested, DEFAULT_VOICE)
        return DEFAULT_VOICE
    return voice


@app.get("/")
@app.get("/v1")
@app.get("/v1/")
@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/v1/models")
async def list_models() -> dict:
    """Satisfy the worker's startup reachability probe + model-id guard."""
    return {
        "object": "list",
        "data": [
            {"id": "tts-1", "object": "model", "created": 0, "owned_by": "edge-tts-bridge"},
            {"id": "tts-1-hd", "object": "model", "created": 0, "owned_by": "edge-tts-bridge"},
        ],
    }


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest) -> Response:
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'input' must be non-empty text")
    voice = _resolve_voice(req.voice)

    try:
        import edge_tts
    except ImportError as exc:
        raise HTTPException(
            status_code=500, detail="edge-tts not installed (pip/uv install edge-tts)"
        ) from exc

    try:
        communicate = edge_tts.Communicate(text, voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                data = chunk.get("data")
                if data:
                    chunks.append(data)
    except Exception as exc:  # noqa: BLE001 - surface Edge failures as 502
        log.exception("edge-tts synthesis failed (voice=%s)", voice)
        raise HTTPException(status_code=502, detail=f"Edge TTS failed: {exc}") from exc

    audio = b"".join(chunks)
    if not audio:
        raise HTTPException(status_code=502, detail="Edge TTS returned no audio")
    return Response(content=audio, media_type="audio/mpeg")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
