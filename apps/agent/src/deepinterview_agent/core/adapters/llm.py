"""LLM adapter factory + real adapters (lazy-imported SDKs).

``get_llm(settings)`` returns the deterministic :class:`MockLLM` unless an LLM
provider is selected *and* its API key is present; otherwise it logs a warning
and falls back to the mock. Real adapters import their SDK inside methods so the
module imports cleanly with no SDK installed.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from typing import TYPE_CHECKING, Any

from ..config import mask_key
from ..logging import get_logger
from .base import LLMAdapter
from .mock import MockLLM

if TYPE_CHECKING:
    from ..config import Settings

log = get_logger(__name__)

# Per-call ceiling so a stalled provider call can never hang a pipeline forever
# (the prep graph's per-node try/except only fires once the call RETURNS).
_DEFAULT_TIMEOUT_SEC = 90.0


def _schema_prompt(system: str, schema: type) -> str:
    """Append the JSON Schema to the system prompt (provider-agnostic).

    Both Gemini's ``response_schema`` and OpenAI's strict structured outputs
    reject free-form maps like our ``LocalizedText = dict[str, str]``
    (additionalProperties), so the reliable cross-provider pattern is JSON mode
    + the schema in the prompt + Pydantic validation of the result.
    """
    schema_json = json.dumps(schema.model_json_schema())
    return (
        f"{system}\n\nReturn ONLY a single JSON value that conforms to this "
        f"JSON Schema (no markdown, no commentary):\n{schema_json}"
    )


def _loads_json(text: str) -> Any:
    """Parse JSON from an LLM response, tolerating ```json fences, stray prose, and
    *trailing* "Extra data".

    The keystone failure this fixes: the question planner has the largest schema,
    so its real Gemini response is the biggest — and a complete JSON object was
    sometimes FOLLOWED by extra content (a second object / trailing commentary).
    Plain ``json.loads`` raises ``Extra data`` on that, and the old greedy
    ``\\{.*\\}`` fallback spanned to the LAST brace (swallowing the trailing junk
    into invalid JSON), so every plan silently fell back to the generic mock —
    an interview that asks one question literally titled "mock". ``raw_decode``
    returns the FIRST complete JSON value and ignores anything after it.
    """
    import re

    t = (text or "").strip()
    # Strip reasoning blocks BEFORE looking for JSON. Local reasoning models
    # (Qwen3 via Ollama) routinely return "<think>…</think>" inline in content,
    # and the tolerant scan below takes the FIRST '{' it sees — so a brace
    # anywhere in the model's scratchpad would be decoded instead of the answer.
    # Cloud models don't emit these tags, so this is a no-op for them.
    t = re.sub(r"<(think|thinking)>.*?</\1>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    # An unterminated block means the reply was cut off mid-thought; everything
    # after the opening tag is scratchpad, never the answer.
    t = re.sub(r"<(think|thinking)>.*\Z", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    # Strip a wrapping ```json ... ``` / ``` ... ``` markdown fence, if present.
    if t.startswith("```"):
        t = re.sub(r"^```[^\n]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t).strip()
    # Fast path: a single clean JSON value.
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Tolerant path: decode the first complete JSON value starting at the first
    # '{' or '[', ignoring any trailing data (raw_decode is "Extra data"-safe).
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _end = decoder.raw_decode(t, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("no JSON value found in LLM response", t, 0)


class GeminiLLM:
    """Google Gemini via ``google-genai`` (lazy import), with key rotation.

    ``api_keys`` is one key or a list (comma-separated ``GEMINI_API_KEY`` is
    split by ``Settings.gemini_keys``): consecutive calls round-robin across
    keys, and a quota error (429) fails over to the next key with NO sleep.
    Clients are cached per key — ``genai.Client`` construction is not free.
    """

    def __init__(
        self,
        api_keys: str | list[str],
        model: str,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        # No default model: ids retire fast, so the current id lives in ONE
        # place (Settings.gemini_model) and must be passed in explicitly.
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        keys = [k.strip() for k in (api_keys or []) if k and k.strip()]
        if not keys:
            raise ValueError("GeminiLLM needs at least one API key")
        self._api_keys = keys
        self._model = model
        self._timeout = timeout_sec
        self._clients: dict[str, Any] = {}
        self._rr = itertools.count()

    def _model_chain(self) -> list[str]:
        candidates = [self._model, "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
        seen: set[str] = set()
        res: list[str] = []
        for m in candidates:
            if m and m not in seen:
                seen.add(m)
                res.append(m)
        return res

    def _is_quota_error(self, exc: Exception) -> bool:
        """429 / quota-exhausted: the KEY is spent, not the model — fail over to
        the next key immediately (no sleep, no model hop on the dead key)."""
        msg = str(exc)
        low = msg.lower()
        return (
            "429" in msg
            or "RESOURCE_EXHAUSTED" in msg
            or "resource exhausted" in low
            or "rate limit" in low
            or "rate_limit" in low
            or "quota" in low
        )

    def _is_retryable_gemini_error(self, exc: Exception) -> bool:
        msg = str(exc)
        low = msg.lower()
        return any(k in msg for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")) or any(
            k in low for k in ("quota", "demand", "rate limit", "rate_limit", "resource exhausted")
        )

    def _client_for(self, key: str) -> Any:
        """Cached ``genai.Client`` for a key (built once, reused every call)."""
        client = self._clients.get(key)
        if client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on optional SDK
                raise RuntimeError(
                    "google-genai is not installed; install the 'gemini' extra."
                ) from exc
            client = genai.Client(api_key=key)
            self._clients[key] = client
        return client

    def _build_config(self, system: str, thinking_budget: int | None) -> dict:
        """Build a generate_content config, attaching thinking_config safely.

        Unknown/retired model ids may reject thinking_config — callers catch
        that per attempt and retry without it, so a budget never hard-fails
        prep/post.
        """
        config: dict[str, Any] = {"system_instruction": system}
        if thinking_budget is not None:
            try:
                from google.genai import types

                config["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
            except Exception:
                # SDK without ThinkingConfig (or import failure) -> plain config.
                pass
        return config

    async def _generate_text(
        self,
        client: Any,
        *,
        system: str,
        user: str,
        thinking_budget: int | None = None,
    ) -> str:
        """One key's attempt: the model fallback chain, exactly as before.

        Quota errors escape IMMEDIATELY (the outer loop fails over to the next
        key with no sleep); other transient errors keep the historic
        sleep-and-next-model behaviour.
        """
        models = self._model_chain()
        max_attempts = min(3, len(models))
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            model = models[attempt]
            try:
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model,
                        contents=user,
                        config=self._build_config(system, thinking_budget),
                    ),
                    timeout=self._timeout,
                )
                return resp.text or ""
            except Exception as exc:
                last_exc = exc
                if self._is_quota_error(exc):
                    raise
                if attempt < max_attempts - 1 and self._is_retryable_gemini_error(exc):
                    delay = 2.0 * (attempt + 1)
                    next_model = models[attempt + 1]
                    log.warning(
                        "Gemini transient error on attempt %d with %s (%s); retrying with %s in %.1fs...",
                        attempt + 1,
                        model,
                        exc,
                        next_model,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        if last_exc:
            raise last_exc
        return ""

    async def _generate_json(
        self,
        client: Any,
        *,
        system: str,
        user: str,
        schema: type,
        thinking_budget: int | None = None,
    ) -> Any:
        """One key's JSON attempt — same quota-immediate / transient-sleep split."""
        models = self._model_chain()
        max_attempts = min(3, len(models))
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            model = models[attempt]
            try:
                base = self._build_config(
                    _schema_prompt(system, schema), thinking_budget
                )
                base["response_mime_type"] = "application/json"
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model,
                        contents=user,
                        config=base,
                    ),
                    timeout=self._timeout,
                )
                return schema.model_validate(_loads_json(resp.text or "{}"))
            except Exception as exc:
                last_exc = exc
                if self._is_quota_error(exc):
                    raise
                if attempt < max_attempts - 1 and self._is_retryable_gemini_error(exc):
                    delay = 2.0 * (attempt + 1)
                    next_model = models[attempt + 1]
                    log.warning(
                        "Gemini transient error on attempt %d with %s (%s); retrying with %s in %.1fs...",
                        attempt + 1,
                        model,
                        exc,
                        next_model,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        if last_exc:
            raise last_exc
        return None

    async def complete_text(
        self, *, system: str, user: str, thinking_budget: int | None = None
    ) -> str:
        """Round-robin across keys; quota failure jumps to the next key at once."""
        n = len(self._api_keys)
        start = next(self._rr) % n
        last_exc: Exception | None = None
        for ki in range(n):
            key = self._api_keys[(start + ki) % n]
            if ki > 0:
                log.warning(
                    "Gemini quota exhausted; failing over to key %s (no delay)...",
                    mask_key(key),
                )
            try:
                return await self._generate_text(
                    self._client_for(key),
                    system=system,
                    user=user,
                    thinking_budget=thinking_budget,
                )
            except Exception as exc:
                last_exc = exc
                if self._is_quota_error(exc) and ki < n - 1:
                    continue
                raise
        if last_exc:  # pragma: no cover - loop always returns or raises
            raise last_exc
        return ""

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type,
        thinking_budget: int | None = None,
    ) -> Any:
        # JSON mode + schema-in-prompt (see _schema_prompt for why not
        # ``response_schema``).
        n = len(self._api_keys)
        start = next(self._rr) % n
        last_exc: Exception | None = None
        for ki in range(n):
            key = self._api_keys[(start + ki) % n]
            if ki > 0:
                log.warning(
                    "Gemini quota exhausted; failing over to key %s (no delay)...",
                    mask_key(key),
                )
            try:
                return await self._generate_json(
                    self._client_for(key),
                    system=system,
                    user=user,
                    schema=schema,
                    thinking_budget=thinking_budget,
                )
            except Exception as exc:
                last_exc = exc
                if self._is_quota_error(exc) and ki < n - 1:
                    continue
                raise
        return None


class OpenAILLM:
    """OpenAI — and any OpenAI-compatible server — via the ``openai`` SDK.

    ``base_url`` is what makes the local path possible: pointed at Ollama's
    ``/v1`` it drives the whole prep/post pipeline with no cloud key. See
    :class:`OllamaLLM`.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
        base_url: str | None = None,
    ) -> None:
        # No default model — see GeminiLLM.__init__; Settings.openai_model is
        # the single source of truth.
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_sec
        self._base_url = base_url

    def _client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional SDK
            raise RuntimeError(
                "openai is not installed; install the 'openai' extra."
            ) from exc
        return AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    async def complete_text(
        self, *, system: str, user: str, thinking_budget: int | None = None
    ) -> str:
        _ = thinking_budget  # OpenAI-compatible path ignores Gemini thinking.
        client = self._client()
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ),
            timeout=self._timeout,
        )
        return resp.choices[0].message.content or ""

    async def complete_json(
        self, *, system: str, user: str, schema: type, thinking_budget: int | None = None
    ) -> Any:
        # JSON mode + schema-in-prompt, NOT strict structured outputs: OpenAI's
        # strict mode rejects free-form maps (additionalProperties) like our
        # ``LocalizedText = dict[str, str]``, which would silently break the
        # question planner (every plan falling back to mock). Same pattern as
        # the Gemini adapter; Pydantic validates the result either way.
        client = self._client()
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _schema_prompt(system, schema)},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            ),
            timeout=self._timeout,
        )
        return schema.model_validate(_loads_json(resp.choices[0].message.content or "{}"))


class OllamaLLM(OpenAILLM):
    """A local Ollama server through its OpenAI-compatible ``/v1`` endpoint.

    Ollama needs no credential, so a non-empty placeholder key is passed (the
    SDK requires *something*). Behaviourally this differs from the cloud path in
    one way that matters: **it retries once when the response doesn't parse.**

    Why only here. Small local models are markedly worse at emitting strict JSON
    than Gemini/GPT, and this pipeline's largest schema (``QuestionPlan``) is
    also its keystone — when it fails, ``prep.nodes.question_planner`` silently
    swaps in the generic mock plan and the candidate sits through an interview
    whose questions are titled "mock". That exact failure has shipped before.
    One cheap retry with a blunter instruction converts most near-misses
    (a stray preamble, a truncated trailing brace) into a usable plan; the cloud
    adapters stay single-shot so their latency and cost are unchanged.
    """

    _RETRY_NUDGE = (
        "Your previous reply could not be parsed as JSON. Reply with the JSON "
        "value ONLY — no explanation, no markdown fence, no <think> block."
    )

    async def complete_json(
        self, *, system: str, user: str, schema: type, thinking_budget: int | None = None
    ) -> Any:
        _ = thinking_budget  # Local path ignores Gemini thinking.
        try:
            return await super().complete_json(system=system, user=user, schema=schema)
        except Exception as exc:  # noqa: BLE001 - any parse/validation miss earns one retry
            log.warning("OllamaLLM: unparseable JSON (%s); retrying once.", exc)
            return await super().complete_json(
                system=f"{system}\n\n{self._RETRY_NUDGE}", user=user, schema=schema
            )


def get_llm(settings: Settings) -> LLMAdapter:
    """Choose an LLM adapter from settings, falling back to the mock."""
    provider = (settings.llm_provider or "mock").lower()
    if provider == "mock":
        return MockLLM()
    timeout = getattr(settings, "llm_call_timeout_sec", _DEFAULT_TIMEOUT_SEC)
    if provider == "gemini":
        keys = settings.gemini_keys
        if keys:
            return GeminiLLM(keys, settings.gemini_model, timeout)
        log.warning("llm_provider=gemini but gemini_api_key is missing; using MockLLM.")
        return MockLLM()
    if provider == "openai":
        if settings.openai_api_key:
            return OpenAILLM(settings.openai_api_key, settings.openai_model, timeout)
        log.warning("llm_provider=openai but openai_api_key is missing; using MockLLM.")
        return MockLLM()
    if provider in {"ollama", "vllm", "llamacpp", "lmstudio", "local"}:
        # Local: a base URL takes the place of an API key. Everything else in
        # the factory contract is unchanged — missing config still degrades to
        # the mock rather than failing the pipeline.
        if settings.ollama_base_url:
            return OllamaLLM(
                settings.local_api_key,
                settings.ollama_model,
                timeout,
                base_url=settings.ollama_base_url,
            )
        log.warning("llm_provider=%s but ollama_base_url is missing; using MockLLM.", provider)
        return MockLLM()
    log.warning("Unknown llm_provider=%r; using MockLLM.", provider)
    return MockLLM()
