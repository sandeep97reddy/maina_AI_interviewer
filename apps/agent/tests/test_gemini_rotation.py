"""Gemini API key rotation: parsing, round-robin, 429 failover.

Covers the prep/scoring path (``GeminiLLM`` adapter). The live worker path
(per-session key via ``settings.next_gemini_key()``) is a one-line pick
covered by the config tests below.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from deepinterview_agent.core.adapters import llm as llm_mod
from deepinterview_agent.core.adapters.llm import GeminiLLM, get_llm
from deepinterview_agent.core.config import Settings, mask_key


class _SmallSchema(BaseModel):
    answer: str


class _ScriptedClient:
    """Fake ``genai.Client``: script of per-call texts (or exceptions)."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.models_seen: list = []

    @property
    def aio(self):
        return self

    @property
    def models(self):
        return self

    async def generate_content(self, **kwargs):
        self.calls += 1
        self.models_seen.append(kwargs.get("model"))
        if not self._script:
            return SimpleNamespace(text="ok-default")
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return SimpleNamespace(text=action)


def _llm_with_clients(monkeypatch, mapping):
    """GeminiLLM whose _client_for returns scripted fakes per key."""
    llm = GeminiLLM(list(mapping), "gemini-3.6-flash", 5.0)
    clients = {key: _ScriptedClient(script) for key, script in mapping.items()}
    monkeypatch.setattr(llm, "_client_for", lambda key: clients[key])
    return llm, clients


def _no_sleep(monkeypatch):
    """Fail the test if the adapter sleeps (failover must be instant)."""
    sleeps: list = []

    async def _record(delay):
        sleeps.append(delay)
        raise AssertionError(f"adapter slept {delay}s during quota failover")

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _record)
    return sleeps


# --- config parsing ----------------------------------------------------------


def test_gemini_keys_parsing():
    assert Settings(gemini_api_key="a,b").gemini_keys == ["a", "b"]
    assert Settings(gemini_api_key="  a , b ,, ").gemini_keys == ["a", "b"]
    assert Settings(gemini_api_key=None).gemini_keys == []
    assert Settings(gemini_api_key="  ").gemini_keys == []
    assert Settings(gemini_api_key="solo").gemini_keys == ["solo"]


def test_next_gemini_key_round_robins_and_handles_empty():
    s = Settings(gemini_api_key="A,B")
    assert [s.next_gemini_key() for _ in range(4)] == ["A", "B", "A", "B"]
    assert Settings(gemini_api_key=None).next_gemini_key() is None
    assert Settings(gemini_api_key="solo").next_gemini_key() == "solo"


def test_mask_key_never_shows_full_key():
    masked = mask_key("AIzaSyAbC1234567890abcdef")
    assert masked == "AIzaSyAb...cdef"
    assert "1234567890" not in masked
    assert mask_key(None) == "<none>"
    assert mask_key("short") == "***"


def test_get_llm_accepts_comma_list_and_empty_still_mocks():
    llm = get_llm(Settings(llm_provider="gemini", gemini_api_key="k1,k2"))
    assert isinstance(llm, GeminiLLM)
    assert llm._api_keys == ["k1", "k2"]
    single = get_llm(Settings(llm_provider="gemini", gemini_api_key="k1"))
    assert isinstance(single, GeminiLLM) and single._api_keys == ["k1"]
    # Empty string is "missing" — same mock fallback as None before.
    assert not isinstance(
        get_llm(Settings(llm_provider="gemini", gemini_api_key="")), GeminiLLM
    )


# --- rotation behaviour ------------------------------------------------------


def test_round_robin_alternates_keys_across_calls(monkeypatch):
    llm, clients = _llm_with_clients(monkeypatch, {"A": ["a1", "a2"], "B": ["b1", "b2"]})
    out = asyncio.run(_collect(llm, 4))
    assert out == ["a1", "b1", "a2", "b2"]
    assert clients["A"].calls == 2
    assert clients["B"].calls == 2


def test_quota_failover_is_instant_and_uses_next_key(monkeypatch):
    _no_sleep(monkeypatch)
    err429 = RuntimeError("429 RESOURCE_EXHAUSTED: quota spent")
    llm, clients = _llm_with_clients(monkeypatch, {"A": [err429], "B": ["from-b"]})
    assert asyncio.run(llm.complete_text(system="s", user="u")) == "from-b"
    # Dead key tried exactly once — no sleep, no model-hop retries on it.
    assert clients["A"].calls == 1
    assert clients["B"].calls == 1


def test_all_keys_spent_raises_without_any_sleep(monkeypatch):
    _no_sleep(monkeypatch)
    err = RuntimeError("rate limit exceeded")
    llm, clients = _llm_with_clients(monkeypatch, {"A": [err], "B": [err]})
    with pytest.raises(RuntimeError, match="rate limit"):
        asyncio.run(llm.complete_text(system="s", user="u"))
    assert clients["A"].calls == 1
    assert clients["B"].calls == 1


def test_single_key_quota_raises_immediately_without_sleep(monkeypatch):
    """Deliberate difference from the old code: no 2s+4s model-hop sleeps on a
    key that is known-spent — the error surfaces at once for prep to handle."""
    _no_sleep(monkeypatch)
    llm, clients = _llm_with_clients(
        monkeypatch, {"solo": [RuntimeError("429 RESOURCE_EXHAUSTED")]}
    )
    with pytest.raises(RuntimeError, match="429"):
        asyncio.run(llm.complete_text(system="s", user="u"))
    assert clients["solo"].calls == 1


def test_transient_503_keeps_sleep_and_model_fallback(monkeypatch):
    """Non-quota transients keep the historic behaviour: sleep, next model."""
    sleeps: list = []

    async def _record(delay):
        sleeps.append(delay)

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _record)
    llm, clients = _llm_with_clients(
        monkeypatch, {"solo": [RuntimeError("503 UNAVAILABLE"), "recovered"]}
    )
    assert asyncio.run(llm.complete_text(system="s", user="u")) == "recovered"
    assert sleeps == [2.0]
    assert clients["solo"].models_seen[0] != clients["solo"].models_seen[1]


def test_complete_json_rotates_and_validates(monkeypatch):
    llm, clients = _llm_with_clients(
        monkeypatch,
        {"A": [RuntimeError("429 too many requests")], "B": ['{"answer": "yes"}']},
    )
    out = asyncio.run(
        llm.complete_json(system="s", user="u", schema=_SmallSchema)
    )
    assert out == _SmallSchema(answer="yes")
    assert clients["A"].calls == 1


def test_empty_keys_rejected():
    with pytest.raises(ValueError, match="at least one API key"):
        GeminiLLM([], "gemini-3.6-flash")


def test_live_worker_picks_rotating_key_per_session(monkeypatch):
    """The LiveKit plugin takes one key, so each interview session gets the
    next key — concurrent interviews spread across the pool."""
    from livekit.plugins import google

    from deepinterview_agent import worker

    seen: list = []

    class _FakeLLM:
        def __init__(self, model, api_key):
            seen.append(api_key)

    monkeypatch.setattr(google, "LLM", _FakeLLM)
    settings = Settings(llm_provider="gemini", gemini_api_key="K1,K2")
    for _ in range(3):
        worker.build_llm(settings)
    assert seen == ["K1", "K2", "K1"]


async def _collect(llm, n):
    return [await llm.complete_text(system="s", user="u") for _ in range(n)]
