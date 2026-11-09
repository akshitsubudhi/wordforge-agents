"""
================================================================================
BLOCK: LLM Provider Layer (Anthropic / OpenAI / deterministic Mock)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : runner.AgentRunner passes a provider-neutral transcript
             (list[Message]) + JSON-schema tool specs.
  Downstream: returns LLMResponse(text, tool_calls, usage, model).
  Contract : complete() is a pure READ (idempotent) -> safe to retry.

Key engineering mechanism
  * MockLLM(mode="naive")   - the WORST-CASE model: obeys every
    `CALL tool {json}` directive it finds ANYWHERE in the transcript, including
    untrusted tool output, and "reads" obfuscations (NFKC, homoglyphs,
    zero-width, Unicode tags, base64, ROT13, letter-spacing) the way a real LLM
    would. It ignores spotlight boundaries. If the harness stops attacks with
    THIS model, it stops them independent of model alignment.
  * MockLLM(mode="compliant") - normal deterministic agent runs: follows only
    directives in trusted (system / trusted-user) messages.
  * Stateless: which directive to issue next is derived from the transcript
    (already-issued calls are counted), so every run is replayable bit-for-bit.
  * Real providers via raw httpx (no SDK lock-in); transient errors (429/5xx/
    timeouts) raise TransientLLMError so resilience.retry_call can back off.
  * get_llm(): ANTHROPIC_API_KEY -> Anthropic, OPENAI_API_KEY -> OpenAI, else
    Mock; real providers are wrapped in FallbackLLM(circuit breaker -> mock).

Failure modes & 1-step mitigation
  * Provider outage / rate limit -> retry w/ full jitter, then breaker opens
    and FallbackLLM serves the deterministic mock (degraded, never down).
  * Unknown model price -> priced at a deliberately HIGH default so the cost
    circuit breaker errs on the side of stopping early.
================================================================================
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Protocol

from core.harness import textnorm as tn

log = logging.getLogger("liveops.llm")

Role = Literal["system", "user", "assistant", "tool"]

# USD per 1M tokens (input, output). Unknown models fall back to EXPENSIVE.
PRICING: dict[str, tuple[float, float]] = {
    "mock-naive": (0.15, 0.60),
    "mock-compliant": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-5-5": (1.00, 5.00),
    "claude-sonnet-5-5": (3.00, 15.00),
}
UNKNOWN_MODEL_PRICE = (15.0, 75.0)


def price_for(model: str) -> tuple[float, float]:
    return PRICING.get(model, UNKNOWN_MODEL_PRICE)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    trusted: bool = True


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str


class TransientLLMError(RuntimeError):
    """429 / 5xx / network timeout - retryable."""


class PermanentLLMError(RuntimeError):
    """4xx (bad request, auth) - do not retry."""


class QuotaExhaustedLLMError(PermanentLLMError):
    """HTTP 429 insufficient_quota / billing: the account is out of credits. Not a rate limit, never retried."""


class AuthLLMError(PermanentLLMError):
    """HTTP 401 / 403: the API key is missing, wrong or lacks access."""


# ============================================================================ provider health (for operators)
class LLMHealth:
    """Process-wide, thread-safe record of how the configured provider is behaving. Written by the ONE call path
    (runner.guarded_completion) and by FallbackLLM; read by GET /api/llm/status and the Control Center banner, so a
    dead key or an empty balance is a visible, actionable state instead of a silent downgrade to the mock."""

    ACTIONS = {
        "ok": "",
        "mock": "",
        "quota_exhausted": "Add credits to the provider account, or set LLM_PROVIDER=mock to run on the deterministic "
                           "mock on purpose. Until then agents keep working on the deterministic fallback.",
        "auth_failed": "Check ANTHROPIC_API_KEY / OPENAI_API_KEY (wrong, revoked or no access to the model), or set "
                       "LLM_PROVIDER=mock. Until then agents keep working on the deterministic fallback.",
        "degraded": "The provider is failing or rate-limiting; calls are retried and the circuit breaker serves the "
                    "deterministic mock until it recovers. No action needed unless it persists.",
        "error": "The provider rejected the request (see last_error). Check LLM_MODEL and the request shape.",
    }

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", _NullLock()):
            self.state, self.last_error, self.last_error_at, self.last_ok_at = "ok", "", None, None
            self.calls = Counter()

    def record_ok(self, model: str, served_by: str) -> None:
        import time

        with self._lock:
            self.calls["ok" if served_by == model else "fallback"] += 1
            if served_by == model:
                self.state, self.last_ok_at = "ok", time.time()
            elif self.state == "ok":
                self.state = "degraded"

    def record_error(self, exc: BaseException) -> str:
        import time

        state = ("quota_exhausted" if isinstance(exc, QuotaExhaustedLLMError) else
                 "auth_failed" if isinstance(exc, AuthLLMError) else
                 "error" if isinstance(exc, PermanentLLMError) else "degraded")
        with self._lock:
            self.calls[state] += 1
            # a quota / auth verdict is sticky until a real answer arrives: later breaker noise must not hide it
            if self.state in ("quota_exhausted", "auth_failed") and state == "degraded":
                return state
            self.state = state
            self.last_error, self.last_error_at = f"{type(exc).__name__}: {str(exc)[:300]}", time.time()
        return state

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "last_error": self.last_error, "last_error_at": self.last_error_at,
                    "last_ok_at": self.last_ok_at, "calls": dict(self.calls)}


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


HEALTH = LLMHealth()


def llm_status() -> dict:
    """What the operator needs to know: which provider is configured, is it answering, and what to do if not."""
    from core.config import get_settings

    cfg = get_settings()
    if cfg.llm_provider == "mock" or not (cfg.anthropic_api_key or cfg.openai_api_key):
        why = "LLM_PROVIDER=mock" if cfg.llm_provider == "mock" else "no ANTHROPIC_API_KEY / OPENAI_API_KEY set"
        return {"mode": "mock", "provider": "mock", "model": "mock-compliant", "state": "mock", "ok": True,
                "message": f"Deterministic mock LLM ({why}).", "action": "", "last_error": "", "calls": {}}
    provider = "anthropic" if cfg.anthropic_api_key else "openai"
    model = cfg.llm_model or ("claude-sonnet-5-5" if provider == "anthropic" else "gpt-4.1-mini")
    h = HEALTH.snapshot()
    msg = {"ok": f"{provider} {model} is answering." if h["last_ok_at"] else f"{provider} {model} configured; no call made yet.",
           "quota_exhausted": f"{provider} reports the account is out of credits (HTTP 429 insufficient_quota).",
           "auth_failed": f"{provider} rejected the API key (HTTP 401/403).",
           "degraded": f"{provider} {model} is failing or rate-limited; the deterministic fallback is answering.",
           "error": f"{provider} rejected the request."}[h["state"]]
    return {"mode": "live", "provider": provider, "model": model, "state": h["state"], "ok": h["state"] == "ok",
            "message": msg, "action": LLMHealth.ACTIONS[h["state"]], "last_error": h["last_error"],
            "last_error_at": h["last_error_at"], "last_ok_at": h["last_ok_at"], "calls": h["calls"]}


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[Message], tools: list[dict], max_output_tokens: int = 1024) -> LLMResponse: ...


def transcript_tokens(messages: list[Message], tools: list[dict]) -> int:
    body = sum(len(m.content) + sum(len(json.dumps(tc.arguments)) for tc in m.tool_calls) for m in messages)
    return estimate_tokens("x" * (body + len(json.dumps(tools))))


# ============================================================================ mock
class MockLLM:
    """Deterministic, offline, replayable. See module header for semantics."""

    def __init__(self, mode: Literal["naive", "compliant"] = "compliant"):
        self.mode = mode
        self.model = f"mock-{mode}"

    @staticmethod
    def _comprehend(text: str) -> list[str]:
        """Every reading of `text` a capable LLM could act on."""
        base = tn.decode_tags(text)
        base = tn.normalize(base)[0]
        base = base.translate(str.maketrans(tn.CONFUSABLES))
        views = [base, tn.rot13(base), tn.unletterspace(base)]
        for m in tn.B64_RUN_RE.finditer(base):
            dec = tn.try_b64_text(m.group())
            if dec:
                views.append(dec)
        return views

    def _directives(self, messages: list[Message]) -> list[tuple[str, dict]]:
        found: list[tuple[str, dict]] = []
        for m in messages:
            if m.role == "assistant":
                continue
            if self.mode == "compliant" and (not m.trusted or m.role == "tool"):
                continue
            seen_here = set()
            for view in self._comprehend(m.content):
                for _, _, name, args in tn.find_directives(view):
                    key = tn.canonical(name, args)
                    if key not in seen_here:
                        seen_here.add(key)
                        found.append((name, args))
        return found

    def complete(self, messages: list[Message], tools: list[dict], max_output_tokens: int = 1024) -> LLMResponse:
        issued = Counter(tn.canonical(tc.name, tc.arguments) for m in messages if m.role == "assistant" for tc in m.tool_calls)
        wanted = Counter()
        nxt = None
        for name, args in self._directives(messages):
            key = tn.canonical(name, args)
            wanted[key] += 1
            if wanted[key] > issued[key]:
                nxt = (name, args)
                break
        in_tok = transcript_tokens(messages, tools)
        if nxt:
            call = ToolCall(id=f"call_{uuid.uuid5(uuid.NAMESPACE_OID, tn.canonical(*nxt) + str(len(messages))).hex[:12]}",
                            name=nxt[0], arguments=nxt[1])
            return LLMResponse("", [call], Usage(in_tok, estimate_tokens(json.dumps(nxt[1])) + 8), self.model)
        tool_msgs = [m for m in messages if m.role == "tool"]
        called = [tc.name for m in messages if m.role == "assistant" for tc in m.tool_calls]
        last = tool_msgs[-1].content[:600] if tool_msgs else ""
        if self.mode == "naive":  # parrots untrusted content -> output filter must matter
            text = f"Done. Here is what I found:\n{last}"
        else:
            text = f"Completed {len(called)} tool call(s): {', '.join(called) or 'none'}.\nLast result:\n{last}"
        return LLMResponse(text, [], Usage(in_tok, estimate_tokens(text)), self.model)


# ============================================================================ real providers
def _http_post(url: str, headers: dict, payload: dict, timeout_s: float = 20.0) -> dict:
    import httpx

    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=timeout_s)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise TransientLLMError(f"network: {e}") from e
    if r.status_code in (402, 429) and any(k in r.text for k in ("insufficient_quota", "billing", "credit balance")):
        # out of credits is not a rate limit: retrying cannot help and silently serving the mock would hide it
        raise QuotaExhaustedLLMError(f"HTTP {r.status_code} (quota/billing, not retryable): {r.text[:300]}")
    if r.status_code in (401, 403):
        raise AuthLLMError(f"HTTP {r.status_code}: {r.text[:300]}")
    if r.status_code == 429 or r.status_code >= 500:
        raise TransientLLMError(f"HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400:
        raise PermanentLLMError(f"HTTP {r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError as e:  # 200 with a truncated / non-JSON body (proxy, gateway): retryable
        raise TransientLLMError(f"non-JSON response: {r.text[:120]}") from e


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini", base_url: str = "https://api.openai.com/v1"):
        self.api_key, self.model, self.base_url = api_key, model, base_url

    def complete(self, messages: list[Message], tools: list[dict], max_output_tokens: int = 1024) -> LLMResponse:
        msgs = []
        for m in messages:
            if m.role == "tool":
                msgs.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant" and m.tool_calls:
                msgs.append({"role": "assistant", "content": m.content or None, "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in m.tool_calls]})
            else:
                msgs.append({"role": m.role, "content": m.content})
        # max_completion_tokens is accepted by every current chat model; max_tokens is rejected by newer ones
        payload = {"model": self.model, "messages": msgs, "max_completion_tokens": max_output_tokens}
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        data = _http_post(f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, payload)
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"__unparseable__": tc["function"]["arguments"]}
            calls.append(ToolCall(tc["id"], tc["function"]["name"], args))
        u = data.get("usage", {})
        return LLMResponse(msg.get("content") or "", calls, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0)), self.model)


class AnthropicClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5", base_url: str = "https://api.anthropic.com/v1"):
        self.api_key, self.model, self.base_url = api_key, model, base_url

    def complete(self, messages: list[Message], tools: list[dict], max_output_tokens: int = 1024) -> LLMResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        msgs: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                content = ([{"type": "text", "text": m.content}] if m.content else []) + [
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in m.tool_calls]
                msgs.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
            else:
                msgs.append({"role": "user", "content": m.content})
        payload = {"model": self.model, "max_tokens": max_output_tokens, "system": system, "messages": msgs}
        if tools:
            payload["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        data = _http_post(f"{self.base_url}/messages",
                          {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, payload)
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in data["content"] if b["type"] == "tool_use"]
        u = data.get("usage", {})
        return LLMResponse(text, calls, Usage(u.get("input_tokens", 0), u.get("output_tokens", 0)), self.model)


class FallbackLLM:
    """Primary provider with jittered retries behind a circuit breaker; degrades to the mock.

    Order per call: breaker check -> primary (up to `retry.max_attempts` tries on TransientLLMError, full
    jitter) -> on exhaustion or an OPEN breaker the deterministic fallback answers. PermanentLLMError (bad
    key, malformed request) is NOT masked: it propagates so a misconfiguration is loud, and the runner turns
    it into status="error" instead of a crash. `served` counts who answered (provable in tests)."""

    def __init__(self, primary: LLMClient, fallback: LLMClient, breaker=None, retry=None, sleep=None, rng=None):
        import random
        import time

        from core.harness.resilience import CircuitBreaker, RetryPolicy

        self.primary, self.fallback = primary, fallback
        self.breaker = breaker or CircuitBreaker(f"llm:{primary.model}", failure_threshold=3, recovery_s=60)
        self.retry = retry or RetryPolicy(max_attempts=3, base_s=0.25, cap_s=4.0)
        self.sleep, self.rng = sleep or time.sleep, rng or random.Random()
        self.model = primary.model
        self.last_served_by = primary.model
        self.served: Counter = Counter()
        self.last_error = ""
        self.events: list[dict] = []     # retry / fallback facts of recent calls; drained into the llm.call span
        self.blocked_until = 0.0         # quota / auth failure: do not call the provider again before this time
        self.blocked_error: PermanentLLMError | None = None
        self.block_s = 300.0

    def drain_events(self) -> list[dict]:
        ev, self.events = self.events, []
        return ev

    def complete(self, messages, tools, max_output_tokens: int = 1024) -> LLMResponse:
        import time

        from core.harness.resilience import CircuitOpenError, retry_call

        if self.blocked_error is not None and time.monotonic() < self.blocked_until:
            # out of credits / bad key: asking again cannot succeed. Fail fast with the SAME typed error (no HTTP
            # request) so callers and the banner keep seeing the real cause; re-probed after block_s.
            self.events.append({"name": "llm.provider_blocked", "error": str(self.blocked_error)[:200]})
            raise self.blocked_error

        def once():
            with self.breaker.guard():
                return self.primary.complete(messages, tools, max_output_tokens)

        def on_retry(attempt: int, err: BaseException, delay: float) -> None:
            self.events.append({"name": "llm.retry", "attempt": attempt, "delay_s": round(delay, 3),
                                "error": f"{type(err).__name__}: {str(err)[:160]}"})

        try:
            resp, _ = retry_call(once, self.retry, retry_on=(TransientLLMError,), rng=self.rng, sleep=self.sleep,
                                 on_retry=on_retry)
            self.last_served_by = self.primary.model
            self.blocked_error = None
        except (QuotaExhaustedLLMError, AuthLLMError) as e:
            self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            self.blocked_error, self.blocked_until = e, time.monotonic() + self.block_s
            log.error("LLM %s unusable (%s); provider calls paused for %.0fs", self.primary.model, self.last_error, self.block_s)
            raise
        except (CircuitOpenError, TransientLLMError) as e:
            self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
            self.events.append({"name": "llm.fallback", "to": self.fallback.model, "error": self.last_error})
            # a real failure is a warning; the (many) calls rejected while the breaker is open are debug noise
            log.log(logging.WARNING if isinstance(e, TransientLLMError) else logging.DEBUG,
                    "LLM %s unavailable (%s); serving %s", self.primary.model, self.last_error, self.fallback.model)
            resp = self.fallback.complete(messages, tools, max_output_tokens)
            self.last_served_by = self.fallback.model
        self.served[self.last_served_by] += 1
        return resp


def is_real_llm(llm) -> bool:
    """True when `llm` talks to a real provider (possibly behind FallbackLLM), False for the mock / None."""
    return llm is not None and not isinstance(llm, MockLLM)


def get_llm(prefer: Literal["auto", "mock", "naive"] = "auto") -> LLMClient:
    """auto: ANTHROPIC_API_KEY -> Anthropic, else OPENAI_API_KEY -> OpenAI, else the deterministic mock.
    Set LLM_PROVIDER=mock to force the mock even when a key is present (offline evals, CI)."""
    from core.config import get_settings

    if prefer == "naive":
        return MockLLM("naive")
    s = get_settings()
    if s.llm_provider == "mock":
        return MockLLM("compliant")
    if prefer == "auto" and s.anthropic_api_key:
        return FallbackLLM(AnthropicClient(s.anthropic_api_key, s.llm_model or "claude-sonnet-5-5"), MockLLM("compliant"))
    if prefer == "auto" and s.openai_api_key:
        return FallbackLLM(OpenAIClient(s.openai_api_key, s.llm_model or "gpt-4.1-mini"), MockLLM("compliant"))
    return MockLLM("compliant")
