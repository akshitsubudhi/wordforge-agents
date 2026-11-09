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
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Protocol

from core.harness import textnorm as tn

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
def _http_post(url: str, headers: dict, payload: dict, timeout_s: float = 60.0) -> dict:
    import httpx

    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=timeout_s)
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise TransientLLMError(f"network: {e}") from e
    if r.status_code == 429 or r.status_code >= 500:
        raise TransientLLMError(f"HTTP {r.status_code}: {r.text[:200]}")
    if r.status_code >= 400:
        raise PermanentLLMError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


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
        payload = {"model": self.model, "messages": msgs, "max_tokens": max_output_tokens}
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
    """Primary provider guarded by a circuit breaker; degrades to the mock."""

    def __init__(self, primary: LLMClient, fallback: LLMClient, breaker=None):
        from core.harness.resilience import CircuitBreaker

        self.primary, self.fallback = primary, fallback
        self.breaker = breaker or CircuitBreaker(f"llm:{primary.model}", failure_threshold=3, recovery_s=60)
        self.model = primary.model
        self.last_served_by = primary.model

    def complete(self, messages, tools, max_output_tokens: int = 1024) -> LLMResponse:
        from core.harness.resilience import CircuitOpenError

        try:
            with self.breaker.guard():
                resp = self.primary.complete(messages, tools, max_output_tokens)
            self.last_served_by = self.primary.model
            return resp
        except (CircuitOpenError, TransientLLMError):
            self.last_served_by = self.fallback.model
            return self.fallback.complete(messages, tools, max_output_tokens)


def get_llm(prefer: Literal["auto", "mock", "naive"] = "auto") -> LLMClient:
    from core.config import get_settings

    if prefer == "naive":
        return MockLLM("naive")
    s = get_settings()
    if prefer == "auto" and s.anthropic_api_key:
        return FallbackLLM(AnthropicClient(s.anthropic_api_key, s.llm_model or "claude-sonnet-5-5"), MockLLM("compliant"))
    if prefer == "auto" and s.openai_api_key:
        return FallbackLLM(OpenAIClient(s.openai_api_key, s.llm_model or "gpt-4.1-mini"), MockLLM("compliant"))
    return MockLLM("compliant")
