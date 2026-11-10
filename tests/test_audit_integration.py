"""
================================================================================
BLOCK: Audit Suite 3/3 - End-to-End Integration
--------------------------------------------------------------------------------
Role & contracts
  1 LLM provider wiring. With ANTHROPIC_API_KEY / OPENAI_API_KEY set, the
    agents issue REAL provider requests (asserted at the HTTP boundary: URL,
    auth header, payload shape, tool round-trip) THROUGH the harness (budget
    pre-check, breaker, retries, llm.call spans). With no key they run on the
    deterministic mock. Provider outages, bad keys and junk output never
    crash a pipeline.
  2 DuckDB single-writer behaviour while the FastAPI app is running: the
    game, the Control Center jobs and concurrent trace reads share one
    process without lock errors; a second PROCESS gets a clear exit-code-2
    message, never a stack trace or a corrupted file.
  3 One schema: game.html only calls routes the server exposes, live play
    lands in the same `events` / `players` / `experiment_exposures` tables
    the seeder and the agents use, and the agents can read those rows back.

Failure modes & 1-step mitigation
  * No network in CI -> the HTTP boundary (`llm._http_post`) is the only
    thing replaced; everything above it is the production code path.
================================================================================
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

import core.harness.llm as llm_mod
from agents.level_factory.critic_and_eval import LevelFactory
from agents.liveops_copilot.agent import LiveOpsCopilot
from agents.soma_intake.extractor import llm_candidates
from core.db import EVENT_COLUMNS, DatabaseLockedError, TelemetryDB, open_rw
from core.harness.llm import (
    AnthropicClient, FallbackLLM, Message, MockLLM, OpenAIClient, PermanentLLMError, TransientLLMError, get_llm,
    is_real_llm,
)
from core.harness.redteam_guard import AgentPolicy
from core.harness.resilience import BudgetExceededError, BudgetGuard, CircuitBreaker
from core.harness.runner import AgentRunner, AgentSpec, guarded_completion
from core.harness.telemetry import DuckDBSpanExporter, InMemoryExporter, Tracer, agent_metrics, traces_connection
from core.harness.tools import Privilege, StrictArgs, ToolRegistry, ToolSpec
from core.wordlist import load_lexicon
from scripts.seed_telemetry import seed_database

REPO = Path(__file__).resolve().parents[1]


# ============================================================================ fixtures
@pytest.fixture(scope="module")
def lex():
    return load_lexicon()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, seeded_copy):
    d = tmp_path_factory.mktemp("audit_integration")
    prev = os.environ.get("LIVEOPS_DATA_DIR")
    os.environ["LIVEOPS_DATA_DIR"] = str(d)
    seeded_copy(d, n_players=400, seed=3)
    yield d
    if prev is None:
        os.environ.pop("LIVEOPS_DATA_DIR", None)
    else:
        os.environ["LIVEOPS_DATA_DIR"] = prev


class FakeProvider:
    """Stands in for the provider's HTTPS endpoint only. Records every request the real client code sends."""

    def __init__(self, script):
        self.script, self.requests = list(script), []

    def __call__(self, url, headers, payload, timeout_s=20.0):
        self.requests.append({"url": url, "headers": headers, "payload": payload})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item(payload) if callable(item) else item


def anthropic_text(text, tin=40, tout=12):
    return {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": tin, "output_tokens": tout}}


def openai_text(text, tin=40, tout=12):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": tin, "completion_tokens": tout}}


class EchoArgs(StrictArgs):
    text: str


def echo_runner(tmp_path, llm, mem=None, **kw) -> AgentRunner:
    reg = ToolRegistry([ToolSpec("echo", "Echo text back", EchoArgs, lambda a, ctx: f"echo:{a.text}", Privilege.READ)])
    return AgentRunner(AgentSpec("t", "You are a test agent.", AgentPolicy(frozenset({"echo"}), tmp_path)), reg, llm,
                       tracer=Tracer(mem or InMemoryExporter()), sleep=lambda s: None, **kw)


# ============================================================================ 1. provider selection
def test_no_key_means_the_deterministic_mock_everywhere(data_dir, lex):
    assert isinstance(get_llm(), MockLLM) and not is_real_llm(get_llm())
    db = TelemetryDB(data_dir / "telemetry.duckdb")
    try:
        f = LevelFactory(db, lex=lex, tracer=Tracer(InMemoryExporter()), seed=1)
        cop = LiveOpsCopilot(db, factory=f, decide=lambda r: (False, "n/a"), tracer=Tracer(InMemoryExporter()))
        assert isinstance(f.llm, MockLLM) and isinstance(cop.llm, MockLLM) and not f.generator.uses_llm
        assert cop.scan()["anomalies"][0]["level_id"] == 7                 # the pipeline runs with no provider at all
        assert cop.summarize({"anomaly": {"message": "L7"}, "experiments": {}}).startswith("- Detected: L7")
        assert len(f.generator.propose(0.3, 4, 100, 3)) == 3 and f.generator.stats["llm_calls"] == 0
    finally:
        db.close()


def test_key_selects_the_real_provider_and_mock_override_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    llm = get_llm()
    assert isinstance(llm, FallbackLLM) and isinstance(llm.primary, OpenAIClient) and llm.model == "gpt-4.1-mini"
    assert isinstance(llm.fallback, MockLLM) and is_real_llm(llm)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")                 # Anthropic takes precedence when both are set
    llm = get_llm()
    assert isinstance(llm.primary, AnthropicClient) and llm.primary.api_key == "sk-ant-test"
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    assert get_llm().model == "claude-haiku-5-5"
    monkeypatch.setenv("LLM_PROVIDER", "mock")                             # explicit offline switch beats any key
    assert isinstance(get_llm(), MockLLM)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")                            # empty string is "unset", not a key
    assert isinstance(get_llm().primary, OpenAIClient)
    assert get_llm("mock").model == "mock-compliant" and get_llm("naive").model == "mock-naive"


# ============================================================================ 2. real API calls through the harness
def test_anthropic_agent_loop_makes_real_requests_with_a_tool_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    api = FakeProvider([
        {"content": [{"type": "text", "text": "calling"}, {"type": "tool_use", "id": "tu_1", "name": "echo", "input": {"text": "hi"}}],
         "usage": {"input_tokens": 120, "output_tokens": 30}},
        anthropic_text("The tool said echo:hi", 180, 20),
    ])
    monkeypatch.setattr(llm_mod, "_http_post", api)
    mem = InMemoryExporter()
    res = echo_runner(tmp_path, get_llm(), mem).run("Echo hi and report.")
    assert res.status == "completed" and res.final_text == "The tool said echo:hi"
    assert [(tc.name, tc.args, tc.ok) for tc in res.tool_calls] == [("echo", {"text": "hi"}, True)]
    first, second = api.requests
    assert first["url"] == "https://api.anthropic.com/v1/messages"
    assert first["headers"] == {"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"}
    p = first["payload"]
    assert p["model"] == "claude-sonnet-5-5" and p["max_tokens"] == 1024 and "test agent" in p["system"]
    assert p["messages"] == [{"role": "user", "content": "Echo hi and report."}]
    assert p["tools"][0]["name"] == "echo" and p["tools"][0]["input_schema"]["properties"]["text"]["type"] == "string"
    assert p["tools"][0]["input_schema"]["additionalProperties"] is False
    # the second request carries the assistant tool_use and the tool_result in Anthropic's wire format
    assert second["payload"]["messages"][1]["content"][-1] == {"type": "tool_use", "id": "tu_1", "name": "echo", "input": {"text": "hi"}}
    assert second["payload"]["messages"][2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "echo:hi"}]}
    # ...and it all went through the harness: priced, budgeted, traced
    spans = [s for s in mem.spans if s.name == "llm.call"]
    assert [s.attributes["model"] for s in spans] == ["claude-sonnet-5-5"] * 2
    assert res.budget["tokens"] == 350 and res.budget["llm_calls"] == 2
    assert res.budget["spent_usd"] == pytest.approx((300 * 3.0 + 50 * 15.0) / 1e6)   # provider usage x published price
    assert sum(s.attributes["cost_usd"] for s in spans) == pytest.approx(res.budget["spent_usd"], abs=1e-7)


def test_openai_agent_loop_makes_real_requests_with_a_tool_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    api = FakeProvider([
        {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}]}}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        openai_text("done: echo:hi", 150, 10),
    ])
    monkeypatch.setattr(llm_mod, "_http_post", api)
    res = echo_runner(tmp_path, get_llm()).run("Echo hi.")
    assert res.status == "completed" and res.final_text == "done: echo:hi" and res.tool_calls[0].ok
    first, second = api.requests
    assert first["url"] == "https://api.openai.com/v1/chat/completions"
    assert first["headers"] == {"Authorization": "Bearer sk-test-openai"}
    assert first["payload"]["model"] == "gpt-4.1-mini" and first["payload"]["max_completion_tokens"] == 1024
    assert first["payload"]["messages"][0]["role"] == "system" and first["payload"]["tools"][0]["type"] == "function"
    assert first["payload"]["tools"][0]["function"]["name"] == "echo"
    assert second["payload"]["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "echo:hi"}
    assert second["payload"]["messages"][-2]["tool_calls"][0]["function"]["arguments"] == '{"text": "hi"}'
    assert res.budget["spent_usd"] == pytest.approx((250 * 0.40 + 30 * 1.60) / 1e6)


def test_real_model_tool_calls_still_pass_through_every_guard(tmp_path, monkeypatch):
    """A real model that hallucinates a tool, smuggles a field or loops is handled exactly like the mock."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    tool = lambda name, args: {"content": [{"type": "tool_use", "id": "tu", "name": name, "input": args}],
                               "usage": {"input_tokens": 50, "output_tokens": 10}}
    monkeypatch.setattr(llm_mod, "_http_post", FakeProvider([tool("drop_database", {}), tool("echo", {"text": "x", "bypass": True}),
                                                             anthropic_text("ok")]))
    res = echo_runner(tmp_path, get_llm()).run("go")
    assert [(b.layer, b.rule) for b in res.blocks] == [("checkpoint", "unknown_tool"), ("checkpoint", "schema")]
    assert res.status == "completed" and not any(tc.executed for tc in res.tool_calls)
    monkeypatch.setattr(llm_mod, "_http_post", FakeProvider([tool("echo", {"text": "again"})]))    # never stops
    res = echo_runner(tmp_path, get_llm(), budget_usd=0.01).run("go")
    assert res.status == "budget_exceeded" and res.budget["spent_usd"] <= 0.01


# ============================================================================ 3. provider failure modes
def test_transient_outage_is_retried_then_served_by_the_mock_and_the_breaker_opens(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    api = FakeProvider([TransientLLMError("HTTP 529: overloaded")])
    monkeypatch.setattr(llm_mod, "_http_post", api)
    sleeps: list[float] = []
    clock = [0.0]
    llm = FallbackLLM(AnthropicClient("sk-ant-test"), MockLLM("compliant"), sleep=sleeps.append,
                      breaker=CircuitBreaker("llm:test", failure_threshold=3, recovery_s=60, clock=lambda: clock[0]))
    res = echo_runner(tmp_path, llm).run('Use the tool.\nCALL echo {"text": "hi"}')
    assert res.status == "completed" and res.tool_calls[0].ok          # degraded to the mock, the run still finished
    assert len(api.requests) == 3 and len(sleeps) == 2                 # 3 attempts with 2 jittered sleeps, then...
    assert all(0 <= s <= 4.0 for s in sleeps)
    assert llm.breaker.state == "open" and llm.served["mock-compliant"] == 2 and llm.served["claude-sonnet-5-5"] == 0
    assert "529" in llm.last_error or "open" in llm.last_error
    n = len(api.requests)
    llm.complete([Message("user", "x")], [])
    assert len(api.requests) == n                                      # ...open breaker: the provider is not hammered
    clock[0] = 61.0                                                    # recovery window passed: one half-open probe
    api.script = [anthropic_text("back")]
    assert llm.complete([Message("user", "x")], []).text == "back" and llm.breaker.state == "closed"
    assert res.budget["spent_usd"] < 1e-4                              # fallback answers are priced as the mock


def test_recovery_after_one_transient_error_uses_the_real_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    api = FakeProvider([TransientLLMError("HTTP 503"), openai_text("real answer"), openai_text("unused")])
    monkeypatch.setattr(llm_mod, "_http_post", api)
    llm = get_llm()
    llm.sleep = lambda s: None
    res = echo_runner(tmp_path, llm).run("hello")
    assert res.final_text == "real answer" and len(api.requests) == 2 and llm.served == {"gpt-4.1-mini": 1}


@pytest.mark.parametrize("status,body,exc", [
    (401, '{"error": {"message": "invalid x-api-key"}}', PermanentLLMError),
    (400, '{"error": {"message": "bad request"}}', PermanentLLMError),
    (429, '{"error": {"type": "insufficient_quota", "message": "You have no credits remaining"}}', PermanentLLMError),
    (429, '{"error": {"type": "rate_limit_error"}}', TransientLLMError),
    (503, "upstream unavailable", TransientLLMError),
])
def test_http_status_classification(monkeypatch, status, body, exc):
    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(status, text=body))
    with pytest.raises(exc):
        llm_mod._http_post("https://api.example.test/v1/x", {}, {})
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectTimeout("timed out")))
    with pytest.raises(TransientLLMError):
        llm_mod._http_post("https://api.example.test/v1/x", {}, {})
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(TransientLLMError):
        llm_mod._http_post("https://api.example.test/v1/x", {}, {})


def test_bad_api_key_ends_the_run_with_an_error_status_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-wrong")
    api = FakeProvider([PermanentLLMError("HTTP 401: invalid x-api-key")])
    monkeypatch.setattr(llm_mod, "_http_post", api)
    mem = InMemoryExporter()
    res = echo_runner(tmp_path, get_llm(), mem).run("hello")
    assert res.status == "error" and "401" in res.stop_reason and len(api.requests) == 1    # loud, and NOT retried
    assert [s.status for s in mem.spans if s.name == "llm.call"] == ["ERROR"]
    # a provider that stays down with NO fallback configured is also a recorded error, not an exception
    monkeypatch.setattr(llm_mod, "_http_post", FakeProvider([TransientLLMError("HTTP 500")]))
    res = echo_runner(tmp_path, AnthropicClient("k")).run("hello")
    assert res.status == "error" and "TransientLLMError" in res.stop_reason


QUOTA_BODY = ('{"error": {"message": "You have no credits remaining. Add credits to continue using the API.", '
              '"type": "insufficient_quota", "code": "insufficient_quota"}}')


def _provider_returns(monkeypatch, status: int, body: str) -> list:
    """Replace only the socket: the real _http_post classification code runs on a real httpx.Response."""
    import httpx

    sent: list = []

    def post(url, **kw):
        sent.append(url)
        return httpx.Response(status, text=body)

    monkeypatch.setattr(httpx, "post", post)
    return sent


def test_quota_429_is_traced_not_retried_and_surfaced_as_an_actionable_status(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _provider_returns(monkeypatch, 429, QUOTA_BODY)
    assert llm_mod.llm_status()["state"] == "ok" and "no call made yet" in llm_mod.llm_status()["message"]
    mem = InMemoryExporter()
    llm = get_llm()
    res = echo_runner(tmp_path, llm, mem).run("hello")
    assert res.status == "error" and "QuotaExhaustedLLMError" in res.stop_reason        # recorded, not raised
    assert len(sent) == 1                                                               # insufficient_quota is NOT retried
    span = next(s for s in mem.spans if s.name == "llm.call")
    assert span.status == "ERROR" and "insufficient_quota" in span.status_message
    assert span.attributes["error_class"] == "quota_exhausted" and span.attributes["retryable"] is False
    assert [e["name"] for e in span.events] == ["llm.error"]
    assert [s.status for s in mem.spans if s.name == "agent.run"] == ["ERROR"]          # the trace shows where it stopped
    st = llm_mod.llm_status()
    assert st == {**st, "mode": "live", "provider": "openai", "state": "quota_exhausted", "ok": False}
    assert "out of credits" in st["message"] and "LLM_PROVIDER=mock" in st["action"] and "insufficient_quota" in st["last_error"]
    # the provider is not hammered: the next calls fail fast with the same typed error and NO HTTP request
    with pytest.raises(llm_mod.QuotaExhaustedLLMError):
        guarded_completion(llm, [Message("user", "x")], budget=BudgetGuard(), tracer=Tracer(mem))
    assert len(sent) == 1 and [e["name"] for e in mem.spans[-1].events] == ["llm.provider_blocked", "llm.error"]
    # breaker noise afterwards must not hide the real cause
    llm_mod.HEALTH.record_error(TransientLLMError("circuit open"))
    assert llm_mod.llm_status()["state"] == "quota_exhausted" and "insufficient_quota" in llm_mod.llm_status()["last_error"]
    # after the pause the provider is probed again; a real answer clears the status
    llm.blocked_until = 0.0
    monkeypatch.setattr(llm_mod, "_http_post", FakeProvider([openai_text("back")]))
    assert guarded_completion(llm, [Message("user", "x")], budget=BudgetGuard(), tracer=Tracer(mem)).text == "back"
    assert llm_mod.llm_status()["ok"] and llm_mod.llm_status()["state"] == "ok"


def test_rate_limit_429_is_retried_with_retry_and_fallback_events_on_the_span(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    sent = _provider_returns(monkeypatch, 429, '{"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}')
    mem = InMemoryExporter()
    llm = get_llm()
    llm.sleep = lambda s: None
    res = echo_runner(tmp_path, llm, mem).run("hello")
    assert res.status == "completed" and len(sent) == 3                                 # 3 attempts, then the mock answers
    span = next(s for s in mem.spans if s.name == "llm.call")
    assert [e["name"] for e in span.events] == ["llm.retry", "llm.retry", "llm.fallback"]
    assert span.events[0]["attributes"]["attempt"] == 1 and "429" in span.events[0]["attributes"]["error"]
    assert span.status == "OK" and span.attributes["fallback"] is True and span.attributes["served_by"] == "mock-compliant"
    st = llm_mod.llm_status()
    assert st["state"] == "degraded" and not st["ok"] and "No action needed" in st["action"]


def test_bad_key_401_and_mock_mode_statuses(tmp_path, monkeypatch):
    assert llm_mod.llm_status() == {**llm_mod.llm_status(), "mode": "mock", "state": "mock", "ok": True, "action": ""}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-wrong")
    sent = _provider_returns(monkeypatch, 401, '{"error": {"type": "authentication_error", "message": "invalid x-api-key"}}')
    res = echo_runner(tmp_path, get_llm()).run("hello")
    assert res.status == "error" and len(sent) == 1
    st = llm_mod.llm_status()
    assert st["state"] == "auth_failed" and "ANTHROPIC_API_KEY" in st["action"] and st["provider"] == "anthropic"
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert llm_mod.llm_status()["mode"] == "mock" and "LLM_PROVIDER=mock" in llm_mod.llm_status()["message"]


# ============================================================================ 4. agents use the provider via the harness
GOOD_GRIDS = ["HEHS/ATNS/AHOA/SHCW", "LDMR/RNOH/EICD/HAIN", "EHSD/SRUS/IPTR/ILEV"]


def test_level_factory_with_a_key_generates_and_critiques_through_the_harness(data_dir, lex, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    def provider(payload):
        prompt = payload["messages"][-1]["content"]
        if "Propose" in prompt:   # generator: 3 good grids, 1 hallucinated/unsolvable-ish, 1 malformed, 1 wrong size
            return anthropic_text(json.dumps({"grids": GOOD_GRIDS + ["QQQQ/ZZZZ/XXXX/JJJJ", "AB/CD", "ABCDE/FGHIJ/KLMNO/PQRST/UVWXY"]}), 200, 80)
        return anthropic_text('Sure! {"solvability_fairness": 5, "difficulty_fit": 5, "lexical_quality": 5, "board_quality": 9, "reason": "fine"}', 300, 40)

    api = FakeProvider([provider])
    monkeypatch.setattr(llm_mod, "_http_post", api)
    mem = InMemoryExporter()
    db = TelemetryDB(data_dir / "telemetry.duckdb")
    try:
        f = LevelFactory(db, lex=lex, tracer=Tracer(mem), seed=5)
        assert isinstance(f.llm, FallbackLLM) and f.generator.uses_llm
        before = db.query("SELECT count(*) AS n FROM levels")[0]["n"]
        with f.runner.workflow("audit:llm_factory") as wf:
            cands = wf.call("generate_candidates", goal=0.3, size=4, time_limit_s=100, n=6)
            assert {c["grid"] for c in cands if c["source"] == "llm"} <= set(GOOD_GRIDS)      # junk grids never get through
            assert f.generator.stats["llm_proposed"] == 6 and f.generator.stats["llm_valid"] == 4
            assert len(cands) == 6                                                            # heuristic filled the quota
            rep = wf.call("critique_level", grid=cands[0]["grid"], target_words=cands[0]["target_words"], time_limit_s=100, goal=0.3)
            assert rep["llm_review"]["reason"] == "fine" and max(rep["scores"].values()) <= 5  # the LLM can only LOWER scores
        gen_req = api.requests[0]
        assert gen_req["url"].endswith("/v1/messages") and gen_req["headers"]["x-api-key"] == "sk-ant-test"
        purposes = [s.attributes.get("purpose") for s in mem.spans if s.name == "llm.call"]
        assert purposes == ["level_generator", "level_critic"]                               # both calls traced + priced
        assert f.llm_budget.llm_calls == 2 and 0 < f.llm_budget.spent_usd < 0.01
        assert f.critic_llm_stats["reviews"] == 1
        assert db.query("SELECT count(*) AS n FROM levels")[0]["n"] == before
    finally:
        db.close()


def test_level_factory_survives_a_dead_or_broke_provider_and_a_spent_budget(data_dir, lex, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    db = TelemetryDB(data_dir / "telemetry.duckdb")
    try:
        for failure in (PermanentLLMError("HTTP 429 (quota/billing, not retryable)"), TransientLLMError("HTTP 500")):
            monkeypatch.setattr(llm_mod, "_http_post", FakeProvider([failure]))
            f = LevelFactory(db, lex=lex, tracer=Tracer(InMemoryExporter()), seed=11)
            f.llm.sleep = lambda s: None
            shipped = f.ship_daily(f"2032-01-0{1 if isinstance(failure, PermanentLLMError) else 2}", candidates_per_slot=10)
            assert shipped and all(s["source"] == "heuristic" for s in shipped)          # degraded, not down
            assert f.generator.stats["llm_proposed"] == 0
        # spend cap: generator + critic share ONE budget per pipeline run and stop calling when it is gone
        monkeypatch.setenv("HARNESS_MAX_RUN_USD", "0.0012")
        api = FakeProvider([lambda p: openai_text('{"solvability_fairness": 5, "difficulty_fit": 5, "lexical_quality": 5, '
                                                  '"board_quality": 5, "reason": "ok", "grids": []}', 400, 100)])
        monkeypatch.setattr(llm_mod, "_http_post", api)
        f = LevelFactory(db, lex=lex, tracer=Tracer(InMemoryExporter()), seed=12)
        with f.runner.workflow("audit:budget") as wf:
            cands = wf.call("generate_candidates", goal=0.3, size=4, time_limit_s=100, n=8)
            for c in cands:
                wf.call("critique_level", grid=c["grid"], target_words=c["target_words"], time_limit_s=100, goal=0.3)
        assert f.llm_budget.spent_usd <= 0.0012 and f.critic_llm_stats["budget_stops"] >= 1
        assert len(api.requests) == f.llm_budget.llm_calls < 9                            # calls stopped at the cap
    finally:
        db.close()


def test_copilot_summary_and_soma_extractor_use_the_harness(data_dir, lex, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    api = FakeProvider([anthropic_text("- Level 7 fails 63% of attempts\n- variant shipped", 500, 60)])
    monkeypatch.setattr(llm_mod, "_http_post", api)
    mem = InMemoryExporter()
    db = TelemetryDB(data_dir / "telemetry.duckdb")
    try:
        cop = LiveOpsCopilot(db, decide=lambda r: (False, "n/a"), tracer=Tracer(mem))
        out = cop.summarize({"anomaly": {"message": "Level 7 has a 63% fail rate"}, "experiments": {}})
        assert out.startswith("- Level 7 fails 63%") and "63% fail rate" in api.requests[0]["payload"]["messages"][0]["content"]
        root = [s for s in mem.spans if s.name == "agent.run"][0]
        assert root.attributes["mode"] == "READ_ONLY" and root.attributes["cost_usd"] > 0
        monkeypatch.setattr(llm_mod, "_http_post", FakeProvider([PermanentLLMError("HTTP 401")]))
        assert cop.summarize({"anomaly": {"message": "L7"}, "experiments": {}}) == "- Detected: L7"   # template fallback
    finally:
        db.close()
    # extractor: the model's proposal is kept only when its quote is verbatim in a customer message
    from agents.soma_intake.workflow import normalize_email

    m = normalize_email("From: owner@acme.test\nSubject: quote\nMessage-ID: <a1@x>\n\nWe are Acme Bakery LLC with 12 full time staff.")
    api = FakeProvider([anthropic_text(json.dumps({"applicant_name": {"value": "Acme Bakery LLC", "quote": "Acme Bakery LLC"},
                                                   "fein": {"value": "12-3456789", "quote": "not in the message"},
                                                   "made_up_field": {"value": 1, "quote": "Acme"}}), 300, 60)])
    monkeypatch.setattr(llm_mod, "_http_post", api)
    budget = BudgetGuard(max_usd=0.05)
    cands = llm_candidates([m], get_llm(), budget, Tracer(mem))
    assert [(c.key, c.value, c.citation.quote) for c in cands] == [("applicant_name", "Acme Bakery LLC", "Acme Bakery LLC")]
    assert len(api.requests) == 1 and budget.llm_calls == 1    # fabricated quote + unknown field dropped; 1 priced call
    assert llm_candidates([m], MockLLM("compliant")) == [] and llm_candidates([m], None) == []
    with pytest.raises(BudgetExceededError):                                              # the cap is real here too
        guarded_completion(get_llm(), [Message("user", "x" * 4000)], budget=BudgetGuard(max_usd=1e-6), max_output_tokens=1200)


def test_soma_extractor_survives_everything_a_real_model_returns(monkeypatch):
    """Regression (found with a live model): a list where a scalar was expected crashed resolve() with
    'unhashable type: list'. Model output is untrusted: wrong types, shapes and vocabulary are dropped."""
    from agents.soma_intake.extractor import SPEC, coerce_llm_value, extract
    from agents.soma_intake.workflow import normalize_email

    ok = [("employees_full_time", 12, [12]), ("employees_full_time", "12", [12]), ("employees_full_time", 12.0, [12]),
          ("annual_revenue", "$3.1M", [3_100_000]), ("annual_revenue", 3100000, [3_100_000]),
          ("policy_effective_date", "2027-01-01", ["2027-01-01"]), ("business_phone", "(415) 555-0106", ["+14155550106"]),
          ("contact_email", "Owen@Juniper.example", ["owen@juniper.example"]), ("mailing_state", "california", ["CA"]),
          ("mailing_state", "ca", ["CA"]), ("entity_type", "LLC", ["llc"]), ("mailing_zip", "94197", ["94197"]),
          ("lines_of_business", ["General Liability", "property", "pet insurance"], ["general_liability", "property"]),
          ("hazards", "liquor", ["liquor"])]
    for key, raw, want in ok:
        assert coerce_llm_value(SPEC[key], raw) == want, (key, raw)
    bad = [("employees_full_time", [12, 3]), ("employees_full_time", {"n": 12}), ("employees_full_time", True),
           ("employees_full_time", -4), ("employees_full_time", 12.5), ("employees_full_time", "a dozen"),
           ("annual_revenue", "lots"), ("policy_effective_date", "1/1"), ("policy_effective_date", "2027-13-40"),
           ("business_phone", "call me"), ("contact_email", "not an email"), ("mailing_state", "ZZ"),
           ("entity_type", "nonprofit cooperative"), ("mailing_zip", "9419"), ("applicant_name", ["Acme", "LLC"]),
           ("applicant_name", ""), ("applicant_name", "x" * 500), ("hazards", ["dragons"]), ("fein", None)]
    for key, raw in bad:
        assert coerce_llm_value(SPEC[key], raw) == [], (key, raw)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = normalize_email("From: Owen <owen@juniper.example>\nSubject: quote\nMessage-ID: <a1@x>\n\n"
                        "I run Juniper Books LLC in San Francisco with 16 employees. We want a GL policy. Phone: (415) 555-0106")
    hostile = {"applicant_name": {"value": "Juniper Books, L.L.C.", "quote": "Juniper Books LLC"},      # other spelling
               "employees_full_time": {"value": [16], "quote": "16 employees"},                          # the crash
               "hazards": {"value": [["liquor"], {"a": 1}, 7], "quote": "GL policy"},
               "lines_of_business": {"value": ["general_liability"], "quote": "GL policy"},
               "annual_revenue": {"value": {"usd": 5}, "quote": "16 employees"},
               "business_description": {"value": "Space elevator operator", "quote": "Juniper Books LLC"},  # derived field
               "prior_losses_3y": {"value": 2, "quote": "this text is not in any message"},
               "mailing_state": {"value": "CA", "quote": "San Francisco"},
               "nonsense": "not even an object", "fein": 5}
    monkeypatch.setattr(llm_mod, "_http_post", FakeProvider([openai_text(json.dumps(hostile), 400, 200)]))
    rules_only = extract([m])
    ext = extract([m], get_llm(), BudgetGuard(max_usd=0.05), Tracer(InMemoryExporter()))        # must not raise
    assert ext.value("applicant_name") == rules_only.value("applicant_name") == "Juniper Books LLC"
    assert ext.fields["applicant_name"].status == rules_only.fields["applicant_name"].status   # no manufactured conflict
    assert ext.value("business_description") == rules_only.value("business_description")
    assert ext.value("prior_losses_3y") is None and ext.value("annual_revenue") is None and ext.value("hazards") in (None, [])
    assert ext.value("mailing_state") == "CA" and ext.value("lines_of_business") == ["general_liability"]
    for junk in ("[]", "not json at all", '{"employees_full_time": [1, 2]}', '["a", "b"]', ""):
        monkeypatch.setattr(llm_mod, "_http_post", FakeProvider([openai_text(junk)]))
        assert extract([m], get_llm(), BudgetGuard(max_usd=0.05)).value("applicant_name") == "Juniper Books LLC"


def test_soma_engine_caps_model_spend_per_case(monkeypatch, tmp_path):
    from agents.soma_intake.workflow import EventStore, IntakeEngine, normalize_email

    monkeypatch.setenv("LIVEOPS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("HARNESS_MAX_RUN_USD", "0.004")
    api = FakeProvider([openai_text("{}", 2000, 500)])                       # $0.0016 per call at gpt-4.1-mini prices
    monkeypatch.setattr(llm_mod, "_http_post", api)
    eng = IntakeEngine(EventStore(tmp_path / "soma.sqlite"), llm=get_llm(), tracer=Tracer(InMemoryExporter()))
    case_id = None
    for i in range(6):   # six messages on one case -> six extractions, but the per-case cap stops the model calls
        case_id = eng.handle_inbound(normalize_email(f"From: Owen <owen@juniper.example>\nSubject: quote\nMessage-ID: <m{i}@x>\n\n"
                                                     f"Juniper Books LLC here, note {i}: we have {10 + i} employees.")) or case_id
    budget = eng._llm_budgets[case_id]
    assert budget.spent_usd <= 0.004 and len(api.requests) == budget.llm_calls < 6
    assert eng.state(case_id).status in ("AWAITING_CUSTOMER", "HUMAN_HANDOFF", "COLLECTING")   # the case kept moving


# ============================================================================ 5. DuckDB: one writer, no lock crashes
def test_traces_db_survives_concurrent_flushes_and_reads_in_one_process(tmp_path):
    """Regression: /api/cc/traces/recent returned 500 ('Unique file handle conflict') when a span flush and a
    dashboard read overlapped."""
    path = tmp_path / "traces.duckdb"
    exp = DuckDBSpanExporter(path, flush_interval_s=0.02)
    tracer = Tracer(exp)
    errors: list[str] = []
    stop = threading.Event()

    def write():
        while not stop.is_set():
            with tracer.span("agent.run", new_trace=True, agent="t"):
                with tracer.span("tool.call", tool="x"):
                    time.sleep(0.004)   # realistic span rate: stay below the exporter's bounded buffer

    def read():
        while not stop.is_set():
            try:
                agent_metrics(path)
                with traces_connection(path) as con:
                    con.execute("SELECT count(*) FROM spans").fetchone()
                exp.flush(force=True)
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=write) for _ in range(3)] + [threading.Thread(target=read) for _ in range(5)]
    for t in threads:
        t.start()
    time.sleep(2.5)
    stop.set()
    for t in threads:
        t.join()
    exp.close()
    assert errors == [] and exp.stats["flush_failures"] == 0 and exp.stats["dropped"] == 0
    with traces_connection(path) as con:
        assert con.execute("SELECT count(*) FROM spans").fetchone()[0] == exp.stats["exported"] > 100


LOCK_HOLDER = """
import sys, time, duckdb
con = duckdb.connect(sys.argv[1])
con.execute("CREATE TABLE IF NOT EXISTS t (i INTEGER)")
print("locked", flush=True)
time.sleep(60)
"""


def _env(data_dir: Path) -> dict:
    return {**os.environ, "LIVEOPS_DATA_DIR": str(data_dir), "PYTHONPATH": str(REPO), "LLM_PROVIDER": "mock"}


def test_second_process_gets_a_clear_lock_error_and_spans_are_never_lost(tmp_path):
    db_path, traces = tmp_path / "telemetry.duckdb", tmp_path / "traces.duckdb"
    holders = [subprocess.Popen([sys.executable, "-c", LOCK_HOLDER, str(p)], stdout=subprocess.PIPE, text=True) for p in (db_path, traces)]
    try:
        for h in holders:
            assert h.stdout.readline().strip() == "locked"
        with pytest.raises(DatabaseLockedError) as e:
            TelemetryDB(db_path)
        assert e.value.pid == holders[0].pid and "stop the server" in str(e.value)
        # spans written while another process owns traces.duckdb: buffered, then spooled, never an exception
        exp = DuckDBSpanExporter(traces, background=False)
        tracer = Tracer(exp)
        for _ in range(5):
            with tracer.span("agent.run", new_trace=True):
                pass
        assert exp.flush(force=True) == 0 and exp.stats["flush_failures"] == 1
        exp.close()
        assert exp.stats["spilled"] == 5 and exp.spool_path.exists()
    finally:
        for h in holders:
            h.kill()
            h.wait()
    exp2 = DuckDBSpanExporter(traces, background=False)                # lock released: the next process ingests the spool
    with Tracer(exp2).span("agent.run", new_trace=True):
        pass
    exp2.close()
    with traces_connection(traces) as con:
        assert con.execute("SELECT count(*) FROM spans").fetchone()[0] == 6


@pytest.fixture(scope="module")
def client(data_dir):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def _wait_job(client, kind: str, params: dict | None = None, timeout_s: float = 240) -> dict:
    jid = client.post("/api/cc/jobs", json={"kind": kind, "params": params or {}}).json()["job_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = client.get(f"/api/cc/jobs/{jid}").json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.1)
    raise AssertionError(f"job {kind} did not finish")


def test_cli_scripts_fail_cleanly_while_the_server_owns_the_database(client, data_dir):
    assert client.get("/api/health").json()["status"] == "ok"
    for cmd in (["scripts/seed_telemetry.py"], ["scripts/run_fault_drills.py"], ["scripts/verify_phase2.py", "--no-trace"],
                ["scripts/run_ad_economy_sim.py"], ["-m", "agents.liveops_copilot.agent", "scan"], ["scripts/build_scorecard.py"]):
        p = subprocess.run([sys.executable, *cmd], cwd=REPO, env=_env(data_dir), capture_output=True, text=True, timeout=120)
        assert p.returncode == 2, (cmd, p.returncode, p.stderr[-300:])
        assert "is locked by another process" in p.stderr and "Traceback" not in p.stderr, (cmd, p.stderr[-300:])
    assert client.get("/api/health").json()["status"] == "ok"          # the server never noticed


def test_game_agents_and_trace_reads_share_the_running_server_without_lock_errors(client):
    """Players play, every fault drill + agents run as Control Center jobs, dashboards poll - all at once."""
    bad: list = []
    stop = threading.Event()

    def player(i: int):
        s = client.post("/api/game/session", json={"user_id": f"audit_{i}"}).json()
        for lvl in (1, 7, 3):
            a = client.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": lvl})
            if a.status_code != 200:
                bad.append(("start", a.status_code, a.text[:200]))
                continue
            att = a.json()["attempt_id"]
            for r in range(4):
                w = client.post("/api/game/level/word", json={"attempt_id": att, "path": [[r, 0], [r, 1], [r, 2]]})
                if w.status_code != 200:
                    bad.append(("word", w.status_code, w.text[:200]))
            client.post("/api/game/ad", json={"attempt_id": att, "placement": "rewarded_hint"})
            e = client.post("/api/game/level/end", json={"attempt_id": att, "reason": "quit"})
            if e.status_code != 200:
                bad.append(("end", e.status_code, e.text[:200]))

    def poller():
        while not stop.is_set():
            for path in ("/api/cc/overview", "/api/cc/metrics", "/api/cc/traces/recent", "/api/cc/funnel", "/api/harness/metrics",
                         "/api/telemetry/funnel", "/api/game/levels", "/api/health"):
                r = client.get(path)
                if r.status_code != 200 or r.json().get("error"):
                    bad.append((path, r.status_code, r.text[:200]))

    threads = [threading.Thread(target=player, args=(i,)) for i in range(8)] + [threading.Thread(target=poller) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        results = {k: _wait_job(client, k, p) for k, p in (
            ("fault_hallucinated_level", {}), ("fault_network_dark", {}), ("fault_peeking", {}), ("fault_prompt_injection", {}),
            ("fault_runaway_cost", {}), ("factory_push", {"candidates": 12}), ("copilot_scan", {}), ("ad_floor", {}))}
        for t in threads[:8]:
            t.join()
    finally:   # ALWAYS stop the pollers: a failure above must not leave threads hammering the server during teardown
        stop.set()
        for t in threads:
            t.join(timeout=60)
    assert bad == []
    assert {k: v["status"] for k, v in results.items()} == {k: "done" for k in results}
    assert all(results[k]["result"]["blocked"] for k in results if k.startswith("fault_"))     # 5/5 mitigated in-server
    tid = results["fault_runaway_cost"]["result"]["trace_ids"][0]
    detail = client.get(f"/api/cc/traces/{tid}")
    assert detail.status_code == 200 and any(s["name"] == "llm.call" for s in detail.json()["spans"])
    assert client.get("/api/cc/traces/" + "0" * 32).status_code == 404


def test_control_center_with_an_exhausted_provider_shows_a_banner_not_a_500(client, monkeypatch):
    """HTTP 429 insufficient_quota from the live provider: jobs finish on the fallback, every route answers 200,
    /api/llm/status and the overview carry an actionable message, and the job log says what happened."""
    assert client.get("/api/llm/status").json()["mode"] == "mock"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent = _provider_returns(monkeypatch, 429, QUOTA_BODY)
    cc = client.app.state.cc
    old_factory, cc["factory"] = cc["factory"], None          # the server builds its factory from the environment
    try:
        job = _wait_job(client, "factory_push", {"candidates": 8})
        assert job["status"] == "done" and job["error"] is None and job["result"]["published"]   # shipped on the heuristic
        warn = [s for s in job["steps"] if s.get("llm")]
        assert len(warn) == 1 and "out of credits" in warn[0]["title"] and "LLM_PROVIDER=mock" in warn[0]["detail"]
        assert 1 <= len(sent) <= 2                             # generator (+ at most one critic probe), then paused
        for path in ("/api/llm/status", "/api/cc/overview", "/api/cc/metrics", "/api/cc/traces/recent", "/api/health", "/control-center"):
            assert client.get(path).status_code == 200, path
        st = client.get("/api/llm/status").json()
        assert st["state"] == "quota_exhausted" and st["ok"] is False and st["provider"] == "openai"
        assert client.get("/api/cc/overview").json()["llm"]["state"] == "quota_exhausted"
        assert job["llm"]["state"] == "quota_exhausted"        # the polled job payload drives the banner
        html = client.get("/control-center").text
        assert 'id="llmBanner"' in html and "showLLM(o.llm)" in html and "showLLM(j.llm)" in html
        tid = job["result"]["trace_id"]
        spans = client.get(f"/api/cc/traces/{tid}").json()["spans"]
        assert any(s["name"] == "llm.call" and s["status"] == "ERROR" and "insufficient_quota" in (s["status_message"] or "")
                   and any(e["name"] == "llm.error" for e in s["events"]) for s in spans)
    finally:
        cc["factory"] = old_factory


# ============================================================================ 6. one schema, end to end
def test_game_html_only_calls_routes_the_server_exposes(client):
    html = (REPO / "app" / "static" / "game.html").read_text()
    called = set(re.findall(r'api\("(/api/[a-z/_]+)"', html))
    routes = set(client.get("/openapi.json").json()["paths"])
    assert called and called <= routes, called - routes
    assert called == {"/api/game/session", "/api/game/levels", "/api/game/level/start", "/api/game/level/word",
                      "/api/game/level/end", "/api/game/ad"}
    assert client.get("/game").status_code == 200 and "LetterLoop" in client.get("/game").text


def test_live_play_lands_in_the_tables_the_agents_read(client, lex):
    db: TelemetryDB = client.app.state.db
    cols = [r["column_name"] for r in db.query("SELECT column_name FROM information_schema.columns WHERE table_name = 'events' "
                                               "ORDER BY ordinal_position")]
    assert cols == EVENT_COLUMNS                                        # the writer's column list IS the table
    s = client.post("/api/game/session", json={"user_id": "schema_probe"}).json()
    a = client.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": 1}).json()
    from core.wordgrid import Trie, parse_grid, solve

    words = list(solve(parse_grid("/".join(a["grid"])), Trie(lex.words)).words.items())
    for _, path in words[: a["target_words"]]:
        last = client.post("/api/game/level/word", json={"attempt_id": a["attempt_id"], "path": path}).json()
    assert last["status"] == "completed"
    live = db.query("SELECT * FROM events WHERE user_id = 'schema_probe' ORDER BY ts")
    seed = db.query("SELECT * FROM events WHERE source = 'seed' AND event_name = 'level_complete' LIMIT 1")[0]
    assert [e["event_name"] for e in live][:2] == ["session_start", "level_start"] and live[-1]["event_name"] == "level_complete"
    for e in live:   # same columns, same types, same NULL conventions as a seeded row
        assert set(e) == set(seed) and e["source"] == "live" and isinstance(json.loads(e["props"]), dict)
    done = live[-1]
    assert (done["level_id"], done["level_version"]) == (1, a["level_version"]) and done["attempt_id"] == a["attempt_id"]
    assert type(done["ts"]) is type(seed["ts"]) and type(done["level_id"]) is type(seed["level_id"])
    assert db.query("SELECT source, ab_group FROM players WHERE user_id = 'schema_probe'") == [{"source": "live", "ab_group": s["ab_group"]}]
    # the copilot's funnel scan (an AGENT query) sees the attempt the browser just produced
    f = next(r for r in client.get("/api/cc/funnel").json()["levels"] if r["level_id"] == 1)
    assert f["live_attempts"] >= 1
    scan = _wait_job(client, "copilot_scan")
    lv1 = next(r for r in scan["result"]["scan"]["levels"] if r["level_id"] == 1)
    assert scan["status"] == "done" and scan["result"]["scan"]["anomalies"][0]["level_id"] == 7 and lv1["attempts"] > 0


def test_live_experiment_exposures_are_evaluated_without_crashing(client):
    """Regression: live exposures are logged with day = NULL; evaluate_experiment sorted/compared them and raised
    TypeError as soon as real (non-simulated) exposure rows were old enough to be analysed."""
    db: TelemetryDB = client.app.state.db
    cop = LiveOpsCopilot(db, decide=lambda r: (True, "audit"), decided_by="audit", tracer=Tracer(InMemoryExporter()))
    with db.cursor() as con:
        ver = con.execute("SELECT coalesce(max(version), 0) + 1 FROM levels WHERE level_id = 2").fetchone()[0]
        con.execute("INSERT INTO levels SELECT * REPLACE (? AS version, 'candidate' AS status) FROM active_levels WHERE level_id = 2", [ver])
    assert cop.request_flag("live_days", 2, ver, 20.0)["ok"]
    # exposure logging by the REAL game endpoint (day is NULL, outcome NULL, source 'live')
    for i in range(40):
        s = client.post("/api/game/session", json={"user_id": f"exp_{i}"}).json()
        r = client.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": 2}).json()
        assert r["experiment"] == "live_days" and r["variant"] in ("control", "treatment")
    rows = db.query("SELECT variant, day, outcome, source FROM experiment_exposures WHERE flag = 'live_days'")
    assert len(rows) == 40 and {r["source"] for r in rows} == {"live"} and all(r["day"] is None for r in rows)
    assert 0 < sum(r["variant"] == "treatment" for r in rows) < 20                     # ~20% bucket, deterministic hash
    # age them so they are analysable (outcomes need a 7-day window), spread over 3 experiment days
    with db.cursor() as con:
        con.execute("""UPDATE experiment_exposures SET exposed_at = exposed_at - INTERVAL 10 DAY
                       + (hash(user_id) % 3) * INTERVAL 1 DAY WHERE flag = 'live_days'""")
    # (1) nobody has completed a level yet: every outcome is 0 -> no information, no decision, and no crash
    out = cop.daily_monitor("live_days")
    assert out["evaluation"]["recommendation"] == "continue" and out["evaluation"]["peeks"] == [] and out["action"] is None
    assert out["evaluation"]["days"] == 3                                               # days derived from exposed_at
    # (2) give the exposed players real post-exposure activity through the SAME events table the game writes
    exposed = db.query("SELECT user_id, exposed_at FROM experiment_exposures WHERE flag = 'live_days' ORDER BY user_id")
    rows = []
    for k, x in enumerate(exposed):
        for n in range(k % 4):                                                          # 0-3 completions in the window
            rows.append({"event_id": f"audit-{k}-{n}", "ts": x["exposed_at"] + timedelta(hours=2 + n), "user_id": x["user_id"],
                         "session_id": "audit", "event_name": "level_complete", "level_id": 2, "level_version": 1,
                         "attempt_id": f"audit-{k}-{n}", "ab_group": "gate_30", "platform": "web", "source": "live", "props": "{}"})
        if k % 2:                                                                       # half of them come back on D1
            rows.append({"event_id": f"audit-d1-{k}", "ts": x["exposed_at"] + timedelta(days=1), "user_id": x["user_id"],
                         "session_id": "audit", "event_name": "session_start", "ab_group": "gate_30", "platform": "web",
                         "source": "live", "props": "{}"})
    db.insert_events(rows)
    out = cop.daily_monitor("live_days")
    ev = out["evaluation"]
    assert ev["days"] == 3 and ev["peeks"] and [p["day"] for p in ev["peeks"]][-1] == 3
    assert ev["peeks"][-1]["n_t"] + ev["peeks"][-1]["n_c"] == 40                        # every live exposure was analysed
    assert ev["cuped"]["theta"] == 0.0 and ev["cuped"]["variance_reduction_pct"] == 0.0   # constant covariate: Welch fallback
    assert ev["recommendation"] == "continue" and out["action"] is None                 # 40 users decide nothing
    json.dumps(out)                                                                     # serialisable for the dashboard
    with cop.runner.workflow("audit:cleanup") as wf:
        wf.call("rollback_experiment", flag="live_days", reason="audit cleanup")
