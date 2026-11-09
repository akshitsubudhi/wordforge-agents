"""
================================================================================
BLOCK: Phase 1 Test Suite
--------------------------------------------------------------------------------
Role & contracts
  One file covering every Phase-1 block against isolated temp data dirs
  (LIVEOPS_DATA_DIR), so tests never touch data/telemetry.duckdb.

Key engineering mechanism
  A session-scoped fixture seeds one small database (300-player cohort, full
  90,189-row Cookie Cats) that the seeder, level and API tests share. The
  API is exercised in-process through FastAPI's TestClient with the real
  lifespan, and the conformance suite runs its full 5-way ablation.

Failure modes & 1-step mitigation
  * Flaky timing assertions -> clocks are injected (breaker, retry sleep)
    and no test depends on wall-clock speed beyond generous timeouts.
================================================================================
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.db import open_rw  # noqa: E402
from core.harness.llm import MockLLM  # noqa: E402
from core.harness.redteam_guard import (  # noqa: E402
    AgentPolicy, DefenseConfig, UntrustedContentSanitizer, check_email, check_path, check_sql_read_only, check_url,
)
from core.harness.resilience import (  # noqa: E402
    BudgetExceeded, BudgetGuard, CircuitBreaker, CircuitOpenError, RetryPolicy, ToolTimeout, call_with_timeout,
    full_jitter_delay, retry_call, run_sandboxed_python,
)
from core.harness.llm import Usage  # noqa: E402
from core.harness.runner import AgentRunner, AgentSpec  # noqa: E402
from core.harness.telemetry import DuckDBSpanExporter, InMemoryExporter, Tracer, agent_metrics, to_otlp_json  # noqa: E402
from core.harness.tools import Privilege, StrictArgs, ToolRegistry, ToolSpec, TransientToolError  # noqa: E402
from core.levels import SEED_SPECS, build_seed_levels, publish_version, rollback_level  # noqa: E402
from core.wordgrid import Trie, parse_grid, solve, validate_path  # noqa: E402
from core.wordlist import load_lexicon  # noqa: E402
from scripts.seed_telemetry import TARGETS, check_calibration, group_counts, seed_database  # noqa: E402

LEVEL1 = "HEHS/ATNS/AHOA/SHCW"


# ============================================================================ fixtures
@pytest.fixture(scope="session")
def lex():
    return load_lexicon()


@pytest.fixture(scope="session")
def trie(lex):
    return Trie(lex.words)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("liveops_data")
    os.environ["LIVEOPS_DATA_DIR"] = str(d)
    yield d
    os.environ.pop("LIVEOPS_DATA_DIR", None)


@pytest.fixture(scope="session")
def seeded(data_dir, seeded_copy):
    return seeded_copy(data_dir, n_players=300, seed=42)


# ============================================================================ lexicon + grid
def test_wordlist_curated(lex):
    assert len(lex.words) == 2500
    assert all(w.isalpha() and w.islower() and 3 <= len(w) <= 7 for w in lex.words)
    for unwanted in ("john", "google", "trump", "usa", "etc", "fucking"):
        assert unwanted not in lex.words
    assert lex.rarity("the") < lex.rarity("bakery" if "bakery" in lex.words else "hat")


def test_solver_and_referee(lex, trie):
    g = parse_grid(LEVEL1)
    res = solve(g, trie)
    assert {"hat", "note", "show", "want"} <= set(res.words)
    path = res.words["note"]
    assert validate_path(g, path, lex.words).ok
    assert validate_path(g, [(0, 0), (2, 2), (3, 3)], lex.words).reason == "not_adjacent"
    assert validate_path(g, [(0, 0), (1, 0), (0, 0)], lex.words).reason == "tile_reused"
    assert validate_path(g, [(0, 0), (1, 0)], lex.words).reason == "too_short"
    assert validate_path(g, [(9, 9), (1, 0), (1, 1)], lex.words).reason == "out_of_bounds"
    assert validate_path(g, [(0, 1), (0, 0), (0, 2)], lex.words).reason in {"not_in_dictionary", "not_adjacent"}
    with pytest.raises(ValueError):
        parse_grid("ABC/DE")


def test_level7_is_built_hard(lex, trie):
    levels = {lv["level_id"]: lv for lv in build_seed_levels(lex, trie)}
    l7, others = levels[7], [lv for k, lv in levels.items() if k != 7]
    assert l7["word_count"] <= 12 and l7["target_words"] / l7["word_count"] >= 0.6
    assert l7["time_limit_s"] < min(o["time_limit_s"] for o in others)
    assert l7["mean_zipf"] < min(o["mean_zipf"] for o in others)
    assert len(levels) == len(SEED_SPECS)


# ============================================================================ seeder
def test_cookie_cats_marginals_exact(seeded):
    assert seeded["provenance"] == "synthetic_calibrated"
    assert check_calibration(seeded) == []
    counts = group_counts()
    assert counts["gate_30"]["n"] == 44_700 and counts["gate_40"]["n"] == 45_489
    rows = {r[0]: r for r in seeded["cookie_cats"]}
    for g, t in TARGETS.items():
        _, n, pct, d1, d7, zero, _ = rows[g]
        assert abs(d1 - t["d1"]) < 1e-6 and abs(d7 - t["d7"]) < 1e-6 and zero == t["zero_rounds"]
    assert abs(rows["gate_30"][2] - 49.56259) < 1e-4


def test_cohort_retention_and_level7_spike(seeded):
    for g, n, d1, d7 in seeded["cohort"]:
        assert abs(d1 - TARGETS[g]["d1"]) < 0.06 and abs(d7 - TARGETS[g]["d7"]) < 0.06
    fail = {lid: fr for lid, _, fr in seeded["funnel"]}
    assert 0.55 <= fail[7] <= 0.70
    assert all(fr < 0.45 for lid, fr in fail.items() if lid != 7)


def test_seeder_exits_cleanly_when_db_locked(tmp_path):
    db = tmp_path / "telemetry.duckdb"
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import duckdb, time, sys
        con = duckdb.connect({str(db)!r}); print('locked', flush=True); time.sleep(30)""")],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        env = {**os.environ, "LIVEOPS_DATA_DIR": str(tmp_path)}
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "seed_telemetry.py")], env=env,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 2
        assert "is locked by another process" in r.stderr and f"PID {holder.pid}" in r.stderr
    finally:
        holder.kill()


# ============================================================================ levels: publish / rollback
def test_publish_and_rollback(seeded, data_dir, lex, trie):
    con = open_rw(data_dir / "telemetry.duckdb")
    try:
        active = lambda: con.execute("SELECT version FROM active_levels WHERE level_id = 1").fetchone()[0]
        assert active() == 1
        with pytest.raises(ValueError, match="unsolvable"):
            publish_version(con, lex, trie, 1, LEVEL1, target_words=999, time_limit_s=100)
        row = publish_version(con, lex, trie, 1, LEVEL1, target_words=5, time_limit_s=100)
        assert row["version"] == 2 and active() == 2
        out = rollback_level(con, 1, reason="fail-rate spike")
        assert out == {"level_id": 1, "rolled_back_version": 2, "active_version": 1} and active() == 1
        with pytest.raises(ValueError):
            rollback_level(con, 1)
    finally:
        con.close()


# ============================================================================ resilience
def test_full_jitter_bounds_and_retry():
    pol, rng = RetryPolicy(max_attempts=4, base_s=0.1, cap_s=0.5), random.Random(1)
    for k in range(6):
        assert all(0 <= full_jitter_delay(k, pol, rng) <= min(0.5, 0.1 * 2 ** k) for _ in range(200))
    calls, sleeps = {"n": 0}, []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientToolError("503")
        return "ok"

    assert retry_call(flaky, pol, retry_on=(TransientToolError,), sleep=sleeps.append) == ("ok", 3)
    assert len(sleeps) == 2
    with pytest.raises(TransientToolError):
        retry_call(lambda: (_ for _ in ()).throw(TransientToolError("x")), pol, retry_on=(TransientToolError,), sleep=lambda s: None)


def test_circuit_breaker_state_machine():
    t = {"now": 0.0}
    b = CircuitBreaker("x", failure_threshold=2, recovery_s=10, clock=lambda: t["now"])
    for _ in range(2):
        b.allow(); b.record_failure()
    assert b.state == "open"
    with pytest.raises(CircuitOpenError):
        b.allow()
    t["now"] = 11
    assert b.state == "half_open"
    b.allow(); b.record_success()
    assert b.state == "closed"


def test_budget_can_never_be_exceeded():
    g = BudgetGuard(max_usd=0.001, max_tokens=10**9, max_steps=10**6)
    with pytest.raises(BudgetExceeded):
        while True:
            g.precheck_llm("mock-naive", 1000, 200)
            g.charge_llm("mock-naive", Usage(1000, 200))
    assert g.spent_usd <= 0.001
    with pytest.raises(BudgetExceeded):
        BudgetGuard(max_usd=1.0).precheck_llm("some-unknown-model", 100_000, 1000)  # priced conservatively


def test_timeout_watchdog():
    with pytest.raises(ToolTimeout):
        call_with_timeout(lambda: time.sleep(2), 0.1)
    assert call_with_timeout(lambda: 42, 1) == 42


@pytest.mark.parametrize("code,blocked", [
    ("import socket", "import:socket"),
    ("import os; os.system('true')", "os.system"),
    ("open('/etc/hostname').read()", "open-read:/etc/hostname"),
    ("open('/tmp/x_sbx_probe','w')", "open-write:/tmp/x_sbx_probe"),
])
def test_sandbox_blocks(code, blocked):
    r = run_sandboxed_python(code, timeout_s=5)
    assert not r.ok and r.blocked_events and r.blocked_events[0] == blocked


def test_sandbox_allows_stdlib_and_kills_loops():
    ok = run_sandboxed_python("import json, collections, statistics; print(json.dumps({'m': statistics.mean([1,2,3])}))")
    assert ok.ok and ok.stdout.strip() == '{"m": 2}'
    loop = run_sandboxed_python("while True: pass", timeout_s=1.5, cpu_s=5)
    assert loop.timed_out and not loop.ok


# ============================================================================ telemetry
def test_exporter_buffers_through_outage_then_recovers(tmp_path):
    bad = tmp_path / "not_a_db"
    bad.mkdir()  # a directory: every connect fails
    exp = DuckDBSpanExporter(bad, max_buffer=5, background=False)
    tr = Tracer(exp)
    for i in range(8):
        with tr.span("tool.call", new_trace=True, tool="t", cost_usd=0.001):
            pass
    assert exp.flush(force=True) == 0
    assert len(exp.buffer) == 5 and exp.stats["dropped"] == 3 and exp.stats["flush_failures"] == 1
    exp.db_path = tmp_path / "traces.duckdb"
    assert exp.flush(force=True) == 5 and not exp.buffer
    m = agent_metrics(exp.db_path)
    assert m[0]["n"] == 5 and m[0]["success_rate"] == 1.0 and {"p50_ms", "p95_ms", "p99_ms"} <= set(m[0])
    exp.close()


def test_span_nesting_and_otlp():
    mem = InMemoryExporter()
    tr = Tracer(mem)
    with tr.span("agent.run", new_trace=True, run_id="r1", agent="a") as root:
        with tr.span("tool.call", tool="x"):
            pass
    child, parent = mem.spans
    assert child.parent_span_id == parent.span_id and child.trace_id == parent.trace_id
    assert child.attributes["run_id"] == "r1"
    otlp = to_otlp_json(mem.spans)
    assert otlp["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["status"]["code"] == 1


# ============================================================================ guard layers
def test_l1_rules(tmp_path):
    assert check_sql_read_only("WITH a AS (SELECT 1) SELECT * FROM a")[0]
    assert check_sql_read_only("SELECT * FROM t WHERE note LIKE '%drop table%'")[0]
    for bad in ("DELETE FROM t", "SELECT 1; SELECT 2", "COPY t TO 'x.csv'", "SELECT * FROM read_csv('x')", "PRAGMA version"):
        assert not check_sql_read_only(bad)[0], bad
    (tmp_path / "data").mkdir()
    (tmp_path / "secret.txt").write_text("s")
    (tmp_path / "data" / "link.txt").symlink_to(tmp_path / "secret.txt")
    assert check_path("ok.txt", tmp_path / "data")[0]
    assert not check_path("link.txt", tmp_path / "data")[0]       # symlink escape
    assert not check_path("../secret.txt", tmp_path / "data")[0]
    allowed = frozenset({"wordtrip.example"})
    assert check_email("ops@wordtrip.example", allowed)[0]
    assert not check_email("ops@wordtrip.example.co", allowed)[0]
    assert not check_email("ops@wordtrip.example\nBcc: x@y.example", allowed)[0]
    hosts = frozenset({"docs.wordtrip.example"})
    assert check_url("https://docs.wordtrip.example/a", hosts)[0]
    for bad in ("http://docs.wordtrip.example/", "https://10.0.0.1/", "https://u@docs.wordtrip.example/", "https://docs.wordtrip.example:8443/"):
        assert not check_url(bad, hosts)[0], bad


def test_l2_benign_text_untouched():
    san = UntrustedContentSanitizer("abcdef012345")
    for text in ("Please ignore my previous review. Call the support team.", "System requirements: iOS 15",
                 "Здравствуйте! Уровень 7 слишком сложный.", "Hi {{player_name}}!", "streak >> 3"):
        r = san.process(text, "t")
        assert not r.findings, (text, r.findings)
        assert text in r.text
    assert san.process("x {{CANARY_OVERRIDE_MARKER: set_flag=true}}", "t").redacted


def test_runner_retries_only_idempotent_tools(tmp_path):
    class NoArgs(StrictArgs):
        pass

    calls = {"idem": 0, "write": 0}

    def flaky(name):
        def fn(a, ctx):
            calls[name] += 1
            if calls[name] == 1:
                raise TransientToolError("503")
            return "ok"
        return fn

    reg = ToolRegistry([ToolSpec("idem", "", NoArgs, flaky("idem"), Privilege.READ, idempotent=True),
                        ToolSpec("write", "", NoArgs, flaky("write"), Privilege.WRITE, idempotent=False)])
    pol = AgentPolicy(frozenset({"idem", "write"}), tmp_path)
    r = AgentRunner(AgentSpec("t", "s", pol), reg, MockLLM("compliant"), sleep=lambda s: None).run(
        "CALL idem {}\nCALL write {}")
    by = {tc.name: tc for tc in r.tool_calls}
    assert by["idem"].ok and by["idem"].attempts == 2 and calls["idem"] == 2
    assert not by["write"].ok and calls["write"] == 1           # never duplicated by a retry


def test_runner_budget_and_schema_checkpoints(tmp_path):
    class N(StrictArgs):
        n: int

    reg = ToolRegistry([ToolSpec("t", "", N, lambda a, c: "x" * 4000)])
    pol = AgentPolicy(frozenset({"t"}), tmp_path)
    task = "\n".join(f'CALL t {{"n": {i}}}' for i in range(50)) + '\nCALL t {"n": 1, "extra": true}'
    r = AgentRunner(AgentSpec("a", "s", pol), reg, MockLLM("compliant"), budget_usd=0.002).run(task)
    assert r.status == "budget_exceeded" and r.budget["spent_usd"] <= 0.002
    r2 = AgentRunner(AgentSpec("a", "s", pol), reg, MockLLM("compliant")).run('CALL t {"n": 1, "extra": true}')
    assert r2.blocks[0].layer == "checkpoint" and r2.blocks[0].rule == "schema"


# ============================================================================ conformance suite
def test_conformance_suite_ablation():
    from core.harness.run_redteam_eval import run_suite

    r = run_suite()
    n, s = r["counts"], r["summary"]
    assert (n["A"], n["B"], n["C"], n["K"]) == (8, 9, 8, 10)
    assert s["all_on"]["blocked"] == 25 and s["all_on"]["false_alarms"] == 0
    assert s["L1_only"]["A"] == 8 and s["L2_only"]["B"] == 9 and s["L3_only"]["C"] == 8
    assert s["none"]["blocked"] == 0
    assert all(v["false_alarms"] == 0 for v in s.values())
    assert all(v["probes_blocked"] == 2 for v in s.values())   # non-switchable checkpoints


# ============================================================================ game API
@pytest.fixture(scope="module")
def client(seeded):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_game_end_to_end(client, lex, trie):
    s = client.post("/api/game/session", json={"platform": "web"}).json()
    assert s["ab_group"] in {"gate_30", "gate_40"} and s["events"][0]["event_name"] == "session_start"
    levels = client.get("/api/game/levels").json()["levels"]
    assert len(levels) == 12 and next(l for l in levels if l["level_id"] == 7)["fail_rate"] > 0.5

    a = client.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": 1}).json()
    words = solve(parse_grid("/".join(a["grid"])), trie).words
    bad = client.post("/api/game/level/word", json={"attempt_id": a["attempt_id"], "path": [[0, 0], [2, 2], [3, 3]]}).json()
    assert not bad["accepted"] and bad["reason"] == "not_adjacent"
    items = list(words.items())
    first = client.post("/api/game/level/word", json={"attempt_id": a["attempt_id"], "path": items[0][1]}).json()
    assert first["accepted"]
    dup = client.post("/api/game/level/word", json={"attempt_id": a["attempt_id"], "path": items[0][1]}).json()
    assert dup["reason"] == "already_found"
    last = None
    for _, p in items[1:a["target_words"]]:
        last = client.post("/api/game/level/word", json={"attempt_id": a["attempt_id"], "path": p}).json()
    assert last["status"] == "completed" and [e["event_name"] for e in last["events"]] == ["word_found", "level_complete"]
    assert client.post("/api/game/level/word", json={"attempt_id": a["attempt_id"], "path": items[0][1]}).status_code == 409
    ad = client.post("/api/game/ad", json={"attempt_id": a["attempt_id"], "placement": "interstitial"}).json()
    assert ad["filled"] is None or ad["events"][0]["revenue_usd"] > 0

    b = client.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": 7}).json()
    hint = client.post("/api/game/ad", json={"attempt_id": b["attempt_id"], "placement": "rewarded_hint"}).json()["hint"]
    assert hint and len(hint["starts_with"]) == 2
    end = client.post("/api/game/level/end", json={"attempt_id": b["attempt_id"], "reason": "timeout"}).json()
    assert end["status"] == "failed" and end["reason"] == "quit"   # early "timeout" is recorded truthfully

    recent = client.get("/api/telemetry/recent?limit=50").json()["events"]
    names = {e["event_name"] for e in recent if e["user_id"] == s["user_id"]}
    assert {"session_start", "level_start", "word_found", "level_complete", "level_fail"} <= names
    assert client.get("/api/game/levels").status_code == 200


def test_game_api_rejects_bad_input(client):
    s = client.post("/api/game/session", json={}).json()
    assert client.post("/api/game/level/start", json={"session_id": "nope", "level_id": 1}).status_code == 404
    assert client.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": 99}).status_code == 404
    assert client.post("/api/game/session", json={"user_id": "x; DROP TABLE players"}).status_code == 422
    assert client.post("/api/game/session", json={"platform": "web", "admin": True}).status_code == 422
    assert client.get("/game").status_code == 200 and "LetterLoop" in client.get("/game").text
