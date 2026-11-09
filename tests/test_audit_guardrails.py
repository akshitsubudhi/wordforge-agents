"""
================================================================================
BLOCK: Audit Suite 2/3 - Guardrail & Mitigation Proofs
--------------------------------------------------------------------------------
Role & contracts
  One section per production failure mode. Every section has the same shape:
    1 inject the fault through the REAL component (no mocked guardrail),
    2 assert the guardrail blocks it AND the protected state is unchanged,
    3 a CONTROL: the same path without the fault (or without the guard)
      behaves differently, so the test cannot pass vacuously.
    a) unsolvable word grid          -> deterministic solver proof, release
                                        gate, DB-layer re-solve
    b) peeking false-positive A/B    -> mSPRT always-valid bounds + the
                                        copilot's ship gate
    c) runaway token / cost loop     -> BudgetExceededError (USD, tokens,
                                        steps), raised BEFORE the spend
    d) ad network going dark         -> automatic rollback to the baseline
                                        floor config + quarantine

Failure modes & 1-step mitigation
  * Shared state between sections -> every section builds its own temp
    DuckDB; the module restores LIVEOPS_DATA_DIR on teardown.
================================================================================
"""
from __future__ import annotations

import json
import math
import os
import random

import pytest

from agents.ad_economy.auction_sim import MediationSimulator
from agents.ad_economy.bandit_agent import AdOpsAgent, EWMADetector
from agents.level_factory.critic_and_eval import LevelFactory
from agents.level_factory.solver import LevelSolver
from agents.liveops_copilot.agent import LiveOpsCopilot, simulated_operator
from agents.liveops_copilot.simulator import Effect, ExperimentSimulator
from agents.liveops_copilot.stats_engine import MSPRT
from app import fault_drills as fd
from core.db import TelemetryDB, open_rw
from core.harness.llm import LLMResponse, ToolCall, Usage
from core.harness.redteam_guard import AgentPolicy
from core.harness.resilience import BudgetExceeded, BudgetExceededError, BudgetGuard
from core.harness.runner import AgentRunner, AgentSpec, ToolCallBlocked
from core.harness.telemetry import InMemoryExporter, Tracer
from core.harness.tools import Privilege, StrictArgs, ToolRegistry, ToolSpec
from core.levels import publish_version
from core.wordlist import load_lexicon
from scripts.seed_telemetry import seed_database


# ============================================================================ fixtures
@pytest.fixture(scope="module")
def lex():
    return load_lexicon()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, lex):
    d = tmp_path_factory.mktemp("audit_guardrails")
    prev = os.environ.get("LIVEOPS_DATA_DIR")
    os.environ["LIVEOPS_DATA_DIR"] = str(d)
    con = open_rw(d / "telemetry.duckdb")
    seed_database(con, n_players=400, seed=3, lex=lex)
    con.close()
    yield d
    if prev is None:
        os.environ.pop("LIVEOPS_DATA_DIR", None)
    else:
        os.environ["LIVEOPS_DATA_DIR"] = prev


@pytest.fixture(scope="module")
def db(data_dir):
    d = TelemetryDB(data_dir / "telemetry.duckdb")
    yield d
    d.close()


@pytest.fixture(scope="module")
def mem():
    return InMemoryExporter()


@pytest.fixture(scope="module")
def factory(db, lex, mem):
    return LevelFactory(db, lex=lex, tracer=Tracer(mem), seed=7)


def levels_snapshot(db: TelemetryDB) -> list[tuple]:
    with db.cursor() as con:
        return con.execute("SELECT level_id, version, grid, status FROM levels ORDER BY 1, 2").fetchall()


# ============================================================================ a) unsolvable word grid
UNSOLVABLE = [
    # (grid, target_words, claimed words, expected reason prefix)
    ("PLNT/ERAQ/ZSBK/WJVM", 20, ["plant", "planet", "zebra", "quartz"], "targets_missing"),   # hallucinated words
    ("PLNT/ERAQ/ZSBK/WJVM", 20, [], "unsolvable"),                                             # target > words on board
    ("BCDF/GHJK/LMNP/QRSV", 3, [], "no_words_on_board"),                                       # no vowels: nothing to find
    ("ABCD/EFG/HIJK/LMNO", 3, [], "malformed_grid"),                                           # ragged rows
    ("AB1D/EFGH/IJKL/MNOP", 3, [], "malformed_grid"),                                          # non-letter tile
]


@pytest.mark.parametrize("grid,target,claimed,reason", UNSOLVABLE)
def test_a_solver_rejects_every_kind_of_unsolvable_grid(lex, grid, target, claimed, reason):
    proof = LevelSolver(lex).prove(grid, claimed, target)
    assert proof.ok is False and proof.reason.startswith(reason), proof.reason
    if reason == "targets_missing":
        assert "plant" not in proof.targets_missing and set(proof.targets_missing) == {"planet", "zebra", "quartz"}
    if reason == "unsolvable":
        assert proof.word_count < target        # the solver COUNTED the words; the verdict follows from the count


def test_a_solver_verdict_is_exhaustive_not_a_heuristic(lex):
    """Brute force (every simple path up to length 5, no trie) finds exactly the solver's words of that length."""
    from core.wordgrid import neighbours, parse_grid

    solver = LevelSolver(lex)
    proof = solver.prove("PLNT/ERAQ/ZSBK/WJVM")
    g = parse_grid(proof.grid)
    found: set[str] = set()

    def walk(path):
        w = "".join(g[r][c] for r, c in path).lower()
        if len(w) >= 3 and w in lex.words:
            found.add(w)
        if len(path) < 5:
            for nxt in neighbours(*path[-1], 4, 4):
                if nxt not in path:
                    walk(path + [nxt])

    for r in range(4):
        for c in range(4):
            walk([(r, c)])
    assert found == {w for w in proof.valid_words if len(w) <= 5}
    assert proof.word_count == 12 and proof.witness_rechecked


def test_a_unsolvable_level_cannot_reach_the_live_game(db, factory, lex):
    before = levels_snapshot(db)
    bad = {"grid": "PLNT/ERAQ/ZSBK/WJVM", "target_words": 20, "time_limit_s": 90}
    with factory.runner.workflow("audit:unsolvable") as wf:
        proof = wf.call("prove_level", **bad)
        assert proof["ok"] is False and proof["reason"].startswith("unsolvable")
        crit = wf.call("critique_level", **bad, goal=0.3)
        assert crit["approve"] is False and crit["scores"]["solvability_fairness"] == 1
        rec = wf.call_record("publish_level", level_id=0, status="live", notes="audit", **bad)          # gate 2
        assert not rec.executed and not rec.ok and rec.decision.rule == "approval_denied"
        assert "solver proof failed" in rec.decision.detail
        with pytest.raises(ToolCallBlocked):                                                             # never critiqued
            wf.call("publish_level", level_id=0, grid="BCDF/GHJK/LMNP/QRSV", target_words=3, time_limit_s=90, status="live")
    with db.cursor() as con:                                                                             # gate 3: DB layer
        with pytest.raises(ValueError, match="unsolvable"):
            publish_version(con, lex, factory.solver.trie, 999, bad["grid"], bad["target_words"], bad["time_limit_s"])
    assert levels_snapshot(db) == before                      # nothing was written anywhere
    denied = db.query("SELECT decision, reason FROM approvals WHERE tool = 'publish_level' ORDER BY ts DESC LIMIT 2")
    assert [d["decision"] for d in denied] == ["denied", "denied"]   # both refusals are on the approval ledger


def test_a_control_a_solvable_level_passes_the_same_gates(db, factory):
    """The gates are not a blanket 'deny': a proven, critic-approved level IS published."""
    shipped = factory.ship_daily("2031-01-01", candidates_per_slot=30)
    assert shipped, "no slot filled"
    lv = shipped[0]
    row = db.query("SELECT grid, target_words, word_count, status FROM levels WHERE level_id = ? AND version = ?",
                   [lv["level_id"], lv["version"]])[0]
    proof = factory.solver.prove(row["grid"], target_count=row["target_words"])
    assert row["status"] == "live" and proof.ok and proof.witness_rechecked and row["target_words"] <= proof.word_count
    fd.drill_hallucinated_level(db, factory, lambda s: None)   # the packaged drill agrees
    assert fd.drill_hallucinated_level(db, factory, lambda s: None)["blocked"] is True


# ============================================================================ b) peeking false positive
def _find_peeking_false_positive(alpha: float = 0.05):
    for seed in range(1, 500):
        t = fd._aa_trajectory(seed, days=14, n_per_day=400, alpha=alpha, tau=0.1)
        if t["naive_ship_day"]:
            return seed, t
    raise AssertionError("no naive false positive in 500 A/A runs")


def test_b_msprt_bounds_block_the_exact_peek_where_a_naive_test_ships_a_null_effect():
    seed, t = _find_peeking_false_positive()
    day = t["naive_ship_day"]
    row = t["rows"][day - 1]
    assert row["naive_p"] < 0.05 and not (row["naive_ci"][0] <= 0 <= row["naive_ci"][1])   # naive test: "ship it"
    assert row["av_p"] > 0.05 and row["av_ci"][0] <= 0 <= row["av_ci"][1]                  # mSPRT: no decision
    assert not row["msprt_reject"] and not t["msprt_ever_rejects"]                         # ...on any of the 14 peeks
    assert all(r["av_ci"][0] <= 0 <= r["av_ci"][1] for r in t["rows"])                     # 0 never leaves the CI
    assert all(r["av_ci"][1] - r["av_ci"][0] > r["naive_ci"][1] - r["naive_ci"][0] for r in t["rows"])


def test_b_false_positive_rate_is_bounded_by_alpha_at_every_alpha():
    for alpha in (0.10, 0.05, 0.01):
        rng = random.Random(100 + int(alpha * 1000))
        sims, naive_fp, msprt_fp = 600, 0, 0
        zc = {0.10: 1.6449, 0.05: 1.96, 0.01: 2.5758}[alpha]
        for _ in range(sims):
            test = MSPRT(alpha=alpha, tau=0.1, min_samples=1)
            st = sc = sst = ssc = 0.0
            n, naive, seq = 0, False, False
            for _ in range(20):                       # 20 peeks
                for _ in range(100):
                    a, b = rng.gauss(0, 1), rng.gauss(0, 1)
                    st += a; sst += a * a; sc += b; ssc += b * b
                n += 100
                mt, mc = st / n, sc / n
                var = ((sst - n * mt * mt) + (ssc - n * mc * mc)) / (n - 1) / n
                naive = naive or abs((mt - mc) / math.sqrt(var)) > zc
                seq = seq or test.update(mt - mc, var, n, n).reject_null
            naive_fp += naive
            msprt_fp += seq
        assert msprt_fp / sims <= alpha, (alpha, msprt_fp / sims)
        assert naive_fp / sims > 2.5 * alpha, (alpha, naive_fp / sims)


def test_b_copilot_never_ships_an_aa_test_however_often_it_peeks(db, factory, mem):
    """End to end: a null variant, peeked daily for 14 days through evaluate_experiment -> ship gate."""
    ref: list = []
    cop = LiveOpsCopilot(db, factory=factory, decide=simulated_operator({"aa_null", "aa_real"}, ref),
                         decided_by="audit", tracer=Tracer(mem))
    ref.append(cop)
    launch = cop.launch("aa_null", 7, goal_fail_rate=0.35)
    assert launch["blocked"] is None and launch["flag_result"]["status"] == "active"
    sim = ExperimentSimulator(db, seed=5)
    recs = []
    for day in range(1, 15):
        sim.simulate_day("aa_null", day, users=1000, treatment_share=0.2, effect=Effect(0.0, 0.0))
        out = cop.daily_monitor("aa_null", max_days=14)
        ev = out["evaluation"]
        recs.append(ev["recommendation"])
        lo, hi = ev["peeks"][-1]["ci"]
        assert lo <= 0 <= hi, f"day {day}: always-valid CI [{lo}, {hi}] excludes the true effect 0"
        assert out["action"] is None or out["action"].get("status") != "shipped"
    assert "ship" not in recs and "rollback" not in recs and recs[-1] == "stop_inconclusive"
    status = db.query("SELECT status FROM feature_flags WHERE flag = 'aa_null'")[0]["status"]
    assert status == "active"                                   # still an experiment: nothing was promoted
    # even a direct ship request is refused by the approval gate (no 'ship' recommendation on record)
    with cop.runner.workflow("audit:force_ship") as wf:
        rec = wf.call_record("ship_experiment", flag="aa_null")
    assert not rec.executed and rec.decision.rule == "approval_denied"
    cop.daily_monitor("aa_null", max_days=14)
    with cop.runner.workflow("audit:cleanup") as wf:            # leave level 7 clean for the next test
        wf.call("rollback_experiment", flag="aa_null", reason="audit cleanup")


def test_b_control_a_real_effect_is_detected_and_shipped(db, factory, mem):
    """The bound is not 'never reject': the same pipeline ships a variant with a true +20% effect."""
    ref: list = []
    cop = LiveOpsCopilot(db, factory=factory, decide=simulated_operator({"aa_real"}, ref), decided_by="audit", tracer=Tracer(mem))
    ref.append(cop)
    assert cop.launch("aa_real", 7, goal_fail_rate=0.35)["blocked"] is None
    sim = ExperimentSimulator(db, seed=6)
    out = None
    for day in range(1, 15):
        sim.simulate_day("aa_real", day, users=1500, treatment_share=0.2, effect=Effect(0.20, 0.0))
        out = cop.daily_monitor("aa_real")
        if out["evaluation"]["recommendation"] == "ship":
            break
    assert out["evaluation"]["recommendation"] == "ship" and out["evaluation"]["days"] >= 7
    assert out["action"]["status"] == "shipped" and out["evaluation"]["peeks"][-1]["ci"][0] > 0


# ============================================================================ c) runaway token / cost loop
class PageArgs(StrictArgs):
    page: int


class LoopingLLM:
    """A model that NEVER stops: every turn it asks for one more tool call. Counts how often it was invoked."""

    def __init__(self, model: str = "mock-naive", out_tokens: int = 200):
        self.model, self.out_tokens, self.calls = model, out_tokens, 0

    def complete(self, messages, tools, max_output_tokens=1024) -> LLMResponse:
        self.calls += 1
        in_tok = sum(len(m.content) for m in messages) // 4 + 1
        return LLMResponse("", [ToolCall(f"c{self.calls}", "fetch_page", {"page": self.calls})], Usage(in_tok, self.out_tokens), self.model)


def _loop_runner(tmp_path, llm, **kw) -> tuple[AgentRunner, list[int]]:
    fetched: list[int] = []

    def fetch(a: PageArgs, ctx) -> str:
        fetched.append(a.page)
        return "row data " * 200

    reg = ToolRegistry([ToolSpec("fetch_page", "Fetch a page", PageArgs, fetch, Privilege.READ)])
    spec = AgentSpec("crawler", "You export pages.", AgentPolicy(frozenset({"fetch_page"}), tmp_path))
    return AgentRunner(spec, reg, llm, tracer=Tracer(InMemoryExporter()), **kw), fetched


def test_c_budget_guard_raises_budget_exceeded_error_before_the_money_is_spent():
    assert BudgetExceededError is BudgetExceeded and issubclass(BudgetExceededError, RuntimeError)
    g = BudgetGuard(max_usd=0.01, max_tokens=10_000, max_steps=3)
    g.charge_llm("claude-sonnet-5-5", Usage(1000, 400))                 # $0.003 + $0.006 = $0.009 spent
    assert g.spent_usd == pytest.approx(0.009)
    with pytest.raises(BudgetExceededError) as e:                       # worst case of the NEXT call would pass the cap
        g.precheck_llm("claude-sonnet-5-5", 100, 100)
    assert e.value.kind == "usd" and g.spent_usd == pytest.approx(0.009)   # refused, nothing charged
    with pytest.raises(BudgetExceededError) as e:
        BudgetGuard(max_usd=10, max_tokens=1000).precheck_llm("mock-naive", 900, 200)
    assert e.value.kind == "tokens"
    for _ in range(3):
        g.charge_step()
    with pytest.raises(BudgetExceededError) as e:
        g.charge_step()
    assert e.value.kind == "steps"
    # an unknown model is priced HIGH, so the breaker errs on the side of stopping
    with pytest.raises(BudgetExceededError):
        BudgetGuard(max_usd=0.05).precheck_llm("some-unreleased-model", 1000, 1000)
    BudgetGuard(max_usd=0.05).precheck_llm("gpt-4o-mini", 1000, 1000)   # the same call on a known cheap model fits


@pytest.mark.parametrize("model,cap", [("mock-naive", 0.002), ("claude-sonnet-5-5", 0.05), ("some-unreleased-model", 0.25)])
def test_c_runaway_loop_is_killed_and_spend_never_passes_the_cap(tmp_path, model, cap):
    llm = LoopingLLM(model)
    runner, fetched = _loop_runner(tmp_path, llm, budget_usd=cap, max_steps=10_000)
    res = runner.run("Export everything.")
    assert res.status == "budget_exceeded" and res.stop_reason.startswith("usd:")
    assert 0 < res.budget["spent_usd"] <= cap                           # hard cap: never exceeded, not merely detected
    calls_at_stop = llm.calls
    assert calls_at_stop == res.budget["llm_calls"] == len(fetched) < 10_000   # the model was NOT called again
    assert len(res.tool_calls) == len(fetched) and all(tc.ok for tc in res.tool_calls)   # partial work preserved
    # control: the identical loop with a 10x cap runs strictly longer -> the cap is what stopped it
    llm2 = LoopingLLM(model)
    runner2, _ = _loop_runner(tmp_path, llm2, budget_usd=cap * 10, max_steps=10_000)
    assert runner2.run("Export everything.").status == "budget_exceeded" and llm2.calls > calls_at_stop


def test_c_token_cap_and_step_cap_also_kill_the_loop(tmp_path):
    llm = LoopingLLM()
    runner, _ = _loop_runner(tmp_path, llm, budget_usd=1e9, budget_tokens=20_000, max_steps=10_000)
    res = runner.run("Export everything.")
    assert res.status == "budget_exceeded" and res.stop_reason.startswith("tokens:") and res.budget["tokens"] <= 20_000
    llm = LoopingLLM()
    runner, fetched = _loop_runner(tmp_path, llm, budget_usd=1e9, budget_tokens=10**12, max_steps=5)
    res = runner.run("Export everything.")
    assert res.status == "budget_exceeded" and res.stop_reason.startswith("steps:") and llm.calls == 5 == len(fetched)


def test_c_env_cap_applies_to_every_runner_by_default(tmp_path, monkeypatch):
    """HARNESS_MAX_RUN_USD was documented but never read by the runner: it is now the default cap."""
    monkeypatch.setenv("HARNESS_MAX_RUN_USD", "0.0005")
    llm = LoopingLLM()
    runner, _ = _loop_runner(tmp_path, llm, max_steps=10_000)
    assert runner.budget_usd == 0.0005
    res = runner.run("Export everything.")
    assert res.status == "budget_exceeded" and res.budget["spent_usd"] <= 0.0005
    monkeypatch.setenv("HARNESS_MAX_RUN_TOKENS", "3000")
    runner, _ = _loop_runner(tmp_path, LoopingLLM(), budget_usd=1e9, max_steps=10_000)
    assert runner.budget_tokens == 3000 and runner.run("Export everything.").stop_reason.startswith("tokens:")


def test_c_runaway_drill_with_the_naive_model_and_every_other_defence_off():
    logged: list[dict] = []
    r = fd.drill_runaway_cost(Tracer(InMemoryExporter()), logged.append, budget_usd=0.002, uncapped_steps=40)
    capped, uncapped = r["runs"]["budget breaker"], r["runs"]["uncapped"]
    assert r["blocked"] and capped["status"] == "budget_exceeded" and capped["spent_usd"] <= 0.002
    assert uncapped["tool_calls"] == 40 and uncapped["spent_usd"] > 10 * capped["spent_usd"]   # the damage prevented
    assert any(s["status"] == "guard" and "BudgetExceededError" in s["title"] for s in logged)


# ============================================================================ d) ad network goes dark
BASELINE_FLOOR = 2.0


def _ad_agent(tmp_path, name: str, guard: bool = True, seed: int = 21):
    db = TelemetryDB(tmp_path / f"{name}.duckdb")
    sim = MediationSimulator(seed=seed)
    mem = InMemoryExporter()
    return db, sim, AdOpsAgent(db, sim, tracer=Tracer(mem), guard_enabled=guard, run_id=name), mem


def _configs(db: TelemetryDB) -> list[dict]:
    return db.query("SELECT version, reserve_cpm, disabled, status, parent, reason FROM ad_mediation_config ORDER BY version")


def test_d_dark_network_during_a_floor_trial_rolls_back_to_the_baseline_floor(tmp_path):
    db, sim, agent, mem = _ad_agent(tmp_path, "dark_trial")
    ecpm = {}
    for h in range(24):
        if h == 10:
            assert agent.apply_floor(6.0, "TS-tuned floor (trial)")["ok"]
        if h == 12:
            assert agent.active() == {"version": 2, "reserve_cpm": 6.0, "disabled": [], "status": "trial"}
            assert agent.baseline() == {"version": 1, "reserve_cpm": BASELINE_FLOOR}     # v1 was promoted to 'good'
            sim.set_dark("admob")
        r = agent.step(h, 1500)
        ecpm[h] = r["ecpm"]
        if h == 12:                                                    # mitigated in the SAME hour it went dark
            kinds = [("rollback" if a.get("cause") == "network_dark" else "quarantine") for a in r["actions"]]
            assert kinds == ["rollback", "quarantine"], r["actions"]
            assert r["actions"][0]["from_reserve_cpm"] == 6.0 and r["actions"][0]["restored_reserve_cpm"] == BASELINE_FLOOR
            now = agent.active()
            assert now["reserve_cpm"] == BASELINE_FLOOR and now["disabled"] == ["admob"]
    cfgs = _configs(db)
    assert cfgs[1]["status"] == "rolled_back" and "admob went dark" in cfgs[1]["reason"]        # the $6 trial
    assert cfgs[2]["status"] in ("good", "superseded") and cfgs[2]["reserve_cpm"] == BASELINE_FLOOR and cfgs[2]["parent"] == 2
    assert all(c["reserve_cpm"] == BASELINE_FLOOR for c in cfgs[2:])                              # floor stays at baseline
    assert agent.active()["reserve_cpm"] == BASELINE_FLOOR
    audit = {a["action"]: json.loads(a["detail"]) for a in db.query("SELECT action, detail FROM agent_audit ORDER BY ts")}
    assert audit["rollback_config"]["cause"] == "network_dark" and audit["rollback_config"]["hour"] == 12
    assert audit["quarantine_network"]["network"] == "admob"
    spans = [s for s in mem.spans if s.name == "tool.call" and s.attributes.get("tool") == "rollback_config"]
    assert len(spans) == 1 and spans[0].status == "OK"                                           # went through the harness
    assert ecpm[12] < 0.7 * ecpm[11] and ecpm[13] > 0.85 * ecpm[9]                               # hit, then recovered
    with db.cursor() as con:
        fills = con.execute("SELECT sum(fills) FROM ad_hourly_metrics WHERE network = 'admob' AND hour >= 13 "
                            "AND config_version NOT IN (SELECT version FROM ad_mediation_config WHERE disabled = '[]')").fetchone()[0]
    assert not fills                                                    # no traffic waits on the dead network
    db.close()


def test_d_dark_in_the_same_hour_as_the_floor_push_is_still_caught(tmp_path):
    """Regression: the detectors re-baseline on every config change; an outage in that hour was never detected."""
    db, sim, agent, _ = _ad_agent(tmp_path, "dark_same_hour")
    for h in range(16):
        if h == 10:
            agent.apply_floor(6.0, "trial")
            sim.set_dark("admob")
        r = agent.step(h, 1500)
        if h == 10:
            assert [a.get("cause", "quarantine") for a in r["actions"]] == ["network_dark", "quarantine"]
    assert agent.active()["reserve_cpm"] == BASELINE_FLOOR and agent.active()["disabled"] == ["admob"]
    d = EWMADetector(warmup=4, seen_positive=True)          # unit level: first sample of a fresh warmup is 0
    s = d.update(0.0)
    assert s.alarm and s.reason == "dark" and s.mean is None
    assert not EWMADetector(warmup=4).update(0.0).alarm     # a series that never filled is not an "outage"
    db.close()


def test_d_outage_before_any_config_was_promoted_falls_back_to_the_initial_baseline(tmp_path):
    db, sim, agent, _ = _ad_agent(tmp_path, "dark_early")
    for h in range(6):
        if h == 2:
            agent.apply_floor(8.0, "early trial")           # v1 has not survived 6 clean hours: no 'good' config yet
        if h == 4:
            assert not db.query("SELECT 1 FROM ad_mediation_config WHERE status = 'good'")
            sim.set_dark("admob")
        agent.step(h, 1500)
    assert agent.active()["reserve_cpm"] == BASELINE_FLOOR and agent.active()["disabled"] == ["admob"]
    db.close()


def test_d_outage_on_the_baseline_config_quarantines_without_touching_the_floor(tmp_path):
    db, sim, agent, _ = _ad_agent(tmp_path, "dark_baseline")
    for h in range(16):
        if h == 12:
            sim.set_dark("admob")
        r = agent.step(h, 1500)
        if h == 12:
            assert len(r["actions"]) == 1 and r["actions"][0]["disabled"] == ["admob"]      # quarantine only
    assert agent.active()["reserve_cpm"] == BASELINE_FLOOR
    assert not [e for e in agent.events if e["type"] == "rollback"]                         # nothing to roll back
    db.close()


def test_d_recovered_network_is_restored_by_the_half_open_probe(tmp_path):
    db, sim, agent, _ = _ad_agent(tmp_path, "dark_recover")
    for h in range(26):
        if h == 10:
            sim.set_dark("admob")
        if h == 13:
            sim.set_dark("admob", False)
        agent.step(h, 1500)
    assert [e["type"] for e in agent.events if e["type"] in ("quarantine", "probe")] == ["quarantine", "probe"]
    assert agent.active()["disabled"] == [] and agent.active()["reserve_cpm"] == BASELINE_FLOOR
    db.close()


def test_d_control_without_the_guard_the_trial_floor_and_the_dead_network_stay(tmp_path):
    out = {}
    for mode in ("guarded", "unguarded"):
        db, sim, agent, _ = _ad_agent(tmp_path, f"ctl_{mode}", guard=mode == "guarded")
        rev = 0.0
        for h in range(24):
            if h == 10:
                agent.apply_floor(6.0, "trial")
            if h == 12:
                sim.set_dark("admob")
            r = agent.step(h, 1500)
            rev += r["revenue_usd"] if h >= 12 else 0.0
        out[mode] = (agent.active(), rev)
        db.close()
    assert out["unguarded"][0] == {"version": 2, "reserve_cpm": 6.0, "disabled": [], "status": "trial"}
    assert out["guarded"][0]["reserve_cpm"] == BASELINE_FLOOR and out["guarded"][0]["disabled"] == ["admob"]
    assert out["guarded"][1] > 1.25 * out["unguarded"][1]              # post-outage revenue the mitigation saved


def test_d_packaged_drill_reports_the_rollback(db):
    logged: list[dict] = []
    r = fd.drill_network_dark(db, Tracer(InMemoryExporter()), logged.append, hours=16, requests=800)
    assert r["blocked"] and r["rolled_back_to"] == r["baseline_floor"] == BASELINE_FLOOR and r["trial_floor"] == 6.0
    assert any(s["status"] == "guard" and "ROLLBACK to the baseline floor" in s["title"] for s in logged)
    assert r["revenue_after_fault"]["guarded"] > r["revenue_after_fault"]["unguarded"]
