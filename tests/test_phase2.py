"""
================================================================================
BLOCK: Phase 2 Test Suite (LiveOps Copilot + Level Factory)
--------------------------------------------------------------------------------
Role & contracts
  Exercises every Phase-2 block against its OWN temp-seeded DuckDB, so the
  real data/telemetry.duckdb is never touched and Phase-1 tests are
  unaffected (LIVEOPS_DATA_DIR is set for this module and restored after).

Key engineering mechanism
  Textbook reference values for the statistics; end-to-end runs of the
  copilot playbook and the factory through the real harness; the real FastAPI
  app for flag bucketing and the daily rotation.

Failure modes & 1-step mitigation
  * Stochastic assertions -> fixed seeds, tolerances wide enough to be stable.
================================================================================
"""
from __future__ import annotations

import json
import math
import os
import random

import pytest

from agents.common import inject_sim_attempts
from agents.level_factory.critic_and_eval import (
    GOLDEN_PATH, CriticAgent, LevelFactory, cohens_kappa, evaluate_critic, load_golden,
)
from agents.level_factory.difficulty_model import DifficultyModel, LevelFeatures, nelder_mead
from agents.level_factory.solver import LevelSolver
from agents.liveops_copilot.agent import LiveOpsCopilot, simulated_operator
from agents.liveops_copilot.simulator import Effect, ExperimentSimulator
from agents.liveops_copilot.stats_engine import (
    MSPRT, always_valid_ci, cuped, guardrail_breached, peeking_false_positive_simulation, sample_size_means,
    sample_size_proportions,
)
from core.db import TelemetryDB, open_rw
from core.harness.runner import ToolCallBlocked
from core.harness.telemetry import InMemoryExporter, Tracer
from core.wordlist import load_lexicon
from scripts.seed_telemetry import seed_database

LEVEL1 = "HEHS/ATNS/AHOA/SHCW"


# ============================================================================ fixtures
@pytest.fixture(scope="module")
def lex():
    return load_lexicon()


@pytest.fixture(scope="module")
def solver(lex):
    return LevelSolver(lex)


@pytest.fixture(scope="module")
def p2_dir(tmp_path_factory, lex):
    d = tmp_path_factory.mktemp("phase2_data")
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
def db(p2_dir):
    d = TelemetryDB(p2_dir / "telemetry.duckdb")
    yield d
    d.close()


@pytest.fixture(scope="module")
def tracer():
    return Tracer(InMemoryExporter())


@pytest.fixture(scope="module")
def factory(db, lex, tracer):
    return LevelFactory(db, lex=lex, tracer=tracer, seed=7)


# ============================================================================ stats engine
def test_power_matches_textbook():
    p = sample_size_proportions(0.10, 0.02)
    assert (p.n_treatment, p.n_control) == (3841, 3841)            # classic 10% -> 12%
    m = sample_size_means(1.0, 0.1)
    assert (m.n_treatment, m.n_control) == (1570, 1570)            # 2 * (1.96 + 0.8416)^2 / 0.01
    c = sample_size_means(1.0, 0.1, cuped_rho=0.6)
    assert c.n_total == pytest.approx(m.n_total * (1 - 0.36), rel=0.01)
    u = sample_size_means(1.0, 0.1, treatment_share=0.2)
    assert u.n_total > m.n_total and u.n_treatment == pytest.approx(0.2 * u.n_total, abs=1)
    with pytest.raises(ValueError):
        sample_size_proportions(0.99, 0.05)


def test_cuped_reduction_tracks_rho_squared():
    rng = random.Random(4)
    def arm(n, lift):
        x = [rng.gauss(10, 3) for _ in range(n)]
        return [0.8 * xi + rng.gauss(0, 2) + lift for xi in x], x
    yt, xt = arm(4000, 0.2)
    yc, xc = arm(4000, 0.0)
    r = cuped(yt, xt, yc, xc)
    assert r.var_cuped < r.var_raw
    assert r.variance_reduction_pct == pytest.approx(100 * r.rho ** 2, abs=4)
    assert r.p_cuped < r.p_raw and r.ci_cuped[0] < 0.2 < r.ci_cuped[1]
    with pytest.raises(ValueError):
        cuped([1, 2], [5, 5], [1, 2], [5, 5])                       # zero-variance covariate


def test_msprt_always_valid_and_guardrail():
    lo, hi = always_valid_ci(0.5, 0.01, 0.04, 0.05)
    assert lo < 0.5 < hi and hi - lo > 2 * 1.96 * 0.1             # wider than a fixed-horizon CI
    t = MSPRT(alpha=0.05, tau=0.2, min_samples=1)
    steps = [t.update(-0.3, v, 1000, 1000) for v in (0.04, 0.01, 0.0025)]
    assert [s.p_always_valid for s in steps] == sorted((s.p_always_valid for s in steps), reverse=True)
    assert all(a.ci_high >= b.ci_high for a, b in zip(steps, steps[1:]))   # running intersection
    assert guardrail_breached(steps[-1]) and not guardrail_breached(steps[0])
    pk = peeking_false_positive_simulation(sims=200, peeks=10, n_per_peek=100, seed=1)
    assert pk.msprt_false_positive_rate <= 0.07 and pk.naive_false_positive_rate >= 0.12


# ============================================================================ solver + difficulty
def test_solver_proof(solver):
    p = solver.prove(LEVEL1, targets=["hat", "note"], target_count=6)
    assert p.ok and p.witness_rechecked and p.word_count == 34 and set(p.witness) == {"hat", "note"}
    assert not solver.prove(LEVEL1, targets=["zebra"]).ok
    assert solver.prove(LEVEL1, target_count=99).reason.startswith("unsolvable")
    assert solver.prove("ABC/DE").reason.startswith("malformed")
    assert 0 < p.branching_factor < 2 and 0 <= p.rare_share <= 1


def test_difficulty_model_monotone_and_targetable(solver, lex):
    p = solver.prove(LEVEL1)
    m = DifficultyModel()
    feats = lambda t, tl=100: LevelFeatures.from_words(p.valid_words, lex, t, tl, p.branching_factor)
    preds = [m.predict(feats(t)) for t in range(3, 20)]
    assert preds == sorted(preds)                                   # harder with more required words
    assert m.predict(feats(10, 60)) > m.predict(feats(10, 120))     # harder with less time
    t, pred = m.best_target(feats(3), 0.3)
    assert abs(pred - 0.3) < 0.1
    x, fx = nelder_mead(lambda v: (v[0] - 3) ** 2 + (v[1] + 1) ** 2, [0, 0])
    assert abs(x[0] - 3) < 1e-3 and abs(x[1] + 1) < 1e-3


def test_difficulty_calibrated_on_telemetry(factory):
    cal = factory.model.calibration
    assert cal["calibrated"] and cal["mae_after"] < cal["mae_before"] and cal["mae_after"] < 0.12
    l7 = next(p for p in factory.calibration_points if p["level_id"] == 7)
    assert l7["predicted"] == max(p["predicted"] for p in factory.calibration_points)


# ============================================================================ critic
def test_cohens_kappa_reference():
    y = ["a"] * 20 + ["b"] * 30
    p = ["a"] * 15 + ["b"] * 5 + ["a"] * 10 + ["b"] * 20
    assert cohens_kappa(y, p) == pytest.approx((0.7 - 0.5) / 0.5)
    assert cohens_kappa(y, y) == 1.0


def test_critic_on_golden(solver, factory):
    items = load_golden()
    assert len(items) == 100 and sum(i["label"] == "reject" for i in items) == 46
    seeds = [set(json.loads(r["valid_words"])) for r in factory.db.query("SELECT valid_words FROM levels WHERE created_by='seed'")]
    ev = evaluate_critic(CriticAgent(solver, factory.model, seeds), items)
    assert ev["recall_reject"] >= 0.95 and ev["cohens_kappa"] >= 0.75 and ev["agreement_pct"] >= 90


# ============================================================================ factory end to end
def test_factory_ship_monitor_and_release_gate(factory, db):
    shipped = factory.ship_daily("2026-11-09")
    assert len(shipped) == 3 and all(s["status"] == "live" for s in shipped)
    again = factory.ship_daily("2026-11-09")
    assert [s["level_id"] for s in again] == [s["level_id"] for s in shipped] and again[0]["already_shipped"]
    with db.cursor() as con:
        pred = {r[0]: r[1] for r in con.execute("SELECT level_id, predicted_fail_rate FROM levels WHERE created_by='level_factory'").fetchall()}
    assert all(pred[s["level_id"]] == s["predicted_fail_rate"] for s in shipped)

    with factory.runner.workflow("bypass", mode="READ_WRITE") as wf:      # publish without passing the gates
        with pytest.raises(ToolCallBlocked) as e:
            wf.call("publish_level", level_id=0, grid="ABCD/EFGH/IJKL/MNOP", target_words=3, time_limit_s=60, status="live")
    assert e.value.record.decision.rule == "approval_denied"

    victim = shipped[0]
    inject_sim_attempts(db, victim["level_id"], victim["version"], 50, 0.9, seed=9)
    acts = {a["level_id"]: a for a in factory.monitor_and_rollback(min_attempts=40, margin=0.2)}
    assert acts[victim["level_id"]]["status"] == "retired"
    assert not db.query("SELECT 1 FROM active_levels WHERE level_id = ?", [victim["level_id"]])


# ============================================================================ copilot end to end
@pytest.fixture(scope="module")
def copilot(db, factory, tracer):
    ref: list = []
    cop = LiveOpsCopilot(db, factory=factory, decide=simulated_operator({"l7_bad", "l7_good"}, ref),
                         decided_by="test_operator", tracer=tracer)
    ref.append(cop)
    return cop


def test_copilot_scan_and_power(copilot):
    scan = copilot.scan()
    assert scan["anomalies"] and scan["anomalies"][0]["level_id"] == 7
    plan = copilot.plan(7, daily_users=4000)
    assert plan["power"]["primary"]["n_total"] <= plan["power"]["primary"]["n_total_without_cuped"]
    assert plan["power"]["primary"]["treatment_share"] == 0.2


def test_copilot_cap_gate_rollback_and_ship(copilot, db):
    bad = copilot.launch("l7_bad", 7, goal_fail_rate=0.06)
    assert bad["blocked"] is None and bad["flag_result"]["status"] == "active"
    wide = copilot.request_flag("l7_wide", 7, bad["candidate"]["version"], 50)
    assert wide["blocked"]["layer"] == "L1_allowlist" and wide["blocked"]["rule"] == "rollout_cap"
    rogue = copilot.request_flag("l7_rogue", 7, bad["candidate"]["version"], 10)
    assert rogue["blocked"]["rule"] == "approval_denied"
    with db.cursor() as con, pytest.raises(Exception, match="(?i)constraint"):
        con.execute("INSERT INTO feature_flags VALUES ('x', 7, 1, 2, 50, 'active', now(), now(), '')")

    sim = ExperimentSimulator(db, seed=2)
    for day in range(1, 6):
        sim.simulate_day("l7_bad", day, 4000, 0.2, Effect(primary_lift=-0.03, d1_delta=-0.08))
        res = copilot.daily_monitor("l7_bad", min_days=7)
        if res["action"]:
            break
    assert res["action"]["status"] == "rolled_back"
    assert db.query("SELECT status FROM levels WHERE level_id=7 AND version=?", [bad["candidate"]["version"]])[0]["status"] == "rolled_back"

    good = copilot.launch("l7_good", 7, goal_fail_rate=0.35)
    for day in range(1, 15):
        sim.simulate_day("l7_good", day, 4000, 0.2, Effect(primary_lift=0.15, d1_delta=0.0))
        res = copilot.daily_monitor("l7_good", min_days=7)
        if res["action"]:
            break
    assert res["action"]["status"] == "shipped" and res["evaluation"]["days"] >= 7
    assert db.query("SELECT version FROM active_levels WHERE level_id = 7")[0]["version"] == good["candidate"]["version"]
    decisions = {(r["tool"], r["decision"]) for r in db.query("SELECT tool, decision FROM approvals WHERE agent='liveops_copilot'")}
    assert {("configure_flag", "approved"), ("configure_flag", "denied"), ("ship_experiment", "approved")} <= decisions


def test_game_serves_flagged_treatment(copilot, db, factory, p2_dir):
    from fastapi.testclient import TestClient

    from app.game_api import flag_bucket
    from app.main import app

    cand = factory.propose_variant(3, 0.2, note="flag test")
    with db.cursor() as con:
        con.execute("INSERT OR REPLACE INTO feature_flags VALUES ('l3_test', 3, 1, ?, 20, 'active', now(), now(), 'test')",
                    [cand["version"]])
    treated = next(u for u in (f"u{i}" for i in range(500)) if flag_bucket("l3_test", u) < 20)
    control = next(u for u in (f"u{i}" for i in range(500)) if flag_bucket("l3_test", u) >= 20)
    with TestClient(app) as cli:
        for uid, want in ((treated, cand["version"]), (control, 1)):
            s = cli.post("/api/game/session", json={"user_id": uid}).json()
            r = cli.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": 3}).json()
            assert r["level_version"] == want and r["experiment"] == "l3_test"
        levels = {l["level_id"]: l for l in cli.get("/api/game/levels").json()["levels"]}
        assert levels[3]["experiment"] == "l3_test" and any(l["daily"] for l in levels.values())
    exp = db.query("SELECT user_id, variant, source FROM experiment_exposures WHERE flag='l3_test' ORDER BY user_id")
    assert {(e["user_id"], e["variant"]) for e in exp} == {(treated, "treatment"), (control, "control")}
