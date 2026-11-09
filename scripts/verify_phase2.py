"""
================================================================================
BLOCK: Phase 2 Verification & Scorecard (Projects 1 + 2, end to end)
--------------------------------------------------------------------------------
Role & contracts
  One command runs both agent pipelines against data/telemetry.duckdb through
  the core.harness runtime and prints Markdown scorecards (also written to
  data/reports/phase2_scorecard.md). It then boots the real FastAPI app
  in-process to prove the game serves what the agents shipped.

  Project 1 (LiveOps Copilot): funnel scan -> power analysis -> exposure-cap
    and approval-gate checks -> harmful variant with automatic rollback ->
    good variant with CUPED + mSPRT and an approved ship -> A/A peeking
    simulation.
  Project 2 (Level Factory): difficulty-model calibration -> 100-level golden
    eval of the critic -> daily ship into the live rotation -> simulated live
    fail spike -> automatic retire.

Key engineering mechanism
  --reset (default) deletes ONLY Phase-2 state (factory levels, flags,
  approvals, exposures, audit, source='sim' events), so every run is
  reproducible. Seeded data and live player events are never touched.

Failure modes & 1-step mitigation
  * Game server holds the DuckDB lock -> exit 2 with the one-line fix.
  * Any pipeline invariant fails (cap not enforced, no rollback, critic
    kappa < 0.6, nothing shipped ...) -> printed as FAIL and exit code 1.
================================================================================
Usage: python scripts/verify_phase2.py [--date 2026-11-09] [--daily-users 4000] [--no-reset] [--no-trace]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date as date_cls
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.level_factory.critic_and_eval import (  # noqa: E402
    GOLDEN_PATH, CriticAgent, LevelFactory, build_golden_dataset, evaluate_critic, load_golden, save_golden,
)
from agents.common import inject_sim_attempts  # noqa: E402
from agents.liveops_copilot.agent import LiveOpsCopilot, simulated_operator  # noqa: E402
from agents.liveops_copilot.simulator import Effect, ExperimentSimulator  # noqa: E402
from agents.liveops_copilot.stats_engine import peeking_false_positive_simulation  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.db import DatabaseLockedError, TelemetryDB  # noqa: E402
from core.harness.telemetry import DuckDBSpanExporter, InMemoryExporter, Tracer  # noqa: E402
from core.levels import build_seed_levels  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def table(headers: list[str], rows: list[list]) -> str:
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


class Tee:
    def __init__(self, *exporters):
        self.exporters = exporters

    def export(self, span) -> None:
        for e in self.exporters:
            e.export(span)


def reset_phase2(db: TelemetryDB) -> None:
    with db.cursor() as con:
        con.execute("DELETE FROM levels WHERE created_by = 'level_factory'")
        for t in ("feature_flags", "approvals", "experiment_exposures", "agent_audit"):
            con.execute(f"DELETE FROM {t}")
        con.execute("DELETE FROM events WHERE source = 'sim'")


def run_experiment(cop: LiveOpsCopilot, sim: ExperimentSimulator, flag: str, effect: Effect, users: int, share: float,
                   max_days: int) -> dict:
    timeline = []
    for day in range(1, max_days + 1):
        sim.simulate_day(flag, day, users, share, effect)
        res = cop.daily_monitor(flag, mde_rel=0.10, min_days=7, max_days=max_days)
        timeline.append(res)
        if res["action"] is not None:
            break
    return {"timeline": timeline, "final": timeline[-1]}


def project1(db: TelemetryDB, factory: LevelFactory, tracer: Tracer, daily_users: int, days: int) -> tuple[list[str], dict]:
    md: list[str] = ["## Project 1 - LiveOps Experimentation Copilot", ""]
    ref: list = []
    cop = LiveOpsCopilot(db, factory=factory, decide=simulated_operator({"level7_trivial", "level7_easier"}, ref),
                         decided_by="simulated_operator", tracer=tracer)
    ref.append(cop)

    # 1. funnel scan
    scan = cop.scan()
    md += ["### 1. Funnel scan (active versions, robust z vs other levels)", "",
           table(["level", "version", "attempts", "fail rate", "robust z", "anomaly"],
                 [[l["level_id"], l["version"], l["attempts"], l["fail_rate"], l["robust_z"],
                   "**YES**" if any(a["level_id"] == l["level_id"] for a in scan["anomalies"]) else ""] for l in scan["levels"]]), ""]
    anomaly = scan["anomalies"][0] if scan["anomalies"] else {}
    md += [f"> {anomaly.get('message', 'no anomaly found')}", ""]
    check("P1 scan flags level 7", anomaly.get("level_id") == 7, anomaly.get("message", ""))

    # 2. power analysis
    plan = cop.plan(7, daily_users=daily_users, mde_rel=0.10, rollout_pct=20)
    b, pw, g = plan["baseline"], plan["power"]["primary"], plan["power"]["guardrail_d1"]
    md += ["### 2. Baselines (measured from telemetry) and power analysis at 20% exposure", "",
           table(["quantity", "value"], [
               ["primary metric", b["primary_metric"]], ["baseline mean / sigma", f"{b['mean']} / {b['sigma']}"],
               ["CUPED covariate rho (pre vs post)", b["rho"]], ["D1 retention baseline", b["d1_baseline"]],
               ["MDE (relative) / alpha / power", "10% / 0.05 / 0.80"], ["treatment share (cap)", "20%"],
               ["users needed WITHOUT CUPED", pw["n_total_without_cuped"]], ["users needed WITH CUPED", pw["n_total"]],
               ["eligible users per day (assumed production)", daily_users], ["planned duration (days)", pw["days"]],
               ["D1 guardrail: users to detect -2pp", g["n_total"]]]), ""]
    check("P1 CUPED reduces required sample", pw["n_total"] < pw["n_total_without_cuped"],
          f"{pw['n_total_without_cuped']} -> {pw['n_total']}")

    # 3. scenario B: harmful variant, automatic rollback
    sim = ExperimentSimulator(db, seed=11)
    launch_b = cop.launch("level7_trivial", 7, goal_fail_rate=0.06, rollout_pct=20)
    wide = cop.request_flag("level7_trivial_wide", 7, launch_b["candidate"]["version"], 50)
    unplanned = cop.request_flag("level7_unplanned", 7, launch_b["candidate"]["version"], 15)
    with db.cursor() as con:
        try:
            con.execute("INSERT INTO feature_flags VALUES ('raw_sql_50', 7, 1, 2, 50, 'active', now(), now(), 'bypass')")
            db_check = "accepted (BUG)"
        except Exception as e:
            db_check = f"rejected: {type(e).__name__}"
    md += ["### 3. Exposure cap and Human Approval Gate", "",
           table(["request", "rollout", "outcome", "enforced by"], [
               ["level7_trivial (planned)", "20%", "approved" if launch_b["flag_result"] else "blocked",
                "approval ledger: simulated_operator"],
               ["level7_trivial_wide", "50%", "BLOCKED" if wide["blocked"] else "allowed (BUG)",
                f"{wide['blocked']['layer']}:{wide['blocked']['rule']}" if wide["blocked"] else ""],
               ["level7_unplanned", "15%", "DENIED" if unplanned["blocked"] else "allowed (BUG)",
                f"{unplanned['blocked']['layer']}:{unplanned['blocked']['rule']}" if unplanned["blocked"] else ""],
               ["raw SQL bypass", "50%", db_check, "DuckDB CHECK (rollout_pct <= 20)"]]), ""]
    check("P1 50% exposure blocked by L1", bool(wide["blocked"]) and wide["blocked"]["layer"] == "L1_allowlist")
    check("P1 unplanned flag denied at approval gate", bool(unplanned["blocked"]) and unplanned["blocked"]["rule"] == "approval_denied")
    check("P1 DB CHECK rejects 50%", db_check.startswith("rejected"))

    exp_b = run_experiment(cop, sim, "level7_trivial", Effect(primary_lift=-0.03, d1_delta=-0.06), daily_users, 0.20, days)
    ev_b = exp_b["final"]["evaluation"]
    md += ["### 4. Scenario B - too-easy variant hurts D1 retention -> automatic rollback", "",
           f"Candidate level 7 v{launch_b['candidate']['version']} (`{launch_b['candidate']['grid']}`, target "
           f"{launch_b['candidate']['target_words']}/{launch_b['candidate']['word_count']}). Simulated truth: D1 -6pp, "
           "primary -3%.", "",
           table(["day", "n treat", "n ctrl", "D1 delta", "D1 always-valid CI", "guardrail breached"],
                 [[p["day"], p["n_t"], p["n_c"], p["d1_delta"], f"[{p['d1_ci'][0]:+.4f}, {p['d1_ci'][1]:+.4f}]",
                   "**YES**" if p["guardrail_breached"] else "no"] for p in ev_b["peeks"]]), "",
           f"**Action:** `{(exp_b['final']['action'] or {}).get('status')}` on day {ev_b['days']} - {ev_b['reason']}", ""]
    check("P1 harmful variant auto-rolled back", (exp_b["final"]["action"] or {}).get("status") == "rolled_back",
          f"day {ev_b['days']}")

    # 5. scenario A: good variant, CUPED + mSPRT, approved ship
    launch_a = cop.launch("level7_easier", 7, goal_fail_rate=0.35, rollout_pct=20)
    exp_a = run_experiment(cop, sim, "level7_easier", Effect(primary_lift=0.12, d1_delta=0.01), daily_users, 0.20, days)
    ev_a = exp_a["final"]["evaluation"]
    cu = ev_a["cuped"]
    md += ["### 5. Scenario A - easier level 7 (goal fail 0.35) -> CUPED + mSPRT -> approved ship", "",
           f"Candidate level 7 v{launch_a['candidate']['version']} (`{launch_a['candidate']['grid']}`, target "
           f"{launch_a['candidate']['target_words']}/{launch_a['candidate']['word_count']}, predicted fail "
           f"{launch_a['candidate']['predicted_fail_rate']}). Simulated truth: primary +12%, D1 +1pp.", "",
           table(["day", "n treat", "n ctrl", "CUPED delta", "always-valid 95% CI", "likelihood ratio", "always-valid p", "D1 delta"],
                 [[p["day"], p["n_t"], p["n_c"], p["delta_cuped"], f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]", p["lr"],
                   p["p_av"], p["d1_delta"]] for p in ev_a["peeks"]]), "",
           "**CUPED at the final peek**", "",
           table(["", "raw", "CUPED"], [
               ["effect (completions / user / 7d)", cu["effect_raw"], cu["effect_cuped"]],
               ["Var(effect estimator)", cu["var_raw"], cu["var_cuped"]],
               ["p-value (fixed horizon)", cu["p_raw"], cu["p_cuped"]],
               ["95% CI", f"[{cu['ci_raw'][0]:+.4f}, {cu['ci_raw'][1]:+.4f}]", f"[{cu['ci_cuped'][0]:+.4f}, {cu['ci_cuped'][1]:+.4f}]"]]), "",
           f"theta = {cu['theta']}, rho = {cu['rho']}, **variance reduction {cu['variance_reduction_pct']}%**, "
           f"relative lift {cu['relative_lift_cuped']:+.2%}.", "",
           f"**Action:** `{(exp_a['final']['action'] or {}).get('status')}` on day {ev_a['days']} - {ev_a['reason']}", ""]
    check("P1 good variant shipped after approval", (exp_a["final"]["action"] or {}).get("status") == "shipped")
    check("P1 CUPED variance reduction > 0", cu["variance_reduction_pct"] > 0, f"{cu['variance_reduction_pct']}%")

    # 6. peeking
    t0 = time.perf_counter()
    pk = peeking_false_positive_simulation(sims=400, peeks=14, n_per_peek=200)
    md += ["### 6. Peeking safety (A/A simulation, 400 experiments x 14 daily peeks)", "",
           table(["procedure", "false-positive rate", "nominal alpha"], [
               ["fixed-horizon z-test, single look at the end", pk.naive_single_look_rate, pk.alpha],
               ["fixed-horizon z-test, stop at first p<0.05 (peeking)", pk.naive_false_positive_rate, pk.alpha],
               ["mSPRT always-valid, stop at first p<=0.05", pk.msprt_false_positive_rate, pk.alpha]]),
           f"\n_{time.perf_counter() - t0:.1f}s_", ""]
    check("P1 mSPRT FPR <= alpha under peeking", pk.msprt_false_positive_rate <= pk.alpha, f"{pk.msprt_false_positive_rate}")
    check("P1 naive peeking inflates FPR", pk.naive_false_positive_rate > 2 * pk.alpha, f"{pk.naive_false_positive_rate}")

    with db.cursor() as con:
        appr = con.execute("SELECT tool, args, decision, decided_by, reason FROM approvals WHERE agent='liveops_copilot' ORDER BY ts").fetchall()
    md += ["### 7. Approval ledger (`approvals` table)", "",
           table(["tool", "args", "decision", "by", "reason"],
                 [[r[0], r[1][:70], r[2], r[3], r[4][:90]] for r in appr]), ""]
    last_a, last_b = ev_a["peeks"][-1], ev_b["peeks"][-1]
    metrics = {"anomaly": anomaly.get("message"), "cuped_variance_reduction_pct": cu["variance_reduction_pct"], "cuped_rho": cu["rho"],
               "relative_lift": cu["relative_lift_cuped"], "good_d1_delta": last_a["d1_delta"], "good_d1_ci": last_a["d1_ci"],
               "good_ship_day": ev_a["days"], "harmful_d1_delta": last_b["d1_delta"], "harmful_d1_ci": last_b["d1_ci"],
               "harmful_rollback_day": ev_b["days"], "n_without_cuped": pw["n_total_without_cuped"], "n_with_cuped": pw["n_total"],
               "naive_peeking_fpr": pk.naive_false_positive_rate, "msprt_fpr": pk.msprt_false_positive_rate}
    return md, {"scan": scan, "launch_a": launch_a, "launch_b": launch_b, "anomaly": anomaly, "metrics": metrics}


def project2(db: TelemetryDB, factory: LevelFactory, run_date: str) -> tuple[list[str], dict]:
    md: list[str] = ["## Project 2 - Autonomous Level Factory", ""]
    cal = factory.model.calibration
    md += ["### 1. Difficulty model calibration (observed vs predicted per-attempt fail rate)", "",
           table(["level", "version", "attempts", "observed", "predicted", "abs error"],
                 [[p["level_id"], p["version"], p["attempts"], round(p["fail_rate"], 4), p.get("predicted"),
                   round(abs(p["fail_rate"] - p.get("predicted", 0)), 4)] for p in factory.calibration_points]), "",
           f"Attempt-weighted MAE {cal.get('mae_before')} (priors) -> **{cal.get('mae_after')}** (calibrated). "
           f"Params: `{json.dumps(cal.get('params'))}`. Caveat: the seeded fail rates were set per level in Phase 1, "
           "not produced by board features, so this measures fit to that data, not real-player validity.", ""]
    check("P2 difficulty model calibrated MAE < 0.10", cal.get("mae_after", 1) < 0.10, str(cal.get("mae_after")))

    # golden eval
    if GOLDEN_PATH.exists():
        items = load_golden()
    else:
        items = build_golden_dataset(factory.solver, factory.model, seed_levels=build_seed_levels(factory.lex, factory.solver.trie))
        save_golden(items)
    with db.cursor() as con:
        seed_sets = [set(json.loads(r[0])) for r in con.execute("SELECT valid_words FROM levels WHERE created_by='seed'").fetchall()]
    ev = evaluate_critic(CriticAgent(factory.solver, factory.model, seed_sets), items)
    c = ev["confusion"]
    md += ["### 2. Critic vs 100-level golden dataset", "",
           f"Labels: **{', '.join(ev['label_source'])}** (`data/golden_levels.jsonl`; add `data/golden_levels_human.csv` "
           "with `id,label` to evaluate against human labels instead).", "",
           table(["metric", "value"], [
               ["agreement", f"{ev['agreement_pct']}%"], ["Cohen's kappa", ev["cohens_kappa"]],
               ["precision (reject = defect)", ev["precision_reject"]], ["recall (reject = defect)", ev["recall_reject"]],
               ["F1 (reject)", ev["f1_reject"]], ["precision (accept)", ev["precision_accept"]],
               ["recall (accept)", ev["recall_accept"]]]), "",
           table(["", "critic: reject", "critic: accept"], [
               ["label: reject (defect)", c["tp_reject"], c["fn_missed_defect"]],
               ["label: accept (clean)", c["fp_wrongly_rejected"], c["tn_accept"]]]), "",
           table(["defect type", "n", "critic agrees"],
                 [[k, v["n"], f"{v['agree']}/{v['n']}"] for k, v in sorted(ev["per_defect"].items())]), "",
           "Disagreements (critic stricter than the construction label):", ""]
    md += [f"- {d['id']} {d['defect']}: {'; '.join(d['failed']) or 'critic accepted'}" for d in ev["disagreements"]] + [""]
    check("P2 critic kappa >= 0.6", ev["cohens_kappa"] >= 0.6, str(ev["cohens_kappa"]))
    check("P2 critic catches >= 95% of injected defects", ev["recall_reject"] >= 0.95, str(ev["recall_reject"]))

    # daily ship
    t0 = time.perf_counter()
    shipped = factory.ship_daily(run_date)
    f = factory.funnel
    md += [f"### 3. Daily ship ({run_date})", "",
           table(["stage", "count"], [[k.replace("_", " "), v] for k, v in f.items()]), "",
           table(["level id", "slot", "grid", "target / words", "time", "goal", "predicted fail", "critic scores", "source"],
                 [[s["level_id"], s.get("slot"), f"`{s['grid']}`", f"{s['target_words']}/{s.get('word_count')}",
                   s.get("time_limit_s"), s.get("goal"), s["predicted_fail_rate"], s.get("scores"), s.get("source")] for s in shipped]),
           f"\n_{time.perf_counter() - t0:.1f}s, every publish approved by the release gate (solver proof + critic in ledger)_", ""]
    check("P2 daily levels shipped", len(shipped) == 3, f"{len(shipped)} levels")
    with db.cursor() as con:
        gate = con.execute("SELECT decision, count(*) FROM approvals WHERE agent='level_factory' GROUP BY 1").fetchall()
    md += [f"Release-gate decisions: {dict(gate)}", ""]
    return md, {"shipped": shipped, "golden": ev, "metrics": {
        "agreement_pct": ev["agreement_pct"], "kappa": ev["cohens_kappa"], "recall_reject": ev["recall_reject"],
        "precision_reject": ev["precision_reject"], "calibration_mae": cal.get("mae_after"), "funnel": dict(f),
        "calibration_points": [{k: p.get(k) for k in ("level_id", "attempts", "fail_rate", "predicted", "target", "time_limit_s")} |
                               {"words": len(p["words"])} for p in factory.calibration_points]}}


def game_check(db: TelemetryDB, factory: LevelFactory, shipped: list[dict]) -> list[str]:
    """Boot the real app in-process and play against what the agents shipped."""
    from fastapi.testclient import TestClient

    from app.main import app

    md = ["## Live game wiring (real FastAPI app, in-process)", ""]
    with TestClient(app) as cli:
        levels = cli.get("/api/game/levels").json()["levels"]
        by_id = {l["level_id"]: l for l in levels}
        md += [table(["level", "version", "created by", "daily", "target", "time", "fail rate (this version)", "predicted"],
                     [[l["level_id"], l["version"], l["created_by"], l["daily"], l["target_words"], l["time_limit_s"],
                       l["fail_rate"], l["predicted_fail_rate"]] for l in levels]), ""]
        check("GAME level 7 now serves the shipped version", by_id[7]["created_by"] == "level_factory", f"v{by_id[7]['version']}")
        check("GAME daily levels in rotation", all(s["level_id"] in by_id for s in shipped))
        s = cli.post("/api/game/session", json={}).json()
        start = cli.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": shipped[0]["level_id"]}).json()
        check("GAME daily level playable", start["level_id"] == shipped[0]["level_id"] and start["grid"],
              "/".join(start.get("grid", [])))

        # spike on the hard daily level -> monitor retires it
        hard = shipped[-1]
        inject_sim_attempts(db, hard["level_id"], hard["version"], 60, 0.85, seed=5)
        actions = factory.monitor_and_rollback(min_attempts=40, margin=0.20)
        act = next(a for a in actions if a["level_id"] == hard["level_id"])
        after = {l["level_id"] for l in cli.get("/api/game/levels").json()["levels"]}
        md += ["### Auto-rollback on live fail-rate spike", "",
               table(["level", "version", "predicted", "observed", "attempts", "spike", "action"],
                     [[a["level_id"], a["version"], a["predicted"], a["observed"], a["attempts"], a["spike"],
                       a.get("status", a["action"])] for a in actions]), "",
               f"Level {hard['level_id']} in rotation after monitor: **{hard['level_id'] in after}**", ""]
        check("P2 spiking daily level retired", act.get("status") == "retired" and hard["level_id"] not in after)
    return md


def harness_summary(mem: InMemoryExporter) -> list[str]:
    spans = mem.spans
    by_tool: dict[str, list] = {}
    for sp in spans:
        if sp.name == "tool.call":
            by_tool.setdefault(sp.attributes.get("tool", "?"), []).append(sp)
    rows = []
    for tool, ss in sorted(by_tool.items()):
        d = sorted(x.duration_ms for x in ss)
        p95 = d[min(len(d) - 1, int(0.95 * len(d)))]
        rows.append([tool, len(ss), sum(x.status != "ERROR" for x in ss), sum(bool(x.attributes.get("blocked")) for x in ss),
                     round(d[len(d) // 2], 2), round(p95, 2)])
    return ["## Harness telemetry for this run", "",
            f"{sum(s.name == 'workflow.run' for s in spans)} workflows, {len(by_tool) and sum(len(v) for v in by_tool.values())} tool calls, "
            "all exported to `data/traces.duckdb` (`make metrics`).", "",
            table(["tool", "calls", "ok", "blocked by guard", "p50 ms", "p95 ms"], rows), ""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0])
    ap.add_argument("--date", default=date_cls.today().isoformat())
    ap.add_argument("--daily-users", type=int, default=4000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    s = get_settings()
    try:
        db = TelemetryDB(s.telemetry_db)
    except DatabaseLockedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    if not a.no_reset:
        reset_phase2(db)
    mem = InMemoryExporter()
    tracer = Tracer(Tee(mem) if a.no_trace else Tee(mem, DuckDBSpanExporter(s.traces_db)), service_name="phase2-verify")
    factory = LevelFactory(db, tracer=tracer, seed=7)
    md1, p1 = project1(db, factory, tracer, a.daily_users, a.days)
    md2, p2 = project2(db, factory, a.date)
    md3 = game_check(db, factory, p2["shipped"])
    md4 = harness_summary(mem)
    head = ["# Phase 2 Scorecard", "", f"_Generated {a.date}, {time.perf_counter() - t0:.1f}s, LLM: deterministic mock "
            "(set ANTHROPIC_API_KEY or OPENAI_API_KEY for a real provider)._", "",
            table(["check", "result", "detail"], [[n, "PASS" if ok else "**FAIL**", d] for n, ok, d in CHECKS]), ""]
    report = "\n".join(head + md1 + md2 + md3 + md4)
    print(report)
    out = a.out or s.data_dir / "reports" / "phase2_scorecard.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    out.with_suffix(".json").write_text(json.dumps({"project1": p1["metrics"], "project2": p2["metrics"],
                                                    "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS]}, indent=2, default=str))
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed -> {out}")
    db.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
