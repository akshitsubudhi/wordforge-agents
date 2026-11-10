"""
================================================================================
BLOCK: Scorecard Builder (single source for README + Control Center tiles)
--------------------------------------------------------------------------------
Role & contracts
  Merges the machine-readable outputs of every evaluation into
  docs/scorecard.json (read by /api/cc/overview) and docs/SCORECARD.md:
    docs/phase2_scorecard.json             (make phase2)
    docs/phase3_ad_economy_scorecard.json  (make ad-economy)
    docs/phase3_intake_eval_scorecard.json   (make intake-eval)
    conformance suite                      (re-run here, ~4 s)
    data/traces.duckdb                     (p50/p95 per task, LLM cost per task)
    level-taste analysis                   (calibration points + critic
                                            verdicts on 150 fresh candidates)

Key engineering mechanism
  Every headline number carries its definition and its caveat in `caption`,
  so the README never shows a number without what it measures.

Failure modes & 1-step mitigation
  * A missing input -> that tile says "run make <target>" instead of a stale
    or invented number.
================================================================================
Usage: python scripts/build_scorecard.py   (after make phase2 phase3; server stopped)
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402

from core.config import REPO_ROOT, get_settings  # noqa: E402
from core.db import DatabaseLockedError  # noqa: E402
from core.harness.run_redteam_eval import run_suite  # noqa: E402
from core.harness.telemetry import InMemoryExporter, TracesBusyError, Tracer, traces_connection  # noqa: E402

DOCS = REPO_ROOT / "docs"


def load(name: str) -> dict | None:
    p = DOCS / name
    return json.loads(p.read_text()) if p.exists() else None


def trace_stats() -> dict:
    """Latency / cost of the scorecard's OWN workloads. Fault-drill spans are excluded: drills deliberately loop,
    fail and burn budget, and whether `make faults` ran before this script must not change the published row."""
    p = get_settings().traces_db
    if not p.exists():
        return {}
    keep = "service_name <> 'fault-drills'"
    try:
        with traces_connection(p) as con:
            con.execute("SELECT 1 FROM spans LIMIT 1")
            roots = con.execute(f"""SELECT name, count(*) n, quantile_cont(duration_ms, 0.5) p50, quantile_cont(duration_ms, 0.95) p95,
                                          avg((status <> 'ERROR')::INT) ok
                                   FROM spans WHERE parent_span_id IS NULL AND {keep} GROUP BY 1 ORDER BY n DESC""").fetchall()
            tools = con.execute(f"""SELECT count(*), quantile_cont(duration_ms, 0.95) FROM spans WHERE name = 'tool.call' AND {keep}""").fetchone()
            cost = con.execute(f"""WITH llm AS (SELECT trace_id, sum(cost_usd) c FROM spans WHERE name = 'llm.call' AND {keep} GROUP BY 1)
                                  SELECT count(*), avg(c), max(c) FROM llm""").fetchone()
            allroot = con.execute(f"SELECT count(*), quantile_cont(duration_ms, 0.95) FROM spans WHERE parent_span_id IS NULL AND {keep}").fetchone()
    except TracesBusyError as e:
        print(f"WARNING: {e}", file=sys.stderr)
        return {}
    except duckdb.Error:
        return {}
    return {"roots": [{"name": r[0], "n": r[1], "p50_ms": round(r[2], 2), "p95_ms": round(r[3], 2), "success": round(r[4], 4)} for r in roots],
            "tool_calls": tools[0], "tool_p95_ms": round(tools[1] or 0, 2), "tasks": allroot[0], "task_p95_ms": round(allroot[1] or 0, 2),
            "llm_tasks": cost[0], "llm_cost_per_task": cost[1] or 0.0, "llm_cost_max": cost[2] or 0.0}


def level_taste(p2: dict | None) -> dict:
    """What makes a level hard, and what the critic rejects."""
    out: dict = {}
    pts = (p2 or {}).get("project2", {}).get("calibration_points", [])
    if pts:
        rows = [{"level": p["level_id"], "fail": p["fail_rate"], "pred": p["predicted"], "ratio": p["target"] / p["words"],
                 "sec_per_word": p["time_limit_s"] / p["target"], "words": p["words"]} for p in pts]
        fail = [r["fail"] for r in rows]
        corr = lambda k: round(statistics.correlation([r[k] for r in rows], fail), 3)
        out["levels"] = rows
        out["correlation_with_fail_rate"] = {"target_ratio": corr("ratio"), "seconds_per_required_word": corr("sec_per_word"),
                                             "words_on_board": corr("words")}
    try:
        from agents.level_factory.critic_and_eval import DAILY_SLOTS, CriticAgent, LevelGenerator
        from agents.level_factory.difficulty_model import calibrated_model
        from agents.level_factory.solver import LevelSolver
        from core.wordlist import load_lexicon

        con = duckdb.connect(str(get_settings().telemetry_db), read_only=True)
        try:
            lex = load_lexicon()
            model, _ = calibrated_model(con, lex)
            seeds = [set(json.loads(r[0])) for r in con.execute("SELECT valid_words FROM levels WHERE created_by='seed'").fetchall()]
        finally:
            con.close()
        solver = LevelSolver(lex)
        gen, critic = LevelGenerator(solver, model, seed=99), CriticAgent(solver, model, seeds)
        reasons, approved, n, scores = Counter(), 0, 0, Counter()
        for _, goal, size, tl in DAILY_SLOTS:
            for c in gen.propose(goal, size, tl, 50):
                rep = critic.review(c)
                n += 1
                approved += rep.approve
                for s in rep.scores:
                    scores[s.rubric] += s.score
                    if s.score < 3:
                        reasons[s.rubric + ": " + s.reason.split(":")[0].split("(")[0].strip()[:48]] += 1
        out["critic_on_fresh_candidates"] = {"n": n, "approved": approved, "approval_rate": round(approved / n, 3),
                                             "top_rejection_reasons": reasons.most_common(6),
                                             "mean_rubric_score": {k: round(v / n, 2) for k, v in scores.items()}}
    except Exception as e:  # DB locked by the server, etc.
        out["critic_on_fresh_candidates"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return out


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def main() -> int:
    tdb = get_settings().telemetry_db
    if tdb.exists():  # refuse to write a degraded scorecard while the server owns the single-writer lock
        try:
            duckdb.connect(str(tdb), read_only=True).close()
        except duckdb.IOException as e:
            print(f"ERROR: {DatabaseLockedError(tdb, str(e))}", file=sys.stderr)
            return 2
    p2, ad, intake = load("phase2_scorecard.json"), load("phase3_ad_economy_scorecard.json"), load("phase3_intake_eval_scorecard.json")
    conf = run_suite(Tracer(InMemoryExporter()))
    tr = trace_stats()
    taste = level_taste(p2)
    s_all, n = conf["summary"], conf["counts"]
    total = n["A"] + n["B"] + n["C"]
    best_single = max(s_all[k]["blocked"] for k in ("L1_only", "L2_only", "L3_only"))
    pm1, pm2 = (p2 or {}).get("project1", {}), (p2 or {}).get("project2", {})
    need = lambda t: {"value": "-", "caption": f"run `make {t}`"}
    head = [
        {"key": "eval_agreement", "label": "Critic ↔ golden agreement",
         **({"value": f"{pm2['agreement_pct']}%", "caption": f"Cohen's κ {pm2['kappa']}, defect recall {pm2['recall_reject']}; 100 levels, labels by construction"} if pm2 else need("phase2"))},
        {"key": "redteam_block", "label": "Red-team block rate",
         "value": pct(s_all["all_on"]["blocked"] / total, 0) + f" ({s_all['all_on']['blocked']}/{total})",
         "caption": f"{s_all['all_on']['false_alarms']}/{n['K']} false alarms; best single layer alone blocks {best_single}/{total}"},
        {"key": "p95_latency", "label": "p95 agent task latency",
         **({"value": f"{tr['task_p95_ms']:.0f} ms", "caption": f"{tr['tasks']:,} traced tasks of this scorecard run; tool-call p95 {tr['tool_p95_ms']:.1f} ms (traces.duckdb; wall-clock, so it varies with the machine)"} if tr else need("phase2"))},
        {"key": "cost_per_task", "label": "LLM USD cost / task",
         **({"value": f"${tr['llm_cost_per_task']:.5f}", "caption": f"mean over {tr['llm_tasks']} LLM-driven runs at mock pricing; hard cap $0.25/run; workflows spend $0"} if tr else need("phase2"))},
        {"key": "cuped", "label": "CUPED variance reduction",
         **({"value": f"{pm1['cuped_variance_reduction_pct']:.1f}%", "caption": f"ρ={pm1['cuped_rho']:.2f} on the seeded cohort ({pm1['n_without_cuped']:,}→{pm1['n_with_cuped']:,} users); ≈ρ² as theory says"} if pm1 else need("phase2"))},
        {"key": "d1_lift", "label": "D1 retention: guardrail catch",
         **({"value": f"{100 * pm1['harmful_d1_delta']:+.1f}pp caught",
             "caption": f"harmful variant rolled back on day {pm1['harmful_rollback_day']}; shipped variant D1 {100 * pm1['good_d1_delta']:+.1f}pp (always-valid CI spans 0: not claimed)"} if pm1 else need("phase2"))},
        {"key": "arpdau", "label": "ARPDAU lift: guarded bandit ops",
         **({"value": f"{100 * ad['guarded_arpdau_lift']:+.1f}%", "caption": f"72h incident timeline vs unguarded; TS floor alone {100 * ad['ts_vs_default_floor_lift']:+.1f}% vs $2 default; UA ROAS {100 * ad['ua_lift_vs_uniform']:+.1f}%"} if ad else need("ad-economy"))},
        {"key": "intake_f1", "label": "Intake ACORD extraction F1",
         **({"value": pct(intake["dev"]["extraction"]["f1"]), "caption": f"held-out phrasing {pct(intake['heldout']['extraction']['f1'])} (portal fill {intake['heldout']['portal_fill']['succeeded']}/{intake['heldout']['portal_fill']['should_submit']}); handoff recall {pct(intake['heldout']['handoff']['recall'], 0)}"} if intake else need("intake-eval"))},
    ]
    out = {"headline": head, "conformance": s_all, "traces": tr, "level_taste": taste,
           "phase2": p2, "ad_economy": {k: v for k, v in (ad or {}).items() if k != "checks"},
           "intake": {k: {kk: intake[k][kk] for kk in ("extraction", "portal_fill", "turns", "handoff", "calibration")} for k in ("dev", "heldout")} if intake else None}
    (DOCS / "scorecard.json").write_text(json.dumps(out, indent=2, default=str))
    md = ["# Scorecard", "", "| metric | value | what it means |", "|---|---|---|"] + \
         [f"| {h['label']} | **{h['value']}** | {h['caption']} |" for h in head]
    if "correlation_with_fail_rate" in taste:
        md += ["", "## Level taste", "", "| feature | correlation with observed fail rate (12 seeded levels) |", "|---|---|"] + \
              [f"| {k} | {v} |" for k, v in taste["correlation_with_fail_rate"].items()]
    c = taste.get("critic_on_fresh_candidates", {})
    if "n" in c:
        md += ["", f"Critic on {c['n']} fresh candidates: {c['approved']} approved ({pct(c['approval_rate'], 0)}).", "",
               "| top rejection reason | count |", "|---|---|"] + [f"| {r} | {k} |" for r, k in c["top_rejection_reasons"]]
    (DOCS / "SCORECARD.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
