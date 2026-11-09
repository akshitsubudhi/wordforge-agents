"""
================================================================================
BLOCK: LiveOps Experimentation Copilot (Project 1)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : data/telemetry.duckdb (events, players, levels, exposures),
             agents/level_factory (builds the proposed easier level),
             stats_engine (power, CUPED, mSPRT).
  Downstream: feature_flags (which version each player gets in the live game),
             levels (candidate -> live on ship, -> rolled_back on harm),
             approvals + agent_audit (who decided what, and why).
  Every action is a tool call through core.harness AgentRunner.workflow(), so
  guards, approvals, timeouts and spans apply to the copilot exactly as they
  apply to an LLM-driven agent.

Key engineering mechanism (the Level-7 playbook)
  1 scan_funnel        per-level fail rate on the ACTIVE version; anomaly if
                       fail >= 0.5 AND robust z = (x - median)/(1.4826 MAD) >= 3
                       vs the other levels.
  2 baseline_metrics   sigma, mean and CUPED rho of the primary metric (7-day
                       level completions) measured from the seeded cohort.
  3 power_analysis     sample size + duration at the capped exposure.
  4 propose_easier_level  the Level Factory builds a solver-proven,
                       critic-approved variant stored as a CANDIDATE version.
  5 configure_flag     rollout <= 20 %, enforced three times: L1 percent cap,
                       the tool itself, and a DuckDB CHECK constraint. It is
                       requires_approval, so the Human Approval Gate is asked
                       on every call.
  6 evaluate_experiment  replays every daily peek: CUPED on the primary metric
                       plus mSPRT always-valid CIs on the primary metric and
                       on the D1-retention guardrail.
  7 automatic rollback hook: daily_monitor() calls rollback_experiment
                       WITHOUT waiting for a human as soon as the guardrail's
                       always-valid CI lies entirely below -tolerance. Rollback
                       is the safe direction, so it is pre-approved. Shipping
                       is not: ship_experiment is requires_approval.

Failure modes & 1-step mitigation
  * Over-exposure request (e.g. 50 %) -> blocked by L1 before any approver
    sees it; the DB CHECK is the last line of defence.
  * Operator denies -> flag stays off; denial recorded in `approvals`.
  * Peeking every day -> safe by construction (mSPRT, Ville's inequality).
  * Retention harm -> automatic rollback within one peek of significance:
    flag set to 0 %, candidate version marked rolled_back, players served
    the control version on their next level_start.
================================================================================
CLI:  python -m agents.liveops_copilot.agent scan
      python -m agents.liveops_copilot.agent evaluate --flag level7_easier
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict
from typing import Callable

from pydantic import Field

from agents.common import ApprovalLedger, audit, interactive_decider, utcnow
from agents.liveops_copilot.simulator import COHORT_SQL
from agents.liveops_copilot.stats_engine import (
    MSPRT, cuped, guardrail_breached, sample_size_means, sample_size_proportions,
)
from core.config import get_settings
from core.db import DatabaseLockedError, TelemetryDB
from core.harness.llm import LLMClient, Message, MockLLM
from core.harness.redteam_guard import (
    READ_ONLY, READ_WRITE, AgentPolicy, ApprovalRequest, DefenseConfig, ExpectedAction,
)
from core.harness.runner import AgentRunner, AgentSpec, ToolCallBlocked
from core.harness.telemetry import Tracer, default_tracer
from core.harness.tools import ArgKind, Privilege, StrictArgs, ToolContext, ToolRegistry, ToolSpec
from core.levels import set_version_status

MAX_ROLLOUT_PCT = 20.0
COPILOT_TOOLS = frozenset({"scan_funnel", "query_telemetry", "baseline_metrics", "power_analysis",
                           "propose_easier_level", "configure_flag", "evaluate_experiment",
                           "rollback_experiment", "ship_experiment"})
FLAG_RE = r"^[a-z][a-z0-9_]{2,40}$"


# ============================================================================ tool args
class ScanArgs(StrictArgs):
    min_attempts: int = Field(default=200, ge=1)
    abs_threshold: float = Field(default=0.5, ge=0, le=1)
    z_threshold: float = Field(default=3.0, ge=0)


class SqlArgs(StrictArgs):
    sql: str = ArgKind("sql")


class BaselineArgs(StrictArgs):
    level_id: int


class PowerArgs(StrictArgs):
    baseline_mean: float = Field(gt=0)
    sigma: float = Field(gt=0)
    mde_rel: float = Field(gt=0, le=1)
    rho: float = Field(ge=-0.99, le=0.99)
    treatment_share_pct: float = ArgKind("percent", gt=0)
    daily_eligible_users: int = Field(gt=0)
    d1_baseline: float = Field(gt=0, lt=1)
    d1_mde_abs: float = Field(default=0.02, gt=0, lt=0.5)


class ProposeArgs(StrictArgs):
    level_id: int
    goal_fail_rate: float = Field(ge=0.02, le=0.8)
    size: int = Field(default=4, ge=4, le=6)
    time_limit_s: int = Field(default=90, ge=30, le=240)
    note: str = Field(default="", max_length=120)


class FlagArgs(StrictArgs):
    flag: str = Field(pattern=FLAG_RE)
    level_id: int
    treatment_version: int
    rollout_pct: float = ArgKind("percent", ge=0)


class EvalArgs(StrictArgs):
    flag: str = Field(pattern=FLAG_RE)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    mde_rel: float = Field(default=0.10, gt=0, le=1)
    guardrail_tolerance: float = Field(default=0.0, ge=0, le=0.2)
    d1_tau: float = Field(default=0.03, gt=0)
    min_days: int = Field(default=7, ge=1)
    max_days: int = Field(default=21, ge=1)


class RollbackArgs(StrictArgs):
    flag: str = Field(pattern=FLAG_RE)
    reason: str = Field(max_length=300)


class ShipArgs(StrictArgs):
    flag: str = Field(pattern=FLAG_RE)


# ============================================================================ copilot
class LiveOpsCopilot:
    SYSTEM = ("You are the LiveOps Experimentation Copilot for a word game. Find funnel anomalies, size experiments "
              "with power analysis, propose fixes, expose at most 20% of traffic behind human approval, evaluate "
              "with CUPED and always-valid sequential tests, and roll back when guardrails are harmed.")

    def __init__(self, db: TelemetryDB, *, factory=None, llm: LLMClient | None = None,
                 decide: Callable[[ApprovalRequest], tuple[bool, str]] | None = None, decided_by: str = "operator",
                 tracer: Tracer | None = None, defenses: DefenseConfig = DefenseConfig()):
        self.db, self.factory, self.llm = db, factory, llm or MockLLM("compliant")
        self.last_eval: dict[str, dict] = {}
        self.approvals = ApprovalLedger(db, "liveops_copilot", decide or interactive_decider, decided_by)
        self.runner = AgentRunner(
            AgentSpec("liveops_copilot", self.SYSTEM,
                      AgentPolicy(COPILOT_TOOLS, get_settings().data_dir, max_rollout_pct=MAX_ROLLOUT_PCT)),
            self._registry(), self.llm, defenses=defenses, tracer=tracer or default_tracer(), approver=self.approvals)

    # ------------------------------------------------------------------ tools
    def _t_scan(self, a: ScanArgs, ctx: ToolContext) -> dict:
        with self.db.cursor() as con:
            rows = con.execute("""
                SELECT l.level_id, l.version, count(*) AS attempts,
                       avg((e.event_name = 'level_fail')::INT) AS fail_rate
                FROM active_levels l JOIN events e ON e.level_id = l.level_id AND e.level_version = l.version
                WHERE e.event_name IN ('level_fail', 'level_complete')
                GROUP BY 1, 2 ORDER BY 1""").fetchall()
        levels = [{"level_id": r[0], "version": r[1], "attempts": r[2], "fail_rate": round(r[3], 4)} for r in rows]
        anomalies = []
        for lv in levels:
            others = [o["fail_rate"] for o in levels if o["level_id"] != lv["level_id"] and o["attempts"] >= a.min_attempts]
            if len(others) < 3 or lv["attempts"] < a.min_attempts:
                lv["robust_z"] = None
                continue
            med = statistics.median(others)
            mad = statistics.median(abs(x - med) for x in others) or 1e-6
            lv["robust_z"] = round((lv["fail_rate"] - med) / (1.4826 * mad), 2)
            if lv["fail_rate"] >= a.abs_threshold and lv["robust_z"] >= a.z_threshold:
                anomalies.append({**lv, "median_other_levels": round(med, 4),
                                  "message": f"Level {lv['level_id']} has a {lv['fail_rate']:.0%} fail rate "
                                             f"(median of other levels {med:.0%}, robust z {lv['robust_z']})"})
        return {"levels": levels, "anomalies": anomalies}

    def _t_query(self, a: SqlArgs, ctx: ToolContext) -> list[list]:
        with self.db.cursor() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                return [list(r) for r in con.execute(a.sql).fetchmany(200)]
            finally:
                con.execute("ROLLBACK")

    def _t_baseline(self, a: BaselineArgs, ctx: ToolContext) -> dict:
        with self.db.cursor() as con:
            rows = con.execute(COHORT_SQL).fetchall()
            d1 = con.execute("SELECT avg(retention_1::INT) FROM cookie_cats").fetchone()[0]
            reach = con.execute("""SELECT count(DISTINCT user_id) / count(DISTINCT ts::DATE) FROM events
                                   WHERE source='seed' AND event_name='level_start' AND level_id=?""", [a.level_id]).fetchone()[0]
        pre = [float(r[1]) for r in rows]
        post = [float(r[2]) for r in rows]
        rho = statistics.correlation(pre, post)
        return {"primary_metric": "level completions in the 7 days after exposure", "n_players": len(rows),
                "mean": round(statistics.fmean(post), 4), "sigma": round(statistics.stdev(post), 4),
                "cuped_covariate": "level completions in the 7 days before exposure", "rho": round(rho, 4),
                "d1_baseline": round(d1, 6), "seeded_daily_reach_of_level": round(reach or 0, 1)}

    def _t_power(self, a: PowerArgs, ctx: ToolContext) -> dict:
        share = a.treatment_share_pct / 100
        primary = sample_size_means(a.sigma, a.baseline_mean * a.mde_rel, baseline=a.baseline_mean, treatment_share=share,
                                    cuped_rho=a.rho, daily_eligible_users=a.daily_eligible_users)
        guard = sample_size_proportions(a.d1_baseline, -a.d1_mde_abs, treatment_share=share,
                                        daily_eligible_users=a.daily_eligible_users)
        return {"primary": primary.to_dict(), "guardrail_d1": guard.to_dict()}

    def _t_propose(self, a: ProposeArgs, ctx: ToolContext) -> dict:
        if self.factory is None:
            raise RuntimeError("no Level Factory attached")
        return self.factory.propose_variant(a.level_id, a.goal_fail_rate, a.size, a.time_limit_s,
                                            note=a.note or f"copilot variant goal={a.goal_fail_rate}")

    def _t_flag(self, a: FlagArgs, ctx: ToolContext) -> dict:
        if not 0 <= a.rollout_pct <= MAX_ROLLOUT_PCT:  # defence in depth behind L1
            raise ValueError(f"rollout {a.rollout_pct}% exceeds the {MAX_ROLLOUT_PCT}% cap")
        with self.db.cursor() as con:
            cand = con.execute("SELECT status FROM levels WHERE level_id=? AND version=?", [a.level_id, a.treatment_version]).fetchone()
            if not cand or cand[0] != "candidate":
                raise ValueError(f"level {a.level_id} v{a.treatment_version} is not a candidate version")
            control = con.execute("SELECT version FROM active_levels WHERE level_id=?", [a.level_id]).fetchone()[0]
            con.execute("""INSERT OR REPLACE INTO feature_flags VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
                        [a.flag, a.level_id, control, a.treatment_version, a.rollout_pct, utcnow(), utcnow(),
                         "approved via human approval gate"])
        audit(self.db, "liveops_copilot", "configure_flag", **a.model_dump(), control_version=control)
        return {"flag": a.flag, "status": "active", "control_version": control, **a.model_dump()}

    def _exposures(self, flag: str) -> list[tuple]:
        with self.db.cursor() as con:
            return con.execute("""
                SELECT x.variant, x.day, x.pre_metric,
                       coalesce(x.outcome, (SELECT count(*) FROM events e WHERE e.user_id = x.user_id
                                              AND e.event_name = 'level_complete'
                                              AND e.ts BETWEEN x.exposed_at AND x.exposed_at + INTERVAL 7 DAY)) AS y,
                       coalesce(x.retained_d1, EXISTS (SELECT 1 FROM events e WHERE e.user_id = x.user_id
                                              AND e.event_name = 'session_start'
                                              AND e.ts::DATE = x.exposed_at::DATE + 1)) AS d1
                FROM experiment_exposures x
                WHERE x.flag = ? AND (x.source = 'sim' OR x.exposed_at < now()::TIMESTAMP - INTERVAL 7 DAY)
                ORDER BY x.day""", [flag]).fetchall()

    def _t_evaluate(self, a: EvalArgs, ctx: ToolContext) -> dict:
        rows = self._exposures(a.flag)
        if not rows:
            return {"flag": a.flag, "days": 0, "recommendation": "continue", "reason": "no exposures yet"}
        days = sorted({r[1] for r in rows})
        ctrl_mean = statistics.fmean(r[3] for r in rows if r[0] == "control")
        primary = MSPRT(alpha=a.alpha, tau=max(1e-6, a.mde_rel * ctrl_mean))
        guard = MSPRT(alpha=a.alpha, tau=a.d1_tau)
        peeks = []
        for d in days:  # deterministic replay of every daily peek
            cum = [r for r in rows if r[1] <= d]
            t = [r for r in cum if r[0] == "treatment"]
            c = [r for r in cum if r[0] == "control"]
            if len(t) < 2 or len(c) < 2:
                continue
            cu = cuped([r[3] for r in t], [r[2] for r in t], [r[3] for r in c], [r[2] for r in c], a.alpha)
            p_step = primary.update(cu.effect_cuped, cu.var_cuped, len(t), len(c))
            g_step = guard.update_from_samples([float(r[4]) for r in t], [float(r[4]) for r in c])
            peeks.append({"day": d, "n_t": len(t), "n_c": len(c), "cuped": cu, "primary": p_step, "guardrail": g_step,
                          "guardrail_breached": guardrail_breached(g_step, a.guardrail_tolerance)})
        last = peeks[-1]
        if any(p["guardrail_breached"] for p in peeks):
            first = next(p for p in peeks if p["guardrail_breached"])
            rec, why = "rollback", (f"D1 guardrail harmed: always-valid CI [{first['guardrail'].ci_low:+.4f}, "
                                    f"{first['guardrail'].ci_high:+.4f}] below -{a.guardrail_tolerance} on day {first['day']}")
        elif last["primary"].reject_null and last["primary"].delta > 0 and last["day"] >= a.min_days:
            rec, why = "ship", (f"primary lift {last['cuped'].relative_lift_cuped:+.1%} significant "
                                f"(always-valid p={last['primary'].p_always_valid:.4g}) after {last['day']} days; guardrail OK")
        elif last["primary"].reject_null and last["primary"].delta < 0:
            rec, why = "rollback", "primary metric significantly worse"
        elif last["day"] >= a.max_days:
            rec, why = "stop_inconclusive", f"no decision after {a.max_days} days"
        else:
            rec, why = "continue", "not yet conclusive" if last["day"] >= a.min_days else f"minimum runtime {a.min_days} days"
        out = {
            "flag": a.flag, "days": last["day"], "recommendation": rec, "reason": why,
            "cuped": last["cuped"].to_dict(),
            "peeks": [{"day": p["day"], "n_t": p["n_t"], "n_c": p["n_c"], "delta_cuped": round(p["primary"].delta, 5),
                       "ci": [round(p["primary"].ci_low, 5), round(p["primary"].ci_high, 5)],
                       "lr": round(p["primary"].likelihood_ratio, 3), "p_av": round(p["primary"].p_always_valid, 6),
                       "d1_delta": round(p["guardrail"].delta, 5),
                       "d1_ci": [round(p["guardrail"].ci_low, 5), round(p["guardrail"].ci_high, 5)],
                       "guardrail_breached": p["guardrail_breached"]} for p in peeks],
        }
        self.last_eval[a.flag] = out
        return out

    def _flag(self, con, flag: str):
        row = con.execute("SELECT level_id, control_version, treatment_version, status FROM feature_flags WHERE flag=?", [flag]).fetchone()
        if not row:
            raise ValueError(f"unknown flag {flag}")
        return row

    def _t_rollback(self, a: RollbackArgs, ctx: ToolContext) -> dict:
        with self.db.cursor() as con:
            lid, ctrl, treat, status = self._flag(con, a.flag)
            if status in ("rolled_back", "shipped"):
                return {"flag": a.flag, "status": status, "noop": True}
            con.execute("UPDATE feature_flags SET status='rolled_back', rollout_pct=0, updated_at=?, notes=? WHERE flag=?",
                        [utcnow(), a.reason[:300], a.flag])
            set_version_status(con, lid, treat, "rolled_back", a.reason[:120])
        audit(self.db, "liveops_copilot", "rollback_experiment", flag=a.flag, reason=a.reason)
        return {"flag": a.flag, "status": "rolled_back", "level_id": lid, "serving_version": ctrl}

    def _t_ship(self, a: ShipArgs, ctx: ToolContext) -> dict:
        with self.db.cursor() as con:
            lid, ctrl, treat, status = self._flag(con, a.flag)
            if status != "active":
                raise ValueError(f"flag {a.flag} is {status}, cannot ship")
            set_version_status(con, lid, treat, "live", f"shipped via {a.flag}")
            con.execute("UPDATE feature_flags SET status='shipped', rollout_pct=0, updated_at=? WHERE flag=?", [utcnow(), a.flag])
        audit(self.db, "liveops_copilot", "ship_experiment", flag=a.flag, level_id=lid, version=treat)
        return {"flag": a.flag, "status": "shipped", "level_id": lid, "live_version": treat}

    def _registry(self) -> ToolRegistry:
        R, W = Privilege.READ, Privilege.WRITE
        return ToolRegistry([
            ToolSpec("scan_funnel", "Per-level fail rates with robust-z anomaly detection", ScanArgs, self._t_scan, R),
            ToolSpec("query_telemetry", "Read-only SQL (always rolled back)", SqlArgs, self._t_query, R, untrusted_output=True),
            ToolSpec("baseline_metrics", "Primary metric mean/sigma and CUPED rho from telemetry", BaselineArgs, self._t_baseline, R, timeout_s=30),
            ToolSpec("power_analysis", "Sample size and duration at capped exposure", PowerArgs, self._t_power, R),
            ToolSpec("propose_easier_level", "Level Factory builds a candidate variant", ProposeArgs, self._t_propose, W,
                     idempotent=False, timeout_s=120),
            ToolSpec("configure_flag", "Activate a feature flag (<=20% traffic, human approval)", FlagArgs, self._t_flag, W,
                     idempotent=False, requires_approval=True),
            ToolSpec("evaluate_experiment", "CUPED + mSPRT evaluation with guardrail", EvalArgs, self._t_evaluate, R, timeout_s=60),
            ToolSpec("rollback_experiment", "Turn the flag off and mark the candidate rolled back", RollbackArgs, self._t_rollback, W),
            ToolSpec("ship_experiment", "Promote the candidate to live (human approval)", ShipArgs, self._t_ship, W,
                     idempotent=False, requires_approval=True),
        ])

    # ------------------------------------------------------------------ playbook steps
    def scan(self) -> dict:
        with self.runner.workflow("scan", mode=READ_ONLY) as wf:
            return wf.call("scan_funnel")

    def plan(self, level_id: int, *, daily_users: int, mde_rel: float = 0.10, rollout_pct: float = 20.0) -> dict:
        with self.runner.workflow(f"plan:L{level_id}", mode=READ_ONLY) as wf:
            base = wf.call("baseline_metrics", level_id=level_id)
            power = wf.call("power_analysis", baseline_mean=base["mean"], sigma=base["sigma"], mde_rel=mde_rel,
                            rho=base["rho"], treatment_share_pct=rollout_pct, daily_eligible_users=daily_users,
                            d1_baseline=base["d1_baseline"])
        return {"baseline": base, "power": power}

    def launch(self, flag: str, level_id: int, goal_fail_rate: float, *, rollout_pct: float = 20.0,
               size: int = 4, time_limit_s: int = 90) -> dict:
        """Propose a candidate, then request the flag. Returns what happened, including any block."""
        intent = [ExpectedAction("configure_flag", {"flag": flag, "level_id": level_id})]
        with self.runner.workflow(f"launch:{flag}", mode=READ_WRITE, intent=intent,
                                  trusted_context=f"launch {flag} for level {level_id}") as wf:
            cand = wf.call("propose_easier_level", level_id=level_id, goal_fail_rate=goal_fail_rate, size=size,
                           time_limit_s=time_limit_s, note=f"{flag} goal={goal_fail_rate}")
            rec = wf.call_record("configure_flag", flag=flag, level_id=level_id, treatment_version=cand["version"],
                                 rollout_pct=rollout_pct)
        blocked = rec.decision if rec.decision and not rec.decision.allowed else None
        if blocked:
            audit(self.db, "liveops_copilot", "flag_blocked", flag=flag, layer=blocked.layer, rule=blocked.rule)
        return {"candidate": cand, "flag_result": rec.raw, "blocked": asdict(blocked) if blocked else None}

    def request_flag(self, flag: str, level_id: int, treatment_version: int, rollout_pct: float) -> dict:
        with self.runner.workflow(f"flag:{flag}", mode=READ_WRITE) as wf:
            rec = wf.call_record("configure_flag", flag=flag, level_id=level_id, treatment_version=treatment_version,
                                 rollout_pct=rollout_pct)
        return {"ok": rec.ok, "blocked": asdict(rec.decision) if rec.decision and not rec.decision.allowed else None}

    def daily_monitor(self, flag: str, **eval_kw) -> dict:
        """Run after each day of traffic. Contains the AUTOMATIC ROLLBACK HOOK."""
        with self.runner.workflow(f"monitor:{flag}", mode=READ_WRITE) as wf:
            ev = wf.call("evaluate_experiment", flag=flag, **eval_kw)
            action = None
            if ev["recommendation"] == "rollback":
                action = wf.call("rollback_experiment", flag=flag, reason=ev["reason"])
            elif ev["recommendation"] == "ship":
                rec = wf.call_record("ship_experiment", flag=flag)
                action = rec.raw if rec.ok else {"blocked": asdict(rec.decision) if rec.decision else "error"}
        return {"evaluation": ev, "action": action}

    def summarize(self, facts: dict) -> str:
        """Analyst summary: a real LLM writes it when configured; otherwise a deterministic template."""
        if not isinstance(self.llm, MockLLM):
            res = self.runner.run("Write a 5-bullet LiveOps summary for the CTO from these facts. Facts:\n"
                                  + json.dumps(facts, default=str)[:6000])
            if res.final_text:
                return res.final_text
        a = facts.get("anomaly", {})
        lines = [f"- Detected: {a.get('message', 'no anomaly')}"]
        for name, s in facts.get("experiments", {}).items():
            lines.append(f"- {name}: {s['outcome']} - {s['reason']}")
        return "\n".join(lines)


# ============================================================================ operator policy for evals
def simulated_operator(allowed_flags: set[str], copilot_ref: list) -> Callable[[ApprovalRequest], tuple[bool, str]]:
    """Scripted stand-in for the human: approves planned flags <= cap and ships only on a 'ship' recommendation."""
    def decide(req: ApprovalRequest) -> tuple[bool, str]:
        if req.tool == "configure_flag":
            if req.args.get("flag") not in allowed_flags:
                return False, "flag not in the approved experiment plan"
            if req.args.get("rollout_pct", 100) > MAX_ROLLOUT_PCT:
                return False, "exposure above policy"
            return True, "planned experiment within exposure policy"
        if req.tool == "ship_experiment":
            ev = copilot_ref[0].last_eval.get(req.args.get("flag"), {}) if copilot_ref else {}
            ok = ev.get("recommendation") == "ship"
            return ok, ev.get("reason", "no evaluation") if ok else "no ship recommendation on record"
        return False, f"operator does not approve {req.tool}"
    return decide


def main() -> int:
    ap = argparse.ArgumentParser(description="LiveOps Experimentation Copilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    e = sub.add_parser("evaluate")
    e.add_argument("--flag", required=True)
    a = ap.parse_args()
    try:
        db = TelemetryDB(get_settings().telemetry_db)
    except DatabaseLockedError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    cop = LiveOpsCopilot(db)
    out = cop.scan() if a.cmd == "scan" else cop.daily_monitor(a.flag)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
