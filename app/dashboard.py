"""
================================================================================
BLOCK: Evaluator Control Center (backend)
--------------------------------------------------------------------------------
Role & contracts
  One FastAPI router that puts every project behind a single page
  (/control-center, app/static/control_center.html):
    Overview     scorecard (docs/scorecard.json) + live system counters
    Game         the live LetterLoop game + per-version funnel
    SRE & Red-Team  conformance ablation, p50/p95/p99 from traces.duckdb,
                 recent traces with a span waterfall per trace
    Copilot      funnel scan + power analysis + a LIVE A/B experiment with a
                 REAL human approval gate (the job pauses for your click)
    Level Factory  "Generate, Verify & Push Level to Live Game"
    Ad Economy   mechanism comparison, Thompson floors, UA bandit
    Insurance Intake  one synthetic customer, end to end, day by day
    Fault drills five injected production failures + the guardrail that
                 catches each (app/fault_drills.py)
  Everything runs IN the server process on the server's own DuckDB writer, so
  there is no lock fight with the game.

Key engineering mechanism
  * JobQueue: one worker thread, FIFO. Jobs never contend for DuckDB, and every
    step is appended to a log the page polls (GET /api/cc/jobs/{id}?since=n),
    so the evaluator watches guardrails fire in real time.
  * Human approval gate: copilot approvals block on a threading.Event until
    POST /api/cc/approvals/{id}; they DENY automatically after 120 s
    (fail-closed).
  * Traces: the default tracer exports to traces.duckdb; jobs flush it on
    completion so trace ids in the UI resolve immediately.

Failure modes & 1-step mitigation
  * Job crashes -> status=error with the exception; the queue keeps running.
  * Nobody answers an approval -> denied after the timeout; flag never
    activates.
  * Drills and experiments only touch sandboxed or owned state; experiment
    flags are unique per run and always end shipped, rolled back or stopped.
================================================================================
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import duckdb
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.config import REPO_ROOT, get_settings
from core.harness.llm import HEALTH, llm_status
from core.harness.telemetry import TracesBusyError, agent_metrics, default_tracer, traces_connection

router = APIRouter(tags=["control-center"])
STATIC = Path(__file__).parent / "static"
SCORECARD = REPO_ROOT / "docs" / "scorecard.json"


# ============================================================================ jobs
@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"          # queued | running | done | error
    steps: list[dict] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None


class JobQueue:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.handlers: dict[str, Callable[[Job, Callable[[dict], None]], Any]] = {}
        self._worker = threading.Thread(target=self._run, daemon=True, name="cc-jobs")
        self._worker.start()

    def submit(self, kind: str, params: dict) -> Job:
        if kind not in self.handlers:
            raise HTTPException(404, f"unknown job kind {kind}")
        job = Job(uuid.uuid4().hex[:10], kind, params)
        with self.lock:
            self.jobs[job.id] = job
            if len(self.jobs) > 200:
                for old in sorted(self.jobs.values(), key=lambda j: j.created)[:50]:
                    self.jobs.pop(old.id, None)
        self.q.put(job.id)
        return job

    def _run(self) -> None:
        while True:
            job = self.jobs.get(self.q.get())
            if job is None:
                continue
            job.status = "running"
            log = lambda s, j=job: j.steps.append({"i": len(j.steps), "t": round(time.time(), 3), **s})
            before = HEALTH.snapshot()["calls"]
            try:
                job.result = self.handlers[job.kind](job, log)
                job.status = "done"
                st = llm_status()
                if st["mode"] == "live" and not st["ok"] and HEALTH.snapshot()["calls"] != before:
                    # the job finished on the deterministic fallback: say so, with the fix, instead of hiding it
                    log({"status": "info", "title": f"LLM provider unavailable during this job: {st['message']}",
                         "detail": f"{st['action']} Last error: {st['last_error'][:200]}", "llm": st})
            except Exception as e:  # never kill the worker
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
                log({"status": "bad", "title": "job failed", "detail": traceback.format_exc()[-800:]})
            finally:
                job.finished = time.time()
                try:
                    default_tracer().exporter.flush(force=True)
                except Exception:
                    pass


class ApprovalBroker:
    """Human-in-the-loop approvals for the UI. Fail-closed after `timeout_s`."""

    def __init__(self, timeout_s: float = 120.0):
        self.timeout_s = timeout_s
        self.pending: dict[str, dict] = {}

    def ask(self, log, tool: str, args: dict) -> tuple[bool, str]:
        aid = uuid.uuid4().hex[:8]
        ev = threading.Event()
        self.pending[aid] = {"event": ev, "decision": None, "tool": tool, "args": args}
        log({"status": "approval", "title": f"HUMAN APPROVAL REQUIRED: {tool}", "detail": json.dumps(args, default=str),
             "approval_id": aid})
        ok = ev.wait(self.timeout_s)
        dec = self.pending.pop(aid, {}).get("decision") if ok else None
        if dec is None:
            log({"status": "guard", "title": "No answer within 120 s - denied (fail-closed)", "detail": tool})
            return False, "timed out (fail-closed)"
        log({"status": "ok" if dec else "guard", "title": f"Human {'APPROVED' if dec else 'DENIED'} {tool}", "detail": ""})
        return dec, f"human {'approved' if dec else 'denied'} via Control Center"

    def answer(self, aid: str, approve: bool) -> bool:
        p = self.pending.get(aid)
        if not p:
            return False
        p["decision"] = approve
        p["event"].set()
        return True


def cc_state(request: Request) -> dict:
    app = request.app
    if not hasattr(app.state, "cc"):
        app.state.cc = {"jobs": JobQueue(), "approvals": ApprovalBroker(), "factory": None, "lock": threading.Lock()}
        register_handlers(app)
    return app.state.cc


def factory_for(app):
    cc = app.state.cc
    with cc["lock"]:
        if cc["factory"] is None:
            from agents.level_factory.critic_and_eval import LevelFactory

            cc["factory"] = LevelFactory(app.state.db, tracer=default_tracer(), seed=int(time.time()) % 10_000)
        return cc["factory"]


# ============================================================================ job handlers
def register_handlers(app) -> None:
    jobs: JobQueue = app.state.cc["jobs"]
    db = app.state.db
    H = jobs.handlers

    def fault(kind):
        def run(job, log):
            from app import fault_drills as fd

            if kind == "hallucinated_level":
                return fd.drill_hallucinated_level(db, factory_for(app), log)
            if kind == "network_dark":
                return fd.drill_network_dark(db, default_tracer(), log)
            if kind == "peeking":
                return fd.drill_peeking(default_tracer(), log)
            if kind == "runaway_cost":
                return fd.drill_runaway_cost(default_tracer(), log)
            return fd.drill_prompt_injection(default_tracer(), log)
        return run

    for k in ("hallucinated_level", "network_dark", "peeking", "prompt_injection", "runaway_cost"):
        H[f"fault_{k}"] = fault(k)

    def conformance(job, log):
        from core.harness.run_redteam_eval import run_suite

        log({"status": "info", "title": "Running 35 fixtures x 5 defence configs through the real AgentRunner", "detail": ""})
        r = run_suite(default_tracer())
        s = r["summary"]["all_on"]
        log({"status": "ok" if s["blocked"] == 25 and s["false_alarms"] == 0 else "bad",
             "title": f"all_on blocked {s['blocked']}/25, false alarms {s['false_alarms']}/10", "detail": f"{len(r['outcomes'])} runs in {r['seconds']}s"})
        return {"summary": r["summary"], "counts": r["counts"], "seconds": r["seconds"],
                "fixtures": r["fixtures"], "outcomes": [{k: o[k] for k in ("fixture", "category", "config", "blocked", "passed", "fired")}
                                                        for o in r["outcomes"]]}
    H["conformance"] = conformance

    def factory_push(job, log):
        f = factory_for(app)
        goal = float(job.params.get("goal", 0.3))
        size = int(job.params.get("size", 4))
        tl = int(job.params.get("time_limit_s", 100))
        n = int(job.params.get("candidates", 24))
        from core.harness.redteam_guard import READ_WRITE

        log({"status": "info", "title": f"Generating {n} candidates: {size}x{size}, {tl}s, target fail rate {goal:.0%}",
             "detail": f"generator: {'LLM (' + f.llm.model + ') + heuristic' if f.generator.uses_llm else 'deterministic heuristic (mock LLM)'}"})
        with f.runner.workflow("cc:factory_push", mode=READ_WRITE) as wf:
            cands = wf.call("generate_candidates", goal=goal, size=size, time_limit_s=tl, n=n, slot="control-center")
            best, rows = None, []
            for c in cands:
                proof = wf.call("prove_level", grid=c["grid"], target_words=c["target_words"], time_limit_s=tl)
                row = {"grid": c["grid"], "words": proof["word_count"], "target": c["target_words"], "proof": proof["ok"],
                       "pred": c["predicted_fail_rate"], "witness_rechecked": proof["witness_rechecked"]}
                if proof["ok"] and c["predicted_fail_rate"] is not None and abs(c["predicted_fail_rate"] - goal) <= 0.12:
                    rep = wf.call("critique_level", grid=c["grid"], target_words=c["target_words"], time_limit_s=tl, goal=goal)
                    row.update(critic=rep["scores"], approve=rep["approve"],
                               failed=[r + ": " + rep["reasons"][r] for r, s in rep["scores"].items() if s < 3])
                    if rep["approve"]:
                        key = (sum(rep["scores"].values()), -abs(rep["predicted_fail_rate"] - goal))
                        if best is None or key > best[0]:
                            best = (key, c, rep)
                else:
                    row.update(approve=False, failed=["difficulty band" if proof["ok"] else proof["reason"]])
                rows.append(row)
                log({"status": "ok" if row.get("approve") else "info",
                     "title": f"{c['grid']}  {proof['word_count']} words, target {c['target_words']}, pred fail "
                              f"{c['predicted_fail_rate']}", "detail": "APPROVED by critic" if row.get("approve") else "; ".join(row["failed"])[:160],
                     "candidate": row})
            if best is None:
                log({"status": "bad", "title": "No candidate passed every gate", "detail": "nothing published"})
                return {"published": None, "candidates": rows}
            c = best[1]
            pub = wf.call("publish_level", level_id=0, grid=c["grid"], target_words=c["target_words"], time_limit_s=tl,
                          status="live", notes=f"daily:{time.strftime('%Y-%m-%d')} slot=control-center")
            trace = wf.root.trace_id
        log({"status": "guard", "title": f"Release gate approved: proof ok + critic {best[2]['scores']}", "detail": f"trace {trace}"})
        log({"status": "ok", "title": f"LIVE: level {pub['level_id']} is now in the game rotation", "detail": c["grid"],
             "level_id": pub["level_id"]})
        return {"published": {**pub, "grid": c["grid"], "target_words": c["target_words"], "time_limit_s": tl,
                              "critic": best[2]["scores"]}, "candidates": rows, "trace_id": trace}
    H["factory_push"] = factory_push

    def golden(job, log):
        from agents.level_factory.critic_and_eval import CriticAgent, evaluate_critic, load_golden

        f = factory_for(app)
        seeds = [set(json.loads(r["valid_words"])) for r in db.query("SELECT valid_words FROM levels WHERE created_by='seed'")]
        ev = evaluate_critic(CriticAgent(f.solver, f.model, seeds), load_golden())
        log({"status": "ok", "title": f"agreement {ev['agreement_pct']}%, kappa {ev['cohens_kappa']}", "detail": ""})
        return {k: v for k, v in ev.items() if k != "disagreements"} | {"disagreements": [
            {"id": d["id"], "defect": d["defect"], "failed": d["failed"]} for d in ev["disagreements"]]}
    H["golden_eval"] = golden

    def copilot_scan(job, log):
        from agents.liveops_copilot.agent import LiveOpsCopilot

        cop = LiveOpsCopilot(db, factory=factory_for(app), decide=lambda r: (False, "scan only"), tracer=default_tracer())
        scan = cop.scan()
        for a in scan["anomalies"]:
            log({"status": "fault", "title": a["message"], "detail": ""})
        plan = cop.plan(7, daily_users=int(job.params.get("daily_users", 4000)))
        p = plan["power"]["primary"]
        log({"status": "info", "title": f"Power: {p['n_total']:,} users with CUPED (vs {p['n_total_without_cuped']:,}) at 20% exposure",
             "detail": f"rho {plan['baseline']['rho']}, sigma {plan['baseline']['sigma']}"})
        return {"scan": scan, "plan": plan}
    H["copilot_scan"] = copilot_scan

    def copilot_experiment(job, log):
        from agents.liveops_copilot.agent import LiveOpsCopilot
        from agents.liveops_copilot.simulator import Effect, ExperimentSimulator

        scen = job.params.get("scenario", "good")
        eff = Effect(0.12, 0.01) if scen == "good" else Effect(-0.03, -0.06)
        goal = 0.35 if scen == "good" else 0.06
        flag = f"cc_{scen}_{int(time.time())}"
        broker: ApprovalBroker = app.state.cc["approvals"]
        human = job.params.get("human_approval", True)
        ref: list = []

        def decide(req):
            if req.tool == "ship_experiment":
                ev = ref[0].last_eval.get(req.args.get("flag"), {})
                if ev.get("recommendation") != "ship":
                    return False, "no ship recommendation"
            return broker.ask(log, req.tool, req.args) if human else (True, "auto-approved (human gate off)")

        cop = LiveOpsCopilot(db, factory=factory_for(app), decide=decide, decided_by="control_center_human" if human else "auto",
                             tracer=default_tracer())
        ref.append(cop)
        log({"status": "info", "title": f"Building a level-7 variant (goal fail {goal:.0%}) with the Level Factory", "detail": flag})
        launch = cop.launch(flag, 7, goal_fail_rate=goal, rollout_pct=20)
        cand = launch["candidate"]
        log({"status": "info", "title": f"Candidate level 7 v{cand['version']} ({cand['grid']}, target {cand['target_words']}/{cand['word_count']})",
             "detail": f"predicted fail {cand['predicted_fail_rate']}"})
        if launch["blocked"]:
            log({"status": "guard", "title": f"Flag not activated: {launch['blocked']['layer']}:{launch['blocked']['rule']}",
                 "detail": launch["blocked"]["detail"]})
            from core.levels import set_version_status
            with db.cursor() as con:
                set_version_status(con, 7, cand["version"], "rolled_back", "flag denied")
            return {"flag": flag, "outcome": "denied"}
        log({"status": "ok", "title": f"Flag {flag} ACTIVE at 20% of level-7 traffic", "detail": "players on /game are now bucketed"})
        sim = ExperimentSimulator(db, seed=int(time.time()) % 1000)
        users = int(job.params.get("daily_users", 3000))
        res, peeks = None, []
        for day in range(1, 15):
            sim.simulate_day(flag, day, users, 0.20, eff)
            res = cop.daily_monitor(flag, mde_rel=0.10, min_days=7, max_days=14)
            p = res["evaluation"]["peeks"][-1]
            peeks.append(p)
            log({"status": "bad" if p["guardrail_breached"] else "info",
                 "title": f"day {day}: CUPED delta {p['delta_cuped']:+.3f}  always-valid CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  p_av {p['p_av']:.4g}",
                 "detail": f"D1 delta {p['d1_delta']:+.4f} CI [{p['d1_ci'][0]:+.4f}, {p['d1_ci'][1]:+.4f}]", "peek": p})
            if res["action"] is not None:
                break
        ev, act = res["evaluation"], res["action"] or {}
        if act.get("status") == "rolled_back":
            log({"status": "guard", "title": "AUTO-ROLLBACK: D1 guardrail proven harmed", "detail": ev["reason"]})
        elif act.get("status") == "shipped":
            log({"status": "ok", "title": f"SHIPPED: level 7 v{act['live_version']} is now live for everyone", "detail": ev["reason"]})
        else:
            log({"status": "info", "title": f"Final: {ev['recommendation']}", "detail": ev["reason"]})
            with cop.runner.workflow(f"cc:cleanup:{flag}") as wf:
                wf.call_record("rollback_experiment", flag=flag, reason=f"control-center run ended: {ev['recommendation']}")
        return {"flag": flag, "scenario": scen, "peeks": peeks, "cuped": ev.get("cuped"), "recommendation": ev["recommendation"],
                "reason": ev["reason"], "action": act}
    H["copilot_experiment"] = copilot_experiment

    def ad_mechanisms(job, log):
        from agents.ad_economy.auction_sim import MediationSimulator, revenue_equivalence_check, uniform_closed_form_check

        n = int(job.params.get("impressions", 15000))
        sim = MediationSimulator(seed=1)
        rows = []
        for m, kw in [("waterfall", {}), ("waterfall_strategic", {}), ("hb_second", {"reserve": 2.0}),
                      ("hb_first_truthful", {"reserve": 2.0}), ("hb_first_shaded", {"reserve": 2.0})]:
            r = sim.run(m, n, **kw)
            rows.append(dict(zip(["mechanism", "requests", "fill", "ecpm", "ecpm_filled", "lat_mean", "lat_p95", "efficiency", "surplus"], r.row())))
            log({"status": "info", "title": f"{m}: eCPM ${r.ecpm:.2f}, latency {r.latency_mean_ms:.0f} ms, efficiency {r.efficiency:.1%}", "detail": ""})
        u = uniform_closed_form_check(5)
        re_ = revenue_equivalence_check(auctions=40_000)
        log({"status": "ok", "title": f"BNE shading = closed form (max err {u['max_abs_error']:.1e}); revenue equivalence z = {re_['z']}", "detail": ""})
        return {"rows": rows, "uniform": u, "revenue_equivalence": re_}
    H["ad_mechanisms"] = ad_mechanisms

    def ad_floor(job, log):
        from agents.ad_economy.auction_sim import MediationSimulator
        from agents.ad_economy.bandit_agent import tune_floor

        ts = tune_floor(MediationSimulator(seed=5), [0, 2, 4, 6, 8, 10, 12, 14], rounds=120, batch=300, oracle_n=20_000)
        log({"status": "ok", "title": f"TS floor ${ts['ts_floor']} (oracle ${ts['oracle_floor']}), regret ${ts['regret_usd']} vs ${ts['uniform_regret_usd']} uniform", "detail": ""})
        return ts
    H["ad_floor"] = ad_floor

    def ad_ua(job, log):
        from agents.ad_economy.bandit_agent import run_ua_bandit

        ua = run_ua_bandit(db, None, days=int(job.params.get("days", 40)))
        t = ua["totals"]
        log({"status": "ok", "title": f"LinTS ROAS {t['lints']['roas']:.4f} vs uniform {t['uniform']['roas']:.4f} vs oracle {t['oracle']['roas']:.4f}", "detail": ""})
        return ua
    H["ad_ua"] = ad_ua

    def intake_case(job, log):
        from agents.insurance_intake.synthetic import build_cases
        from agents.insurance_intake.workflow import EventStore
        from scripts.run_intake_eval import run_case

        scen, style = job.params.get("scenario", "clean"), job.params.get("style", "dev")
        case = next(c for c in build_cases(50, style=style) if c.scenario == scen)
        store_dir = get_settings().data_dir / "cc_intake"
        store_dir.mkdir(parents=True, exist_ok=True)
        store = EventStore(store_dir / f"{job.id}.sqlite")
        log({"status": "info", "title": f"Customer: {case.truth['contact_name']} - {case.truth['applicant_name']} ({scen}, {style} phrasing)",
             "detail": "day 1 e-mail -> day 2 Twilio SMS -> day 3 voice call; a fresh engine process every day"})
        seen = {"events": 0}

        def on_day(day, eng, cid):
            evs = store.events(cid)
            for e in evs[seen["events"]:]:
                p = e["payload"]
                if e["type"] == "message_received":
                    m = p["message"]
                    log({"status": "info", "title": f"day {day} INBOUND {m['channel'].upper()} from {m['contact']}",
                         "detail": m["text"][:400], "channel": m["channel"]})
                elif e["type"] == "activity_completed" and p["kind"] == "followup" and p["result"].get("question"):
                    log({"status": "info", "title": f"day {day} OUTBOUND {p['result']['channel'].upper()} follow-up",
                         "detail": p["result"]["question"]})
                elif e["type"] == "activity_completed" and p["kind"] == "handoff":
                    log({"status": "guard", "title": "HUMAN HANDOFF", "detail": "; ".join(p["result"].get("reasons", []))})
                elif e["type"] == "activity_completed" and p["kind"] == "portal":
                    r = p["result"]
                    log({"status": "ok" if r.get("ok") else "bad", "title": f"CARRIER PORTAL {'accepted ' + str(r.get('submission_id')) if r.get('ok') else 'rejected'}",
                         "detail": "; ".join(r.get("errors", []))})
            seen["events"] = len(evs)

        r = run_case(case, store, default_tracer(), on_day=on_day)
        ext = r["extraction"]
        fields = {k: {"value": f.value, "confidence": round(f.confidence, 3), "status": f.status,
                      "truth": case.truth.get(k), "quote": f.citations[0].quote if f.citations else None,
                      "channel": f.citations[0].channel if f.citations else None} for k, f in ext.fields.items()}
        st = r["state"]
        correct = sum(1 for k, v in fields.items() if v["truth"] not in (None, []) and v["value"] == v["truth"])
        present = sum(1 for v in fields.values() if v["truth"] not in (None, []))
        log({"status": "ok" if st.status == "SUBMITTED" else "guard", "title": f"FINAL: {st.status}",
             "detail": f"{correct}/{present} fields match ground truth; {st.turns_in + st.turns_out} turns"})
        from agents.insurance_intake.portal_automation import get_submission

        return {"case": case.case_no, "scenario": scen, "style": style, "status": st.status, "fields": fields,
                "handoff_reasons": st.handoff_reasons, "submission": get_submission(st.submission_id) if st.submission_id else None,
                "submission_id": st.submission_id, "turns": st.turns_in + st.turns_out, "expect_handoff": case.expect_handoff}
    H["intake_case"] = intake_case

    def reset_demo(job, log):
        from scripts.verify_phase2 import reset_phase2

        reset_phase2(db)  # factory levels, flags, approvals, exposures, audit, source='sim' events; seed + live kept
        app.state.game.reset()
        log({"status": "ok", "title": "Demo state reset", "detail": "level 7 serves its seeded version again"})
        return {"reset": True}
    H["reset_demo"] = reset_demo


# ============================================================================ routes
class JobIn(BaseModel):
    kind: str = Field(pattern=r"^[a-z_]{3,40}$")
    params: dict = Field(default_factory=dict)


@router.get("/control-center", include_in_schema=False)
def control_center_page():
    return FileResponse(STATIC / "control_center.html", media_type="text/html")


@router.post("/api/cc/jobs")
def submit_job(body: JobIn, request: Request) -> dict:
    job = cc_state(request)["jobs"].submit(body.kind, body.params)
    return {"job_id": job.id, "status": job.status}


@router.get("/api/cc/jobs/{job_id}")
def job_status(job_id: str, request: Request, since: int = 0) -> dict:
    job = cc_state(request)["jobs"].jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {"job_id": job.id, "kind": job.kind, "status": job.status, "steps": job.steps[since:], "next": len(job.steps),
            "result": job.result if job.status in ("done", "error") else None, "error": job.error,
            "seconds": round((job.finished or time.time()) - job.created, 2), "llm": llm_status()}


@router.post("/api/cc/approvals/{approval_id}")
def approve(approval_id: str, request: Request, decision: str = "approve") -> dict:
    ok = cc_state(request)["approvals"].answer(approval_id, decision == "approve")
    if not ok:
        raise HTTPException(404, "no pending approval with that id")
    return {"approval_id": approval_id, "decision": decision}


def _traces_query(sql: str, params: list | None = None) -> list[dict]:
    try:
        default_tracer().exporter.flush(force=True)
    except Exception:
        pass
    try:
        with traces_connection(get_settings().traces_db) as con:
            cur = con.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except TracesBusyError as e:  # another PROCESS holds the spans DB: tell the client to retry, never a 500
        raise HTTPException(503, str(e), headers={"Retry-After": "1"})


@router.get("/api/cc/overview")
def overview(request: Request) -> dict:
    cc_state(request)
    db = request.app.state.db
    counts = db.query("""SELECT (SELECT count(*) FROM events) AS events,
                                (SELECT count(*) FROM events WHERE source='live') AS live_events,
                                (SELECT count(*) FROM active_levels) AS active_levels,
                                (SELECT count(*) FROM levels WHERE created_by='level_factory' AND status='live') AS factory_levels,
                                (SELECT count(*) FROM feature_flags WHERE status='active') AS active_flags,
                                (SELECT count(*) FROM approvals) AS approvals""")[0]
    try:
        spans = _traces_query("SELECT count(*) AS spans, count(DISTINCT trace_id) AS traces FROM spans")[0]
    except Exception as e:
        spans = {"spans": None, "traces": None, "error": str(e)[:120]}
    score = json.loads(SCORECARD.read_text()) if SCORECARD.exists() else None
    return {"counts": counts, "traces": spans, "scorecard": score, "llm": llm_status()}


@router.get("/api/cc/metrics")
def metrics(since_minutes: float | None = None) -> dict:
    try:
        default_tracer().exporter.flush(force=True)
        return {"metrics": agent_metrics(get_settings().traces_db, since_minutes)}
    except Exception as e:
        return {"metrics": [], "error": f"{type(e).__name__}: {str(e)[:160]}"}


@router.get("/api/cc/traces/recent")
def recent_traces(limit: int = 25) -> dict:
    rows = _traces_query("""
        SELECT trace_id, name, agent, status, round(duration_ms, 1) AS duration_ms, start_ts, attributes
        FROM spans WHERE parent_span_id IS NULL ORDER BY start_ts DESC LIMIT ?""", [max(1, min(limit, 100))])
    out = []
    for r in rows:  # attributes is a JSON *string* column: parse in Python, no DuckDB JSON extension needed
        attrs = json.loads(r.pop("attributes") or "{}")
        out.append({**r, "start_ts": str(r["start_ts"]), "workflow": attrs.get("workflow"), "defenses": attrs.get("defenses")})
    return {"traces": out}


@router.get("/api/cc/traces/{trace_id}")
def trace_detail(trace_id: str) -> dict:
    if not all(c in "0123456789abcdef" for c in trace_id) or len(trace_id) != 32:
        raise HTTPException(400, "bad trace id")
    rows = _traces_query("""SELECT span_id, parent_span_id, name, tool, status, status_message, duration_ms, start_ts,
                                   cost_usd, events FROM spans WHERE trace_id = ? ORDER BY start_ts""", [trace_id])
    if not rows:
        raise HTTPException(404, "trace not found (spans are flushed every second)")
    t0 = rows[0]["start_ts"]
    return {"trace_id": trace_id, "spans": [{**r, "offset_ms": round((r["start_ts"] - t0).total_seconds() * 1000, 2),
                                             "start_ts": str(r["start_ts"]), "events": json.loads(r["events"] or "[]")} for r in rows]}


@router.get("/api/cc/funnel")
def funnel(request: Request) -> dict:
    rows = request.app.state.db.query("""
        SELECT l.level_id, l.version, l.created_by, l.target_words, l.word_count, l.time_limit_s, l.predicted_fail_rate,
               count(e.attempt_id) AS attempts, round(avg((e.event_name = 'level_fail')::INT), 3) AS fail_rate,
               sum((e.source = 'live')::INT) AS live_attempts
        FROM active_levels l LEFT JOIN events e ON e.level_id = l.level_id AND e.level_version = l.version
             AND e.event_name IN ('level_fail', 'level_complete')
        GROUP BY ALL ORDER BY l.level_id""")
    return {"levels": rows}
