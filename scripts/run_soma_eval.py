"""
================================================================================
BLOCK: Soma Intake Evaluation Harness (50 submissions, end to end)
--------------------------------------------------------------------------------
Role & contracts
  Drives every synthetic case (agents/soma_intake/synthetic.py) through the
  REAL pipeline: provider payload -> normaliser -> IntakeEngine (harness
  tools, L1/L2/L3 guards, SQLite event history) -> customer follow-ups ->
  mock carrier portal (HTML form filled by PortalClient) or human handoff.
  Prints a Markdown scorecard and writes it to --out (default
  data/reports/soma_eval.md).

Key engineering mechanism
  * Each simulated DAY uses a brand-new IntakeEngine on the same SQLite
    file: every case survives process restarts across its Email -> SMS ->
    Voice thread by replaying history.
  * Crash injection: in 6 cases the process dies between activity_started
    and activity_completed. The next process's recover() must finish the
    activity exactly once (one portal submission per client_ref, no
    duplicate outbound message).
  * Metrics
      Extraction F1 (micro, all 24 ACORD fields vs ground truth)
      Portal-fill success = submitted AND the carrier stored the true value of
                            every required field, over cases that should submit
      Avg turns to completion = inbound + outbound messages, submitted cases
      Human-handoff precision / recall vs expected handoffs (high-risk + ambiguous)
      Replay determinism, crash recovery, confidence calibration

Failure modes & 1-step mitigation
  * Regression in any stage -> the corresponding metric drops; exit code 1 if
    the success thresholds (F1 >= 0.90, fill >= 0.90, handoff precision
    >= 0.90, exactly-once ok) are not met.
================================================================================
Usage: python scripts/run_soma_eval.py [--cases 50] [--out path.md] [--json path.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.soma_intake.extractor import CONFIDENCE_GATE, FIELDS, REQUIRED, SPEC  # noqa: E402
from agents.soma_intake.portal_automation import PortalClient, get_submission  # noqa: E402
from agents.soma_intake.synthetic import TWILIO_AUTH_TOKEN, Case, CustomerSimulator, build_cases  # noqa: E402
from agents.soma_intake.workflow import (  # noqa: E402
    TERMINAL, EventStore, IntakeEngine, fold, normalize_email, normalize_sms, normalize_voice,
)
from core.harness.telemetry import InMemoryExporter, Tracer  # noqa: E402

PORTAL_KEYS = {"applicant_name": "insured", "entity_type": "entity", "mailing_street": "street", "mailing_city": "city",
               "mailing_state": "state", "mailing_zip": "zip", "contact_name": "contact", "contact_email": "email",
               "business_description": "ops", "employees_full_time": "ft", "annual_revenue": "sales",
               "policy_effective_date": "eff", "lines_of_business": "lines", "gl_each_occurrence": "occ",
               "gl_general_aggregate": "agg"}
DAY_CHANNEL = {1: "email", 2: "sms", 3: "voice"}


class SimulatedCrash(RuntimeError):
    pass


def _norm(v):
    if v is None:
        return None
    if isinstance(v, list):
        return sorted(str(x).lower() for x in v) or None
    if isinstance(v, str):
        return v.strip().lower()
    return v


def ingest(eng: IntakeEngine, channel: str, payload: dict, when: datetime) -> str:
    if channel == "email":
        return eng.handle_inbound(normalize_email(payload["raw"]))
    if channel == "sms":
        return eng.handle_inbound(normalize_sms(payload["form"], payload["signature"], payload["url"], TWILIO_AUTH_TOKEN, when))
    return eng.handle_inbound(normalize_voice(payload))


def run_case(case: Case, store: EventStore, tracer: Tracer, crash_kind: str | None = None, max_days: int = 8) -> dict:
    sim = CustomerSimulator(case)
    base = datetime(2026, 11, 2, tzinfo=timezone.utc) + timedelta(days=case.case_no % 5)
    case_id, crashed, recovered, restarts = None, False, False, 0
    for day in range(1, max_days + 1):
        armed = {"on": crash_kind is not None and not crashed}

        def hook(cid: str, kind: str) -> None:
            if armed["on"] and kind == crash_kind:
                armed["on"] = False
                raise SimulatedCrash(f"process killed during {kind}")

        eng = IntakeEngine(store, tracer=tracer, crash_hook=hook)   # a fresh process every simulated day
        restarts += 1
        if crashed and not recovered:
            recovered = bool(eng.recover(now=(base + timedelta(days=day - 1, hours=8)).isoformat()))
        try:
            # 1) what the customer planned to send that day (always delivered: customer-initiated)
            for p in case.payloads:
                if p["day"] == day:
                    when = base + timedelta(days=day - 1, hours=10)
                    case_id = ingest(eng, p["channel"], p["payload"], when) or case_id
            # 2) the customer answers our latest open follow-up on that day's channel
            if case_id:
                st = eng.state(case_id)
                if st.status == "AWAITING_CUSTOMER" and st.last_asked:
                    ch = DAY_CHANNEL.get(day, "sms")
                    when = base + timedelta(days=day - 1, hours=17)
                    ingest(eng, ch, sim.reply(st.last_asked, ch, when), when)
            if case_id:
                eng.tick((base + timedelta(days=day, hours=-1)).isoformat())
        except SimulatedCrash:
            crashed = True
            continue
        if case_id and eng.state(case_id).status in TERMINAL and day >= max(p["day"] for p in case.payloads):
            break
    eng = IntakeEngine(store, tracer=tracer)
    if crashed and not recovered:
        recovered = bool(eng.recover())
    st = eng.state(case_id)
    return {"case": case, "case_id": case_id, "state": st, "extraction": eng.extraction(st), "crashed": crashed,
            "recovered": recovered, "restarts": restarts, "engine": eng}


def evaluate(n_cases: int = 50, seed: int = 2026, workdir: Path | None = None, style: str = "dev") -> dict:
    prev = os.environ.get("LIVEOPS_DATA_DIR")
    try:
        return _evaluate(n_cases, seed, workdir, style)
    finally:  # never leak the sandbox data dir into the caller's process
        if prev is None:
            os.environ.pop("LIVEOPS_DATA_DIR", None)
        else:
            os.environ["LIVEOPS_DATA_DIR"] = prev


def _evaluate(n_cases: int, seed: int, workdir: Path | None, style: str) -> dict:
    t0 = time.perf_counter()
    workdir = workdir or Path(tempfile.mkdtemp(prefix="soma_eval_"))
    os.environ["LIVEOPS_DATA_DIR"] = str(workdir)          # portal + event store live in the eval sandbox
    store = EventStore(workdir / "soma_intake.sqlite")
    mem = InMemoryExporter()
    tracer = Tracer(mem, service_name="soma-eval")
    cases = build_cases(n_cases, seed, style=style)
    crash_plan = {c.case_no: ("portal" if i % 2 == 0 else "followup")
                  for i, c in enumerate([c for c in cases if not c.expect_handoff][:6])}
    results = [run_case(c, store, tracer, crash_plan.get(c.case_no)) for c in cases]

    tp = fp = fn = 0
    per_field = {f.key: {"correct": 0, "total": 0} for f in FIELDS}
    calib = {"hi": [0, 0], "lo": [0, 0]}
    for r in results:
        truth, ext = r["case"].truth, r["extraction"]
        for f in FIELDS:
            t, p = _norm(truth.get(f.key)), _norm(ext.value(f.key))
            if t is None and p is None:
                continue
            per_field[f.key]["total"] += t is not None
            if t is not None and p == t:
                tp += 1
                per_field[f.key]["correct"] += 1
            else:
                fn += t is not None
                fp += p is not None
            if p is not None:
                bucket = "hi" if ext.fields[f.key].confidence >= CONFIDENCE_GATE else "lo"
                calib[bucket][0] += p == t
                calib[bucket][1] += 1
    prec, rec = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    should_submit = [r for r in results if not r["case"].expect_handoff]
    fills, fill_fail = 0, []
    for r in should_submit:
        st = r["state"]
        ok = st.status == "SUBMITTED" and st.submission_id
        if ok:
            sub = get_submission(st.submission_id) or {}
            bad = {k: (sub.get(pk), r["case"].truth[k]) for k, pk in PORTAL_KEYS.items()
                   if _norm(str(sub.get(pk)) if not isinstance(sub.get(pk), list) else sub.get(pk)) !=
                   _norm(str(r["case"].truth[k]) if not isinstance(r["case"].truth[k], list) else r["case"].truth[k])}
            ok = not bad
            if bad:
                fill_fail.append({"case": r["case"].case_no, "scenario": r["case"].scenario, "mismatch": bad})
        else:
            fill_fail.append({"case": r["case"].case_no, "scenario": r["case"].scenario, "status": st.status,
                              "reasons": st.handoff_reasons, "gaps": r["extraction"].gaps()})
        fills += bool(ok)
    submitted = [r for r in results if r["state"].status == "SUBMITTED"]
    turns = [r["state"].turns_in + r["state"].turns_out for r in submitted]
    pred_h = [r for r in results if r["state"].status == "HUMAN_HANDOFF"]
    h_tp = sum(r["case"].expect_handoff for r in pred_h)
    exp_h = sum(r["case"].expect_handoff for r in results)

    # durability: replay determinism + exactly-once for crashed cases
    replay_ok = all(fold(r["case_id"], store.events(r["case_id"])).__dict__.keys() and
                    fold(r["case_id"], store.events(r["case_id"])).status == r["state"].status for r in results)
    crash_rows = []
    import sqlite3
    pcon = sqlite3.connect(workdir / "carrier_portal.sqlite")
    for r in results:
        if not r["crashed"]:
            continue
        cid = r["case_id"]
        subs = pcon.execute("SELECT count(*) FROM submissions WHERE client_ref LIKE ?", [cid + ":portal:%"]).fetchone()[0]
        outb = len(store.outbound(cid))
        fu_acts = sum(1 for a in r["state"].activities.values() if a["kind"] == "followup")
        crash_rows.append({"case": r["case"].case_no, "crashed_in": crash_plan[r["case"].case_no], "recovered": r["recovered"],
                           "final": r["state"].status, "portal_submissions": subs, "outbound_msgs": outb,
                           "followup_activities": fu_acts, "exactly_once": subs <= 1 and outb == fu_acts})
    pcon.close()
    spans = mem.spans
    return {
        "n": len(results), "seconds": round(time.perf_counter() - t0, 1), "style": style,
        "extraction": {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
                       "per_field": {k: v for k, v in per_field.items() if v["total"]}},
        "calibration": {"conf>=0.80_accuracy": round(calib["hi"][0] / calib["hi"][1], 4) if calib["hi"][1] else None,
                        "conf>=0.80_n": calib["hi"][1],
                        "conf<0.80_accuracy": round(calib["lo"][0] / calib["lo"][1], 4) if calib["lo"][1] else None,
                        "conf<0.80_n": calib["lo"][1]},
        "portal_fill": {"should_submit": len(should_submit), "succeeded": fills,
                        "rate": round(fills / len(should_submit), 4) if should_submit else 0.0, "failures": fill_fail},
        "turns": {"avg_to_completion": round(sum(turns) / len(turns), 2) if turns else None, "min": min(turns, default=None),
                  "max": max(turns, default=None)},
        "handoff": {"predicted": len(pred_h), "expected": exp_h, "true_positive": h_tp,
                    "precision": round(h_tp / len(pred_h), 4) if pred_h else 0.0,
                    "recall": round(h_tp / exp_h, 4) if exp_h else 0.0,
                    "reasons": {r["case"].case_no: (r["case"].scenario, r["state"].handoff_reasons) for r in pred_h}},
        "durability": {"process_restarts": sum(r["restarts"] for r in results), "replay_deterministic": replay_ok,
                       "crash_cases": crash_rows},
        "scenarios": {s: {"n": sum(r["case"].scenario == s for r in results),
                          "submitted": sum(r["case"].scenario == s and r["state"].status == "SUBMITTED" for r in results),
                          "handoff": sum(r["case"].scenario == s and r["state"].status == "HUMAN_HANDOFF" for r in results)}
                      for s in ("clean", "needs_followup", "conflict_corrected", "high_risk", "ambiguous")},
        "harness": {"workflows": sum(s.name == "workflow.run" for s in spans), "tool_calls": sum(s.name == "tool.call" for s in spans),
                    "guard_blocks": sum(bool(s.attributes.get("blocked")) for s in spans if s.name == "tool.call")},
        "example": results[0]["engine"].case_view(results[0]["case_id"]) if results else None,
    }


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows])


def render(r: dict) -> str:
    e, p, h, d = r["extraction"], r["portal_fill"], r["handoff"], r["durability"]
    md = ["# Soma Intake Evaluation", "", f"_{r['n']} synthetic submissions, {r['seconds']}s, deterministic extractor (no LLM)._", "",
          "## Headline metrics", "",
          table(["metric", "value"], [
              ["Extraction F1 (micro, 24 ACORD fields)", f"{100 * e['f1']:.1f}%  (P {100 * e['precision']:.1f}% / R {100 * e['recall']:.1f}%)"],
              ["Portal-fill success rate", f"{100 * p['rate']:.1f}%  ({p['succeeded']}/{p['should_submit']} cases that should submit)"],
              ["Average turns to completion", f"{r['turns']['avg_to_completion']} (min {r['turns']['min']}, max {r['turns']['max']})"],
              ["Human-handoff precision", f"{100 * h['precision']:.1f}%  ({h['true_positive']}/{h['predicted']})"],
              ["Human-handoff recall", f"{100 * h['recall']:.1f}%  ({h['true_positive']}/{h['expected']})"],
              ["Process restarts survived", d["process_restarts"]],
              ["Replay determinism (fold(history) == live state)", d["replay_deterministic"]],
              ["Accuracy when confidence >= 0.80", f"{r['calibration']['conf>=0.80_accuracy']} (n={r['calibration']['conf>=0.80_n']})"],
              ["Accuracy when confidence < 0.80", f"{r['calibration']['conf<0.80_accuracy']} (n={r['calibration']['conf<0.80_n']})"],
          ]), "", "## Outcomes by scenario", "",
          table(["scenario", "cases", "submitted", "handed off"], [[k, v["n"], v["submitted"], v["handoff"]] for k, v in r["scenarios"].items()]),
          "", "## Crash recovery (process killed between activity_started and activity_completed)", "",
          table(["case", "crashed in", "recovered", "final status", "portal submissions", "outbound msgs", "follow-up activities", "exactly once"],
                [[c["case"], c["crashed_in"], c["recovered"], c["final"], c["portal_submissions"], c["outbound_msgs"],
                  c["followup_activities"], c["exactly_once"]] for c in d["crash_cases"]]),
          "", "## Per-field extraction accuracy", "",
          table(["field", "ACORD", "required", "correct / present"],
                [[k, SPEC[k].form, SPEC[k].required, f"{v['correct']}/{v['total']}"] for k, v in e["per_field"].items()]),
          "", "## Handoffs", "",
          table(["case", "scenario", "reasons"], [[k, s, "; ".join(rs)[:110]] for k, (s, rs) in h["reasons"].items()]), ""]
    if p["failures"]:
        md += ["## Portal-fill failures", ""] + [f"- case {f['case']} ({f['scenario']}): " +
                                                 json.dumps({k: v for k, v in f.items() if k not in ('case', 'scenario')}, default=str)[:300]
                                                 for f in p["failures"]] + [""]
    md += [f"Harness: {r['harness']['workflows']} workflows, {r['harness']['tool_calls']} tool calls, "
           f"{r['harness']['guard_blocks']} guard blocks."]
    return "\n".join(md)


def render_generalisation(dev: dict, ho: dict) -> str:
    row = lambda name, f: [name, f(dev), f(ho)]
    pct = lambda x: f"{100 * x:.1f}%"
    md = ["", "## Generalisation check: held-out phrasing", "",
          "Same 50 businesses and scenarios, but every customer statement uses wording that was NEVER used while "
          "writing the extraction rules (e.g. 'per claim', 'yearly turnover', 'zero losses', 'replace shingles'). "
          "Rules were not tuned on this set.", "",
          table(["metric", "development phrasing", "held-out phrasing"], [
              row("Extraction F1", lambda r: pct(r["extraction"]["f1"])),
              row("Extraction precision / recall", lambda r: f"{pct(r['extraction']['precision'])} / {pct(r['extraction']['recall'])}"),
              row("Portal-fill success", lambda r: f"{pct(r['portal_fill']['rate'])} ({r['portal_fill']['succeeded']}/{r['portal_fill']['should_submit']})"),
              row("Avg turns to completion", lambda r: r["turns"]["avg_to_completion"]),
              row("Handoff precision", lambda r: f"{pct(r['handoff']['precision'])} ({r['handoff']['true_positive']}/{r['handoff']['predicted']})"),
              row("Handoff recall", lambda r: f"{pct(r['handoff']['recall'])} ({r['handoff']['true_positive']}/{r['handoff']['expected']})"),
              row("Accuracy at confidence >= 0.80", lambda r: r["calibration"]["conf>=0.80_accuracy"]),
          ]), "", "Held-out per-field accuracy (fields below 100%):", ""]
    weak = [[k, f"{v['correct']}/{v['total']}"] for k, v in ho["extraction"]["per_field"].items() if v["correct"] < v["total"]]
    md += [table(["field", "correct / present"], weak) if weak else "_none_", ""]
    missed = [f"case {f['case']} ({f['scenario']}): " + (f"status {f.get('status')}, reasons {f.get('reasons')}" if "status" in f
              else f"wrong values {list(f.get('mismatch', {}))}") for f in ho["portal_fill"]["failures"]]
    md += ["Held-out cases that should have been submitted but were not (or with wrong values):", ""] + \
          [f"- {m[:200]}" for m in missed] + [""]
    should = {k for k, (s, _) in ho["handoff"]["reasons"].items()}
    md += [f"Held-out high-risk / ambiguous cases handed off: {ho['handoff']['true_positive']}/{ho['handoff']['expected']}. "
           "Any miss here is a high-risk business that reached the carrier portal without underwriter review, "
           "which is the most important failure mode to close before production (see the note at the top).", ""]
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description="Soma intake 50-submission evaluation")
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    from core.config import get_settings

    real_data = get_settings().data_dir
    r = evaluate(a.cases, a.seed)
    ho = evaluate(a.cases, a.seed, style="heldout")
    md = render(r) + "\n" + render_generalisation(r, ho)
    print(md)
    out = a.out or real_data / "reports" / "soma_eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    if a.json:
        a.json.write_text(json.dumps({k: v for k, v in r.items() if k != "example"}, indent=2, default=str))
    ok = (r["extraction"]["f1"] >= 0.90 and r["portal_fill"]["rate"] >= 0.90 and r["handoff"]["precision"] >= 0.90
          and all(c["exactly_once"] and c["recovered"] for c in r["durability"]["crash_cases"]) and r["durability"]["replay_deterministic"])
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
