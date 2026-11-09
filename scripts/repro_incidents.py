"""
================================================================================
BLOCK: Postmortem Reproductions (docs/THREE_WAYS_MY_AGENT_BROKE.md)
--------------------------------------------------------------------------------
Role & contracts
  Re-creates the failures from the postmortem against the current code and
  exports their spans to data/traces.duckdb, writing the trace ids to
  docs/incident_traces.json:
    incident 1  L3 copied-argument rule blocks every Soma follow-up when the
                engine's own identifiers are not declared trusted (the
                original bug), next to the fixed path.
    incident 2  ad-ops quarantine / rollback / escalation decisions on the
                72h incident timeline (fixed behaviour: exactly one quarantine,
                one rollback, escalations for external drops).
    incident 3  the span exporter under concurrent flushes + shutdown spill
                (fixed: 0 flush failures, nothing lost).

Key engineering mechanism
  Incident 1 is reproduced exactly by opening the intake workflow WITHOUT
  trusted_context. That is the precise code path before the fix, not a
  simulation of it.

Failure modes & 1-step mitigation
  * DB locked by the server -> exit 2 (stop the server first).
================================================================================
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import REPO_ROOT, get_settings  # noqa: E402
from core.db import DatabaseLockedError, TelemetryDB  # noqa: E402
from core.harness.telemetry import DuckDBSpanExporter, InMemoryExporter, Tracer  # noqa: E402


class Tee:
    def __init__(self, *e):
        self.e = e

    def export(self, s):
        for x in self.e:
            x.export(s)


def incident_1(tracer: Tracer, mem: InMemoryExporter) -> dict:
    from agents.soma_intake.synthetic import build_cases
    from agents.soma_intake.workflow import EventStore, IntakeEngine, normalize_email

    with tempfile.TemporaryDirectory() as tmp:
        store = EventStore(Path(tmp) / "s.sqlite")
        eng = IntakeEngine(store, tracer=tracer)
        case = build_cases(3)[0]
        cid = eng.handle_inbound(normalize_email(case.payloads[0]["payload"]["raw"]))     # fixed path runs here
        fixed_followups = len(store.outbound(cid))
        fields = eng.extraction(eng.state(cid)).gaps()[:3]
        with eng.runner.workflow("repro:incident1:before_fix") as wf:                     # NO trusted_context
            wf.call("extract_case", case_id=cid)
            rec = wf.call_record("send_followup", case_id=cid, fields=fields, activity_id=f"{cid}:followup:9")
            before_trace = wf.root.trace_id
        fixed_trace = next(s.trace_id for s in reversed(mem.spans) if s.name == "workflow.run"
                           and s.attributes.get("workflow", "").startswith(f"intake:{cid}"))
        return {"before_fix": {"trace_id": before_trace, "decision": f"{rec.decision.layer}:{rec.decision.rule}",
                               "detail": rec.decision.detail},
                "after_fix": {"trace_id": fixed_trace, "followups_sent": fixed_followups}}


def incident_2(db: TelemetryDB, tracer: Tracer, mem: InMemoryExporter) -> dict:
    from agents.ad_economy.bandit_agent import run_ops_scenario

    r = run_ops_scenario(db, guard=True, tracer=tracer, seed=5)
    pick = lambda tool: sorted({s.trace_id for s in mem.spans if s.name == "tool.call" and s.attributes.get("tool") == tool})
    return {"events": [(e["hour"], e["type"], e.get("network", e.get("why", ""))) for e in r["events"] if e["type"] != "floor_change"],
            "quarantine_traces": pick("quarantine_network"), "rollback_traces": pick("rollback_config"),
            "false_quarantines": [e for e in r["events"] if e["type"] == "quarantine" and e.get("network") != "admob"]}


def incident_3() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        exp = DuckDBSpanExporter(Path(tmp) / "t.duckdb", flush_interval_s=0.005, batch_size=16)
        tr = Tracer(exp)

        def produce():
            for _ in range(3000):
                with tr.span("x", new_trace=True):
                    pass

        ths = [threading.Thread(target=produce) for _ in range(2)]
        [t.start() for t in ths]
        for _ in range(50):
            exp.flush(force=True)
            time.sleep(0.002)
        [t.join() for t in ths]
        exp.close()
        import duckdb
        n = duckdb.connect(str(Path(tmp) / "t.duckdb"), read_only=True).execute("SELECT count(*) FROM spans").fetchone()[0]
        return {"spans_produced": 6000, "spans_in_db": n, "stats": exp.stats}


def main() -> int:
    s = get_settings()
    try:
        db = TelemetryDB(s.telemetry_db)
    except DatabaseLockedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    exporter = DuckDBSpanExporter(s.traces_db)
    mem = InMemoryExporter()
    tracer = Tracer(Tee(exporter, mem), service_name="postmortem-repro")
    out = {"incident_1": incident_1(tracer, mem), "incident_2": incident_2(db, tracer, mem), "incident_3": incident_3()}
    exporter.flush(force=True, retries=3)
    out["exporter_stats"] = exporter.stats
    (REPO_ROOT / "docs" / "incident_traces.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
