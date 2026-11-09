"""
================================================================================
BLOCK: Fault Drill CLI (same drills as the Control Center, from a terminal)
--------------------------------------------------------------------------------
Role & contracts
  Runs the four production fault drills in app/fault_drills.py against
  data/telemetry.duckdb with spans exported to data/traces.duckdb, prints
  every step, and writes docs/fault_drills.json (fault -> guardrail -> trace
  ids). docs/THREE_WAYS_MY_AGENT_BROKE.md cites those trace ids.

Key engineering mechanism
  Identical code path to the UI (no CLI-only shortcuts); the exporter is
  force-flushed before exit, so every trace id printed is queryable:
    SELECT * FROM spans WHERE trace_id = '<id>'   (data/traces.duckdb)

Failure modes & 1-step mitigation
  * Server running (DuckDB lock) -> exit 2 with the fix; use the Control
    Center instead.  Any drill NOT mitigated -> exit 1.
================================================================================
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import fault_drills as fd  # noqa: E402
from core.config import REPO_ROOT, get_settings  # noqa: E402
from core.db import DatabaseLockedError, TelemetryDB  # noqa: E402
from core.harness.telemetry import DuckDBSpanExporter, Tracer  # noqa: E402


def main() -> int:
    s = get_settings()
    try:
        db = TelemetryDB(s.telemetry_db)
    except DatabaseLockedError as e:
        print(f"ERROR: {e}\n(or open http://localhost:8000/control-center#faults while the server runs)", file=sys.stderr)
        return 2
    exporter = DuckDBSpanExporter(s.traces_db)
    tracer = Tracer(exporter, service_name="fault-drills")
    from agents.level_factory.critic_and_eval import LevelFactory

    factory = LevelFactory(db, tracer=tracer, seed=17)
    icon = {"fault": "FAULT", "guard": "GUARD", "ok": " OK  ", "bad": "IMPACT", "info": " info", "approval": "APPRV"}
    log = lambda st: print(f"  [{icon.get(st['status'], '?')}] {st['title']}" + (f"\n           {st['detail'][:220]}" if st.get("detail") else ""))
    drills = [("1 LLM hallucinated unsolvable level", lambda: fd.drill_hallucinated_level(db, factory, log)),
              ("2 Ad network goes dark", lambda: fd.drill_network_dark(db, tracer, log)),
              ("3 Peeking false-positive experiment", lambda: fd.drill_peeking(tracer, log)),
              ("4 Indirect prompt injection", lambda: fd.drill_prompt_injection(tracer, log))]
    results = []
    for title, fn in drills:
        print(f"\n=== FAULT {title} ===")
        t0 = time.perf_counter()
        r = fn()
        r["seconds"] = round(time.perf_counter() - t0, 2)
        print(f"  => {'MITIGATED' if r['blocked'] else 'NOT MITIGATED'}: {r['summary']}\n  => trace ids: {', '.join(r['trace_ids'])}")
        results.append({k: v for k, v in r.items() if k not in ("series", "rows", "runs", "events")})
    exporter.flush(force=True)
    print(f"span exporter: {exporter.stats}")
    (REPO_ROOT / "docs" / "fault_drills.json").write_text(json.dumps(results, indent=2, default=str))
    db.close()
    ok = all(r["blocked"] for r in results)
    print(f"\n{sum(r['blocked'] for r in results)}/4 faults mitigated automatically -> docs/fault_drills.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
