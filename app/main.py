"""
================================================================================
BLOCK: FastAPI Application Entry Point (game + telemetry + harness endpoints)
--------------------------------------------------------------------------------
Role & contracts
  `uvicorn app.main:app` - serves the playable game (/game), the game API
  (/api/game/*), telemetry read APIs (/api/telemetry/*), harness metrics and
  the conformance suite (/api/harness/*), admin reseed, the Insurance intake
  webhooks (/intake/*) and the mock carrier portal (/mock-carrier-portal). Phase 4's Control
  Center mounts onto this same app.

Key engineering mechanism
  * Lifespan opens the single long-lived DuckDB writer (core.db.TelemetryDB),
    loads the lexicon once, and SELF-SEEDS when the database is empty, so a
    fresh clone is playable with zero manual steps.
  * All DB access goes through TelemetryDB's lock (DuckDB connections are not
    safe for concurrent use from FastAPI's threadpool).

Failure modes & 1-step mitigation
  * Another process holds the DuckDB lock (e.g. the seeder) -> startup aborts
    with DatabaseLockedError's one-line fix instead of a stack trace.
  * Need fresh data while serving -> POST /api/admin/reseed reseeds through
    the server's own writer (no lock fight).
================================================================================
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse

from agents.insurance_intake.portal_automation import portal_router
from agents.insurance_intake.workflow import router as intake_router
from app.dashboard import router as cc_router
from app.game_api import GameState, router as game_router
from core.config import get_settings
from core.db import DatabaseLockedError, TelemetryDB
from core.wordlist import load_lexicon

log = logging.getLogger("liveops")
STATIC = Path(__file__).parent / "static"


def _seed(db: TelemetryDB) -> dict:
    from scripts.seed_telemetry import check_calibration, seed_database, wipe

    s = get_settings()
    with db.cursor() as con:
        con.execute("BEGIN")
        try:
            wipe(con)
            report = seed_database(con, csv_path=s.cookie_cats_csv)
            errs = check_calibration(report)
            if errs:
                raise RuntimeError("; ".join(errs))
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    return report


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    try:
        db = TelemetryDB(s.telemetry_db)
    except DatabaseLockedError as e:
        log.error("%s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
    if db.is_empty():
        print("telemetry.duckdb is empty - seeding Cookie Cats + 2,500-player cohort (~10 s) ...", flush=True)
        r = _seed(db)
        print(f"seeded {r['events']:,} events in {r['seconds']}s ({r['provenance']})", flush=True)
    app.state.db = db
    app.state.game = GameState(db, load_lexicon())
    if hasattr(app.state, "cc"):   # Control Center handlers bind to the DB: never reuse ones from a previous lifespan
        del app.state.cc
    yield
    db.close()


app = FastAPI(title="LiveOps Agent Platform", version="0.1.0", lifespan=lifespan)
app.include_router(game_router)
app.include_router(portal_router)   # Phase 3: mock carrier portal (/mock-carrier-portal)
app.include_router(intake_router)     # Phase 3: Insurance intake webhooks + case API (/intake/*)
app.include_router(cc_router)       # Phase 4: Evaluator Control Center (/control-center, /api/cc/*)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/control-center")


@app.get("/game", include_in_schema=False)
def game_page():
    return FileResponse(STATIC / "game.html", media_type="text/html")


@app.get("/api/health")
def health(request: Request) -> dict:
    n = request.app.state.db.query("SELECT count(*) AS n FROM events")[0]["n"]
    return {"status": "ok", "events": n}


@app.get("/api/llm/status")
def llm_provider_status() -> dict:
    """Which model provider is configured, whether it is answering, and what to do when it is not (bad key,
    HTTP 429 insufficient_quota, outage). Agents keep running on the deterministic fallback either way."""
    from core.harness.llm import llm_status

    return llm_status()


@app.get("/api/telemetry/recent")
def recent(request: Request, limit: int = 25, source: str = "live") -> dict:
    rows = request.app.state.db.query(
        "SELECT ts, event_name, level_id, level_version, word, revenue_usd, network, placement, user_id "
        "FROM events WHERE source = ? ORDER BY ts DESC LIMIT ?", [source, max(1, min(limit, 200))])
    return {"events": rows}


@app.get("/api/telemetry/funnel")
def funnel(request: Request) -> dict:
    rows = request.app.state.db.query("""
        SELECT level_id, source, sum((event_name='level_start')::INT) AS starts,
               sum((event_name='level_complete')::INT) AS completes, sum((event_name='level_fail')::INT) AS fails,
               round(sum((event_name='level_fail')::INT) / nullif(sum((event_name IN ('level_fail','level_complete'))::INT), 0), 3) AS fail_rate
        FROM events WHERE level_id IS NOT NULL GROUP BY ALL ORDER BY level_id, source""")
    return {"funnel": rows}


@app.get("/api/telemetry/retention")
def retention(request: Request) -> dict:
    rows = request.app.state.db.query("""
        SELECT version AS ab_group, count(*) AS players, round(avg(retention_1::INT), 6) AS d1,
               round(avg(retention_7::INT), 6) AS d7, any_value(provenance) AS provenance
        FROM cookie_cats GROUP BY 1 ORDER BY 1""")
    return {"cookie_cats": rows}


@app.get("/api/harness/metrics")
def harness_metrics(since_minutes: float | None = None) -> dict:
    from core.harness.telemetry import agent_metrics

    try:
        return {"metrics": agent_metrics(get_settings().traces_db, since_minutes)}
    except Exception as e:  # traces DB briefly locked by an eval run
        return {"metrics": [], "error": f"{type(e).__name__}: {str(e)[:120]}"}


@app.post("/api/harness/conformance")
def conformance() -> dict:
    from core.harness.run_redteam_eval import run_suite

    r = run_suite()
    return {"counts": r["counts"], "summary": r["summary"], "seconds": r["seconds"]}


@app.post("/api/admin/reseed")
def reseed(request: Request) -> dict:
    r = _seed(request.app.state.db)
    request.app.state.game.reset()
    return {"events": r["events"], "seconds": r["seconds"], "provenance": r["provenance"]}
