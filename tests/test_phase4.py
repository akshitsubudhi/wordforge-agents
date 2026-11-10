"""
================================================================================
BLOCK: Phase 4 Test Suite (Control Center, fault drills, telemetry durability)
--------------------------------------------------------------------------------
Role & contracts
  Drives the Evaluator Control Center exactly as the browser does
  (submit job -> poll steps -> answer approvals) against the REAL app on a
  temp-seeded DuckDB, and checks the exporter fix from the postmortem.

Key engineering mechanism
  wait_job() polls /api/cc/jobs/{id} and can answer human-approval steps,
  so the approval gate is tested with an actual approve/deny round trip.

Failure modes & 1-step mitigation
  * Slow CI -> generous job timeouts; drills themselves take ~1 s each.
================================================================================
"""
from __future__ import annotations

import os
import time

import duckdb
import pytest

from core.db import open_rw
from core.harness.telemetry import DuckDBSpanExporter, Tracer


@pytest.fixture(scope="module")
def client(tmp_path_factory, seeded_copy):
    from fastapi.testclient import TestClient

    from scripts.seed_telemetry import seed_database

    d = tmp_path_factory.mktemp("phase4")
    prev = os.environ.get("LIVEOPS_DATA_DIR")
    os.environ["LIVEOPS_DATA_DIR"] = str(d)
    import core.harness.telemetry as tel
    tel._default = None                       # traces for this module go to the temp dir
    seeded_copy(d, n_players=400, seed=4)
    from app.main import app

    with TestClient(app) as c:
        yield c
    tel._default = None
    if prev is None:
        os.environ.pop("LIVEOPS_DATA_DIR", None)
    else:
        os.environ["LIVEOPS_DATA_DIR"] = prev


def wait_job(client, kind, params=None, answer=None, timeout=240):
    """Submit a job, stream its steps; `answer(step)` returns 'approve'/'deny' for approval steps."""
    jid = client.post("/api/cc/jobs", json={"kind": kind, "params": params or {}}).json()["job_id"]
    steps, since, t0 = [], 0, time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/cc/jobs/{jid}?since={since}").json()
        for s in j["steps"]:
            steps.append(s)
            if s["status"] == "approval" and answer:
                client.post(f"/api/cc/approvals/{s['approval_id']}?decision={answer(s)}", json={})
        since = j["next"]
        if j["status"] in ("done", "error"):
            assert j["status"] == "done", j["error"]
            return j["result"], steps
        time.sleep(0.2)
    raise TimeoutError(kind)


def test_root_lands_on_control_center(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/control-center"
    page = client.get("/control-center").text
    assert "Simulate production failure" in page and "Generate, verify &amp; push level to live game" in page
    o = client.get("/api/cc/overview").json()
    assert o["counts"]["active_levels"] == 12 and "headline" in (o["scorecard"] or {"headline": []})


@pytest.mark.parametrize("kind,guard_fragment", [
    ("fault_hallucinated_level", "solver.py"),
    ("fault_network_dark", "quarantined"),
    ("fault_peeking", "mSPRT"),
    ("fault_prompt_injection", "redacted"),
])
def test_fault_drills_are_mitigated(client, kind, guard_fragment):
    res, steps = wait_job(client, kind)
    assert res["blocked"] and res["trace_ids"]
    assert any(s["status"] == "fault" for s in steps) and any(s["status"] == "guard" for s in steps)
    assert any(guard_fragment in s["title"] for s in steps)
    detail = client.get(f"/api/cc/traces/{res['trace_ids'][0]}")
    assert detail.status_code == 200 and detail.json()["spans"]       # the cited trace really exists


def test_unsolvable_level_never_reaches_the_game(client):
    before = {l["level_id"] for l in client.get("/api/game/levels").json()["levels"]}
    wait_job(client, "fault_hallucinated_level")
    after = {l["level_id"] for l in client.get("/api/game/levels").json()["levels"]}
    assert before == after


def test_factory_push_goes_live(client):
    res, steps = wait_job(client, "factory_push", {"goal": 0.3, "size": 4, "time_limit_s": 100, "candidates": 24})
    pub = res["published"]
    assert pub and any("LIVE" in s["title"] for s in steps)
    levels = {l["level_id"]: l for l in client.get("/api/game/levels").json()["levels"]}
    assert levels[pub["level_id"]]["daily"] and levels[pub["level_id"]]["created_by"] == "level_factory"
    s = client.post("/api/game/session", json={}).json()
    start = client.post("/api/game/level/start", json={"session_id": s["session_id"], "level_id": pub["level_id"]}).json()
    assert "/".join(start["grid"]) == pub["grid"]


def test_human_gate_deny_keeps_flag_off(client):
    res, steps = wait_job(client, "copilot_experiment", {"scenario": "harmful", "human_approval": True}, answer=lambda s: "deny")
    assert res["outcome"] == "denied"
    assert any("Human DENIED" in s["title"] for s in steps)
    flags = client.get("/api/cc/overview").json()["counts"]["active_flags"]
    assert flags == 0


def test_human_gate_approve_then_auto_rollback(client):
    res, steps = wait_job(client, "copilot_experiment", {"scenario": "harmful", "human_approval": True, "daily_users": 3000},
                          answer=lambda s: "approve")
    assert res["action"]["status"] == "rolled_back"
    assert any("AUTO-ROLLBACK" in s["title"] for s in steps)


def test_scan_intake_ads_conformance_and_reset(client):
    scan, _ = wait_job(client, "copilot_scan")
    assert scan["scan"]["anomalies"][0]["level_id"] == 7
    intake, _ = wait_job(client, "intake_case", {"scenario": "high_risk"})
    assert intake["status"] == "HUMAN_HANDOFF" and intake["expect_handoff"]
    mech, _ = wait_job(client, "ad_mechanisms", {"impressions": 4000})
    by = {r["mechanism"]: r for r in mech["rows"]}
    assert by["hb_second"]["ecpm"] > by["waterfall_strategic"]["ecpm"]
    conf, _ = wait_job(client, "conformance")
    assert conf["summary"]["all_on"]["blocked"] == 25 and conf["summary"]["all_on"]["false_alarms"] == 0
    assert client.get("/api/cc/metrics").json()["metrics"]
    assert client.get("/api/cc/traces/recent").json()["traces"]
    wait_job(client, "reset_demo")
    levels = client.get("/api/game/levels").json()["levels"]
    assert len(levels) == 12 and all(l["created_by"] == "seed" for l in levels)


def test_approval_endpoint_rejects_unknown_ids(client):
    assert client.post("/api/cc/approvals/nope?decision=approve", json={}).status_code == 404
    assert client.post("/api/cc/jobs", json={"kind": "drop_tables"}).status_code == 404
    assert client.get("/api/cc/traces/not-a-trace").status_code == 400


# ============================================================================ postmortem incident #3 regression
def test_exporter_concurrent_flushes_never_conflict(tmp_path):
    import threading

    exp = DuckDBSpanExporter(tmp_path / "t.duckdb", flush_interval_s=0.005, batch_size=16)
    tr = Tracer(exp)

    def produce():
        for _ in range(1500):
            with tr.span("x", new_trace=True):
                pass

    ths = [threading.Thread(target=produce) for _ in range(2)]
    [t.start() for t in ths]
    for _ in range(30):
        exp.flush(force=True)
        time.sleep(0.002)
    [t.join() for t in ths]
    exp.close()
    n = duckdb.connect(str(tmp_path / "t.duckdb"), read_only=True).execute("SELECT count(*) FROM spans").fetchone()[0]
    assert n == 3000 and exp.stats["flush_failures"] == 0


def test_exporter_spills_at_shutdown_and_reingests(tmp_path):
    bad = tmp_path / "dir_not_db"
    bad.mkdir()
    e = DuckDBSpanExporter(bad, background=False)
    tr = Tracer(e)
    for _ in range(5):
        with tr.span("y", new_trace=True):
            pass
    e.close()
    assert e.stats["spilled"] == 5 and e.spool_path.exists()
    good = tmp_path / "good.duckdb"
    e.spool_path.rename(good.with_suffix(".spool.jsonl"))
    e2 = DuckDBSpanExporter(good, background=False)
    with Tracer(e2).span("z", new_trace=True):
        pass
    e2.flush(force=True)
    assert e2.stats["unspooled"] == 5
    assert duckdb.connect(str(good), read_only=True).execute("SELECT count(*) FROM spans").fetchone()[0] == 6
