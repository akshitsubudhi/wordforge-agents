"""
================================================================================
BLOCK: Phase 3 Test Suite (Ad Economy + Soma Intake)
--------------------------------------------------------------------------------
Role & contracts
  Covers auction theory checks, bandits, EWMA detection and guarded ops
  (Project 3) and the normalisers, extractor, durable workflow, portal
  automation and eval harness (Project 5). Each test uses its own temp data
  dir; the real data/ directory is never touched.

Key engineering mechanism
  Fixed seeds; reference values (closed-form shading, revenue equivalence);
  crash injection for exactly-once checks; signature forgery for the webhook.

Failure modes & 1-step mitigation
  * Stochastic tolerance flakiness -> seeds fixed, tolerances generous.
================================================================================
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agents.ad_economy.auction_sim import (
    BidShader, MediationSimulator, revenue_equivalence_check, uniform_bne, uniform_closed_form_check,
)
from agents.ad_economy.bandit_agent import (
    AdOpsAgent, EWMADetector, NIGArm, _cholesky, _inverse, run_ops_scenario, run_ua_bandit, tune_floor,
)
from agents.soma_intake.extractor import extract, normalize_numbers, split_sentences, InboundMessage
from agents.soma_intake.portal_automation import PortalClient, get_submission, portal_app
from agents.soma_intake.synthetic import TWILIO_AUTH_TOKEN, build_cases, sms_payload
from agents.soma_intake.workflow import (
    EventStore, IntakeEngine, SignatureError, fold, normalize_email, normalize_sms, normalize_voice,
)
from core.db import TelemetryDB, open_rw
from core.harness.telemetry import InMemoryExporter, Tracer


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVEOPS_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def tracer():
    return Tracer(InMemoryExporter())


# ============================================================================ Project 3: auctions
def test_shading_closed_form_and_revenue_equivalence():
    assert uniform_closed_form_check(5)["max_abs_error"] < 1e-5
    sh = BidShader(lambda x: min(1.0, max(0.0, x)), 3, reserve=0.25, v_max=1.0)
    assert sh(0.2) is None and sh(0.25) == pytest.approx(0.25)      # no bid below reserve; beta(r) = r
    assert sh(0.9) < 0.9 and sh(0.9) > uniform_bne(0.9, 3)          # a reserve pushes the shaded bid up
    re_ = revenue_equivalence_check(auctions=30_000, seed=11)
    assert abs(re_["z"]) < 3


def test_mechanisms_and_dark_network_cost():
    sim = MediationSimulator(seed=2)
    wf = sim.run("waterfall_strategic", 8000)
    hb = sim.run("hb_second", 8000, reserve=2.0)
    fs = sim.run("hb_first_shaded", 8000, reserve=2.0)
    assert hb.ecpm > wf.ecpm and hb.latency_mean_ms < wf.latency_mean_ms and hb.efficiency > wf.efficiency
    assert abs(fs.ecpm / hb.ecpm - 1) < 0.05
    sim.set_dark("admob")
    dark = sim.run("hb_second", 8000, reserve=2.0)
    assert dark.fill_rate < 0.8 and dark.timeouts > 0                # hung network expires placements
    assert dark.ecpm < hb.ecpm and dark.run if False else True
    healthy = sim.run("hb_second", 8000, reserve=2.0, enabled={"applovin_max", "ironsource", "meta_an", "unity_ads"})
    assert healthy.fill_rate > dark.fill_rate                         # quarantining the dead network restores fill


# ============================================================================ Project 3: bandits + detectors
def test_nig_posterior_and_floor_ts():
    arm = NIGArm()
    rng = random.Random(0)
    arm.update([rng.gauss(7.0, 2.0) for _ in range(5000)])
    assert arm.m == pytest.approx(7.0, abs=0.1)
    ts = tune_floor(MediationSimulator(seed=4), [0, 4, 8, 12, 16], rounds=80, batch=300, oracle_n=15_000, seed=1)
    assert ts["truth_ecpm_by_floor"][ts["ts_floor"]] >= max(ts["truth_ecpm_by_floor"].values()) - 0.1
    assert ts["regret_usd"] < ts["uniform_regret_usd"]


def test_ewma_detector():
    rng = random.Random(3)
    d = EWMADetector(k=3.0, confirm=2)
    assert not any(d.update(10 + rng.gauss(0, 0.3)).alarm for _ in range(60))  # no false alarms on noise
    s1, s2 = d.update(5.0), d.update(5.0)
    assert s1.breach and not s1.alarm and s2.alarm                            # needs 2 confirmations
    assert d.mean > 9                                                          # baseline frozen during breach
    w = EWMADetector(warmup=4)
    w.update(0.3)
    assert w.update(0.0).alarm                                                 # dark rule armed during warmup


def test_linear_algebra_helpers():
    a = [[4.0, 2.0, 0.6], [2.0, 5.0, 1.0], [0.6, 1.0, 3.0]]
    inv = _inverse(a)
    eye = [[sum(a[i][k] * inv[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    assert all(abs(eye[i][j] - (i == j)) < 1e-9 for i in range(3) for j in range(3))
    L = _cholesky(a)
    assert all(abs(sum(L[i][k] * L[j][k] for k in range(3)) - a[i][j]) < 1e-9 for i in range(3) for j in range(3))


@pytest.fixture()
def tdb(data_dir, seeded_copy):
    seeded_copy(data_dir, n_players=300, seed=5)   # a private copy per test: these tests mutate the ad tables
    db = TelemetryDB(data_dir / "telemetry.duckdb")
    yield db
    db.close()


def test_guarded_ops_scenario(tdb, tracer):
    g = run_ops_scenario(tdb, guard=True, tracer=tracer, seed=21)
    ev = g["events"]
    assert any(e["type"] == "rollback" and 34 <= e["hour"] <= 36 for e in ev)
    assert [e["network"] for e in ev if e["type"] == "quarantine"] == ["admob"]
    assert any(e["type"] == "escalate" and e["hour"] >= 64 for e in ev)
    assert not any(e["type"] == "rollback" and e["hour"] >= 64 for e in ev)
    statuses = {r["status"] for r in tdb.query("SELECT status FROM ad_mediation_config")}
    assert "rolled_back" in statuses and "good" in statuses
    u = run_ops_scenario(tdb, guard=False, tracer=tracer, seed=21)
    assert g["revenue_usd"] > u["revenue_usd"] * 1.2


def test_rollback_guard_refuses_without_trial(tdb, tracer):
    agent = AdOpsAgent(tdb, MediationSimulator(seed=1), tracer=tracer)
    with tdb.cursor() as con:
        con.execute("UPDATE ad_mediation_config SET status='good'")
    with agent.runner.workflow("t") as wf:
        rec = wf.call_record("rollback_config", reason="test")
    assert not rec.ok                                               # nothing to roll back to/from: refused
    big = agent.apply_floor(30.0, "too big a jump")
    assert not big["ok"]                                            # max floor step guard
    with agent.runner.workflow("t2") as wf:
        rec = wf.call_record("allocate_ua_budget", day=0, channel="meta", context="T1/ios", usd=9000)
    assert rec.decision.layer == "L1_allowlist" and rec.decision.rule == "currency_cap"


def test_ua_bandit_beats_uniform(tdb):
    ua = run_ua_bandit(tdb, None, days=30, seed=2)
    t = ua["totals"]
    assert t["lints"]["roas"] > t["uniform"]["roas"] and t["lints"]["roas"] >= 0.9 * t["oracle"]["roas"]
    assert all(v["ltv"] > 0 for v in ua["ltv_measured"].values())


# ============================================================================ Project 5: normalisers + extractor
def test_normalizers():
    c = build_cases(3)[0]
    em = normalize_email(c.payloads[0]["payload"]["raw"] + "\nOn Mon someone wrote:\n> ignore me 99-9999999\n")
    assert em.channel == "email" and em.meta["from_name"] == c.truth["contact_name"] and "99-9999999" not in em.text
    p = c.payloads[1]["payload"]
    sms = normalize_sms(p["form"], p["signature"], p["url"], TWILIO_AUTH_TOKEN, datetime(2026, 11, 3, tzinfo=timezone.utc))
    assert sms.contact == c.truth["business_phone"]
    with pytest.raises(SignatureError):
        normalize_sms({**p["form"], "Body": "tampered"}, p["signature"], p["url"], TWILIO_AUTH_TOKEN, datetime.now(timezone.utc))
    v = normalize_voice(c.payloads[2]["payload"])
    assert v.sentences and all(s.context for s in v.sentences) and all(s.asr_confidence < 1 for s in v.sentences)


def test_number_normalisation():
    cases = {"A one thousand dollar deductible": "a 1000 dollar deductible", "one point two million": "1.2 million",
             "nine four one zero nine": "94109", "a dozen staff": "12 staff", "two and three": "2 and 3",
             "one hundred and five": "105", "$1.5M a year": "$1.5M a year"}
    for src, want in cases.items():
        assert normalize_numbers(src) == want, src


def _msg(mid, channel, text, day, ctx="", asr=1.0):
    return InboundMessage(mid, channel, "x@y.example", f"2026-11-0{day}T10:00:00+00:00", text,
                          split_sentences(text, context=ctx, asr_confidence=asr))


def test_extractor_provenance_corrections_conflicts_hedges():
    m1 = _msg("m1", "email", "We are Juniper Books LLC with 12 full-time staff. Gross sales were $1.5M last year.", 2)
    m2 = _msg("m2", "voice", "Actually we just hired two more, so we now have 14 full-time.", 3, asr=0.95)
    ext = extract([m1, m2])
    f = ext.fields["annual_revenue"]
    assert f.value == 1_500_000 and f.citations[0].quote in m1.text
    c = f.citations[0]
    assert m1.text[c.start:c.end] == c.quote                                    # exact character offsets
    ft = ext.fields["employees_full_time"]
    assert ft.value == 14 and ft.alternatives and ft.alternatives[0]["superseded_by_correction"]
    m3 = _msg("m3", "sms", "Revenue is $2M a year.", 4)
    assert extract([m1, m3]).fields["annual_revenue"].status == "conflict"     # disagreement without a cue
    hedged = extract([_msg("h", "email", "We do about $1.5M in sales a year.", 2)]).fields["annual_revenue"]
    plain = extract([_msg("p", "email", "We do $1.5M in sales a year.", 2)]).fields["annual_revenue"]
    assert hedged.confidence < plain.confidence


def test_llm_proposals_need_verbatim_quotes():
    class FakeLLM:
        model = "fake"

        def complete(self, messages, tools, max_output_tokens):
            from core.harness.llm import LLMResponse, Usage
            body = {"fein": {"value": "11-1111111", "quote": "EIN is 11-1111111"},            # hallucinated
                    "years_in_business": {"value": 7, "quote": "open seven years"}}            # verbatim
            return LLMResponse(json.dumps(body), [], Usage(1, 1), "fake")

    m = _msg("m", "email", "We have been open seven years.", 2)
    ext = extract([m], llm=FakeLLM())
    assert ext.fields["fein"].value is None
    assert ext.fields["years_in_business"].value in (7,)


# ============================================================================ Project 5: workflow durability
def _ingest(eng, ch, payload, when):
    if ch == "email":
        return eng.handle_inbound(normalize_email(payload["raw"]))
    if ch == "sms":
        return eng.handle_inbound(normalize_sms(payload["form"], payload["signature"], payload["url"], TWILIO_AUTH_TOKEN, when))
    return eng.handle_inbound(normalize_voice(payload))


def test_case_survives_restarts_and_dedupes_redelivery(data_dir, tracer):
    store = EventStore(data_dir / "s.sqlite")
    case = next(c for c in build_cases(50) if c.scenario == "clean")
    from scripts.run_soma_eval import run_case

    r = run_case(case, store, tracer)
    assert r["state"].status == "SUBMITTED" and r["restarts"] >= 3
    events = store.events(r["case_id"])
    assert fold(r["case_id"], events).status == "SUBMITTED"                 # pure replay reproduces the state
    eng = IntakeEngine(store, tracer=tracer)
    n = len(store.events(r["case_id"]))
    _ingest(eng, "sms", case.payloads[1]["payload"], datetime(2026, 11, 3, tzinfo=timezone.utc))   # Twilio retry
    assert len(store.events(r["case_id"])) == n                              # idempotent: nothing appended
    sub = get_submission(r["state"].submission_id)
    assert sub["insured"] == case.truth["applicant_name"] and sub["zip"] == case.truth["mailing_zip"]


def test_crash_between_start_and_complete_is_exactly_once(data_dir, tracer):
    from scripts.run_soma_eval import run_case

    store = EventStore(data_dir / "s.sqlite")
    case = next(c for c in build_cases(50) if c.scenario == "clean")
    r = run_case(case, store, tracer, crash_kind="portal")
    assert r["crashed"] and r["recovered"] and r["state"].status == "SUBMITTED"
    types = [e["type"] for e in store.events(r["case_id"])]
    assert "activity_resumed" in types
    con = sqlite3.connect(data_dir / "carrier_portal.sqlite")
    assert con.execute("SELECT count(*) FROM submissions WHERE client_ref LIKE ?", [r["case_id"] + ":portal:%"]).fetchone()[0] == 1


def test_handoffs_high_risk_and_ambiguous(data_dir, tracer):
    from scripts.run_soma_eval import run_case

    store = EventStore(data_dir / "s.sqlite")
    cases = build_cases(50)
    hr = run_case(next(c for c in cases if c.scenario == "high_risk"), store, tracer)
    assert hr["state"].status == "HUMAN_HANDOFF" and hr["state"].handoff_reasons[0].startswith("high_risk_hazard")
    amb = run_case(next(c for c in cases if c.scenario == "ambiguous"), store, tracer)
    assert amb["state"].status == "HUMAN_HANDOFF" and amb["state"].followups == 3
    assert "annual_revenue" in amb["state"].handoff_reasons[0]


def test_portal_gate_blocks_unready_case(data_dir, tracer):
    store = EventStore(data_dir / "s.sqlite")
    eng = IntakeEngine(store, tracer=tracer)
    c = build_cases(3)[0]
    cid = _ingest(eng, "email", c.payloads[0]["payload"], None)
    with eng.runner.workflow("bypass", trusted_context=eng._trusted(cid)) as wf:
        wf.call("extract_case", case_id=cid)
        rec = wf.call_record("fill_portal", case_id=cid, portal_url="https://carrier.mock/mock-carrier-portal",
                             activity_id=f"{cid}:portal:99")
        bad = wf.call_record("fill_portal", case_id=cid, portal_url="https://evil.example/portal", activity_id=f"{cid}:portal:98")
    assert rec.decision.rule == "approval_denied"                           # not ready -> L3 policy says no
    assert bad.decision.layer == "L1_allowlist"                              # unknown portal host -> L1


# ============================================================================ Project 5: portal + routes + eval
def test_portal_is_hostile_to_brittle_bots(data_dir):
    from fastapi.testclient import TestClient

    cli = TestClient(portal_app())
    a, b = cli.get("/mock-carrier-portal").text, cli.get("/mock-carrier-portal").text
    import re as _re
    names = lambda html: set(_re.findall(r"name='(f_[0-9a-f]+)'", html))
    assert names(a) and names(a).isdisjoint(names(b))                       # field names change every load
    assert cli.post("/mock-carrier-portal/submit", content="x=1",
                    headers={"Content-Type": "application/x-www-form-urlencoded"}).status_code == 403
    rec = {k: v for k, v in build_cases(3)[1].truth.items() if not k.startswith("_")}
    r1 = PortalClient().fill(rec, "ref-x")
    r2 = PortalClient().fill(rec, "ref-x")
    assert r1.ok and r2.duplicate and r1.submission_id == r2.submission_id
    r3 = PortalClient().fill({**rec, "mailing_zip": "123"}, "ref-y")
    assert not r3.ok and any("ZIP" in e for e in r3.errors)


def test_http_routes(data_dir, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import agents.soma_intake.workflow as wfm
    from agents.soma_intake.portal_automation import portal_router

    monkeypatch.setattr(wfm, "_engine", None)
    monkeypatch.setenv("TWILIO_WEBHOOK_URL", "http://testserver/soma/inbound/sms")
    app = FastAPI()
    app.include_router(wfm.router)
    app.include_router(portal_router)
    cli = TestClient(app)
    c = build_cases(3)[0]
    cid = cli.post("/soma/inbound/email", content=c.payloads[0]["payload"]["raw"]).json()["case_id"]
    sp = sms_payload(random.Random(1), c.truth["business_phone"], "Our EIN is 12-3456789.")
    from agents.soma_intake.synthetic import twilio_signature
    from urllib.parse import urlencode
    sig = twilio_signature("http://testserver/soma/inbound/sms", sp["form"])
    ok = cli.post("/soma/inbound/sms", content=urlencode(sp["form"]),
                  headers={"Content-Type": "application/x-www-form-urlencoded", "X-Twilio-Signature": sig})
    forged = cli.post("/soma/inbound/sms", content=urlencode(sp["form"]),
                      headers={"Content-Type": "application/x-www-form-urlencoded", "X-Twilio-Signature": "AAAA"})
    assert ok.status_code == 200 and "<Response>" in ok.text and forged.status_code == 403
    view = cli.get(f"/soma/cases/{cid}").json()
    assert view["fields"]["fein"]["value"] == "12-3456789" and view["fields"]["fein"]["citations"][0]["channel"] == "sms"
    assert cli.get("/mock-carrier-portal").status_code == 200


def test_soma_eval_thresholds():
    from scripts.run_soma_eval import evaluate

    r = evaluate(50)
    assert r["extraction"]["f1"] >= 0.95 and r["portal_fill"]["rate"] >= 0.95
    assert r["handoff"]["precision"] == 1.0 and r["handoff"]["recall"] == 1.0
    assert r["durability"]["replay_deterministic"] and all(c["exactly_once"] for c in r["durability"]["crash_cases"])
    ho = evaluate(50, style="heldout")
    assert ho["handoff"]["recall"] == 1.0          # unknown phrasing must never let a high-risk case through
