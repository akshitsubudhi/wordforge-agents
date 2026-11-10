"""
================================================================================
BLOCK: Soma Multi-Channel Intake Workflow (durable, replayable, gated)
--------------------------------------------------------------------------------
Role & contracts
  Ingestion normalisers turn provider payloads into InboundMessage:
    normalize_email(raw RFC 5322)        - header parse, quoted-reply strip
    normalize_sms(Twilio webhook form)   - X-Twilio-Signature HMAC-SHA1 check
    normalize_voice(diarised transcript) - caller turns, agent question as
                                           context, per-segment ASR confidence
  IntakeEngine (event-sourced, Temporal-style) runs each case:
    NEW -> COLLECTING -> AWAITING_CUSTOMER <-> COLLECTING -> SUBMITTED
                     \\-> HUMAN_HANDOFF (high-risk hazard | confidence < 0.80
                         after 3 follow-ups | unresolved conflict | portal
                         rejection | no response)
  Every fact is an append-only event in SQLite (`intake_events`); state is a
  pure fold over the history, so ANY process can crash and the next one
  replays the case exactly where it stopped (a 3-day Email -> SMS -> Voice
  thread survives restarts).
  Downstream: harness tools (send_followup, fill_portal, request_handoff)
  -> outbound gateway / mock carrier portal; /soma/* FastAPI routes.

Key engineering mechanism
  * Event history + idempotency keys: provider message ids dedupe redelivery
    (a Twilio webhook retry is a no-op). Activities are recorded as
    activity_started -> activity_completed. A crash in between is
    detected on replay and the activity is re-run with the SAME activity id,
    which the outbound gateway and the portal dedupe: exactly-once effects.
  * Decisions are a pure function decide(state, extraction): replaying the
    history reproduces every decision.
  * Identity resolution across channels: contact index (email <-> phone).
    The e-mail signature's phone links the later SMS and voice call to the
    same case.
  * Durable timers: follow-ups carry remind_at; tick(now) fires reminders
    or a no-response handoff.
  * Security: customer text is untrusted. It is read via a tool marked
    untrusted_output, which taints the harness session, so every outbound
    action needs the policy approver. Tools take only a case_id and field
    KEYS, never customer-supplied strings, so nothing can be smuggled into
    an outbound message or a portal field via tool arguments. Follow-up
    questions are templates that never echo customer text.

Failure modes & 1-step mitigation
  * Process restart mid-case -> IntakeEngine(store).advance(case_id) replays.
  * Crash between "portal submit" and "record result" -> replay re-submits
    with the same client_ref; the portal returns the original submission id.
  * Low confidence / high risk -> HUMAN_HANDOFF with machine-readable reasons
    and the full provenance trail (GET /soma/cases/{id}).
  * Forged SMS webhook -> signature mismatch -> rejected (403) before any
    state change.
================================================================================
"""
from __future__ import annotations

import base64
import email
import email.policy
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import Field

from agents.soma_intake.extractor import (
    FIELDS, REQUIRED, SPEC, CaseExtraction, InboundMessage, Sentence, extract, split_sentences,
)
from agents.soma_intake.portal_automation import PORTAL_URL, PortalClient
from core.config import get_settings
from core.harness.llm import MockLLM, is_real_llm
from core.harness.resilience import BudgetGuard
from core.harness.redteam_guard import READ_WRITE, AgentPolicy, ApprovalRequest, DefenseConfig
from core.harness.runner import AgentRunner, AgentSpec
from core.harness.telemetry import Tracer, default_tracer
from core.harness.tools import ArgKind, Privilege, StrictArgs, ToolContext, ToolRegistry, ToolSpec

MAX_FOLLOWUPS = 3
REMIND_AFTER = timedelta(hours=24)
TERMINAL = {"SUBMITTED", "HUMAN_HANDOFF"}
QUESTIONS = {
    "applicant_name": "What is the full legal name of the business?",
    "entity_type": "Is the business an LLC, a corporation, a partnership or a sole proprietorship?",
    "mailing_street": "What is the business mailing address (street, city, state, ZIP)?",
    "mailing_city": "What is the business mailing address (street, city, state, ZIP)?",
    "mailing_state": "What is the business mailing address (street, city, state, ZIP)?",
    "mailing_zip": "What is the business mailing address (street, city, state, ZIP)?",
    "contact_name": "Who is the best contact person for this application?",
    "contact_email": "What e-mail address should we use for documents?",
    "business_description": "What does the business do day to day?",
    "employees_full_time": "How many full-time and part-time employees do you have?",
    "annual_revenue": "What is your annual revenue (gross sales)?",
    "policy_effective_date": "What date should coverage start?",
    "lines_of_business": "Which coverages do you need (for example general liability or property)?",
    "gl_each_occurrence": "What liability limits do you want (per occurrence / aggregate)?",
    "gl_general_aggregate": "What liability limits do you want (per occurrence / aggregate)?",
    "prior_losses_3y": "How many insurance claims or losses has the business had in the last three years?",
    "prior_carrier": "Who is your current or most recent insurance carrier (or is this your first policy)?",
}


class SignatureError(ValueError):
    pass


# ============================================================================ normalisers
def normalize_email(raw: str) -> InboundMessage:
    msg = email.message_from_string(raw, policy=email.policy.default)
    from_name, from_addr = email.utils.parseaddr(str(msg.get("From", "")))
    try:
        when = parsedate_to_datetime(str(msg["Date"])).astimezone(timezone.utc)
    except (TypeError, ValueError):
        when = datetime.now(timezone.utc)
    part = msg.get_body(preferencelist=("plain",)) if msg.is_multipart() else msg
    body = part.get_content() if part else ""
    lines, out = body.splitlines(), []
    for ln in lines:
        if re.match(r"^On .{3,120} wrote:\s*$", ln.strip()):
            break  # quoted trail from here on
        if ln.lstrip().startswith(">"):
            continue
        out.append(ln)
    text = "\n".join(out).strip()
    mid = str(msg.get("Message-ID", "")).strip("<> ") or "email-" + hashlib.sha1(raw.encode()).hexdigest()[:16]
    return InboundMessage(mid, "email", from_addr.lower(), when.isoformat(), text, split_sentences(text),
                          {"from_name": from_name or None, "subject": str(msg.get("Subject", ""))})


def twilio_signature(url: str, params: dict, token: str) -> str:
    data = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def normalize_sms(form: dict, signature: str, url: str, auth_token: str, received_at: datetime) -> InboundMessage:
    if not hmac.compare_digest(twilio_signature(url, form, auth_token), signature or ""):
        raise SignatureError("X-Twilio-Signature mismatch")
    body = form.get("Body", "").strip()
    return InboundMessage(form["MessageSid"], "sms", form["From"], received_at.astimezone(timezone.utc).isoformat(),
                          body, split_sentences(body), {"to": form.get("To"), "num_media": form.get("NumMedia", "0")})


def normalize_voice(payload: dict) -> InboundMessage:
    pieces, sents, offset, context = [], [], 0, ""
    for seg in payload["segments"]:
        if seg["speaker"] != "caller":
            context = seg["text"]
            continue
        s = split_sentences(seg["text"], context=context, asr_confidence=float(seg.get("confidence", 1.0)),
                            offset=offset, start_idx=len(sents))
        sents += s
        pieces.append(seg["text"])
        offset += len(seg["text"]) + 1
    when = datetime.fromisoformat(payload["started_at"]).astimezone(timezone.utc)
    return InboundMessage(payload["call_sid"], "voice", payload["from"], when.isoformat(), "\n".join(pieces), sents,
                          {"segments": len(payload["segments"])})


# ============================================================================ event store
class EventStore:
    """Append-only SQLite event log (WAL), safe across processes."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=30)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS intake_events (case_id TEXT, seq INTEGER, ts TEXT, type TEXT, payload TEXT,
                                                      idem TEXT UNIQUE, PRIMARY KEY (case_id, seq));
            CREATE TABLE IF NOT EXISTS contact_index (contact TEXT PRIMARY KEY, case_id TEXT);
            CREATE TABLE IF NOT EXISTS outbound_messages (idem TEXT PRIMARY KEY, case_id TEXT, channel TEXT,
                                                          to_addr TEXT, body TEXT, ts TEXT);""")
        self.lock = threading.Lock()

    def append(self, case_id: str, type_: str, payload: dict, idem: str | None = None, ts: str | None = None) -> bool:
        """Returns False if an event with this idempotency key already exists (dedupe)."""
        with self.lock:
            self.con.execute("BEGIN IMMEDIATE")
            try:
                if idem and self.con.execute("SELECT 1 FROM intake_events WHERE idem=?", [idem]).fetchone():
                    self.con.execute("COMMIT")
                    return False
                seq = self.con.execute("SELECT coalesce(max(seq), 0) + 1 FROM intake_events WHERE case_id=?", [case_id]).fetchone()[0]
                self.con.execute("INSERT INTO intake_events VALUES (?, ?, ?, ?, ?, ?)",
                                 [case_id, seq, ts or datetime.now(timezone.utc).isoformat(), type_,
                                  json.dumps(payload, default=str), idem])
                self.con.execute("COMMIT")
                return True
            except Exception:
                self.con.execute("ROLLBACK")
                raise

    def events(self, case_id: str) -> list[dict]:
        rows = self.con.execute("SELECT seq, ts, type, payload FROM intake_events WHERE case_id=? ORDER BY seq", [case_id]).fetchall()
        return [{"seq": r[0], "ts": r[1], "type": r[2], "payload": json.loads(r[3])} for r in rows]

    def case_for(self, contact: str) -> str | None:
        r = self.con.execute("SELECT case_id FROM contact_index WHERE contact=?", [contact.lower()]).fetchone()
        return r[0] if r else None

    def link(self, contact: str, case_id: str) -> None:
        with self.lock:
            self.con.execute("INSERT OR IGNORE INTO contact_index VALUES (?, ?)", [contact.lower(), case_id])

    def case_ids(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT case_id FROM intake_events ORDER BY case_id").fetchall()]

    def send(self, idem: str, case_id: str, channel: str, to: str, body: str, ts: str) -> bool:
        """Outbound gateway (stands in for Twilio / SendGrid). Exactly once per idempotency key."""
        with self.lock:
            cur = self.con.execute("INSERT OR IGNORE INTO outbound_messages VALUES (?, ?, ?, ?, ?, ?)",
                                   [idem, case_id, channel, to, body, ts])
            return cur.rowcount == 1

    def outbound(self, case_id: str) -> list[dict]:
        rows = self.con.execute("SELECT channel, to_addr, body, ts FROM outbound_messages WHERE case_id=? ORDER BY ts", [case_id]).fetchall()
        return [{"channel": r[0], "to": r[1], "body": r[2], "ts": r[3]} for r in rows]


# ============================================================================ state (pure fold)
@dataclass
class CaseState:
    case_id: str
    status: str = "NEW"
    messages: list[InboundMessage] = field(default_factory=list)
    contacts: dict[str, str] = field(default_factory=dict)   # channel -> address
    last_channel: str = "email"
    followups: int = 0
    last_question: str = ""
    last_asked: list[str] = field(default_factory=list)
    inbound_since_followup: int = 0
    remind_at: str | None = None
    activities: dict[str, dict] = field(default_factory=dict)
    handoff_reasons: list[str] = field(default_factory=list)
    submission_id: str | None = None
    portal_errors: list[str] = field(default_factory=list)
    turns_in: int = 0
    turns_out: int = 0
    now: str = ""

    @property
    def pending_activity(self) -> dict | None:
        return next((a for a in self.activities.values() if not a.get("done")), None)


def fold(case_id: str, events: list[dict]) -> CaseState:
    st = CaseState(case_id)
    for e in events:
        p, t = e["payload"], e["type"]
        st.now = p.get("at", st.now)
        if t == "case_opened":
            st.status = "COLLECTING"
        elif t == "message_received":
            m = InboundMessage.from_dict(p["message"])
            st.messages.append(m)
            st.contacts.setdefault(m.channel if m.channel != "voice" else "sms", m.contact)
            st.last_channel = m.channel
            st.turns_in += 1
            st.inbound_since_followup += 1
            st.now = m.received_at
            if st.status == "AWAITING_CUSTOMER":
                st.status = "COLLECTING"
        elif t == "contact_linked":
            st.contacts.setdefault(p["channel"], p["contact"])
        elif t == "activity_started":
            st.activities[p["activity_id"]] = {"kind": p["kind"], "args": p["args"], "done": False}
        elif t == "activity_completed":
            a = st.activities.setdefault(p["activity_id"], {"kind": p["kind"], "args": {}})
            a.update(done=True, result=p["result"])
            r = p["result"]
            if p["kind"] == "followup" and r.get("sent") is not None:
                st.followups += 1
                st.turns_out += 1
                st.last_question, st.last_asked = r["question"], r["fields"]
                st.inbound_since_followup = 0
                st.remind_at = r["remind_at"]
                st.status = "AWAITING_CUSTOMER"
            elif p["kind"] == "portal":
                if r.get("ok"):
                    st.submission_id, st.status = r["submission_id"], "SUBMITTED"
                else:
                    st.portal_errors = r.get("errors", [])
            elif p["kind"] == "handoff" and r.get("reasons"):
                st.handoff_reasons, st.status = r["reasons"], "HUMAN_HANDOFF"
        elif t == "timer_fired":
            st.remind_at = None
    return st


@dataclass
class Decision:
    kind: str                      # wait | followup | submit | handoff
    fields: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def decide(st: CaseState, ext: CaseExtraction, reminder: bool = False) -> Decision:
    """Pure. Replaying the same history always yields the same decision."""
    if st.status in TERMINAL:
        return Decision("wait")
    if ext.high_risk_hazards:
        return Decision("handoff", reasons=[f"high_risk_hazard:{h}" for h in ext.high_risk_hazards])
    if st.portal_errors:
        return Decision("handoff", reasons=["portal_rejected:" + "; ".join(st.portal_errors)[:200]])
    if st.status == "AWAITING_CUSTOMER" and st.inbound_since_followup == 0 and not reminder:
        return Decision("wait")
    need = ext.conflicts() + [k for k in ext.gaps() if k not in ext.conflicts()]
    if need:
        if st.followups >= MAX_FOLLOWUPS:
            low = [f"{k}={ext.fields[k].confidence:.2f}" for k in need]
            return Decision("handoff", reasons=[("unresolved_conflict:" if ext.conflicts() else "low_confidence:") + ",".join(low)])
        return Decision("followup", fields=need)
    return Decision("submit")


# ============================================================================ harness tools
class CaseArgs(StrictArgs):
    case_id: str = Field(pattern=r"^case_[0-9a-f]{10}$")


class FollowupArgs(StrictArgs):
    case_id: str = Field(pattern=r"^case_[0-9a-f]{10}$")
    fields: list[str] = Field(min_length=1, max_length=12)
    activity_id: str = Field(pattern=r"^[a-z0-9_:]{8,80}$")


class PortalArgs(StrictArgs):
    case_id: str = Field(pattern=r"^case_[0-9a-f]{10}$")
    portal_url: str = ArgKind("url")
    activity_id: str = Field(pattern=r"^[a-z0-9_:]{8,80}$")


class HandoffArgs(StrictArgs):
    case_id: str = Field(pattern=r"^case_[0-9a-f]{10}$")
    reasons: list[str] = Field(min_length=1, max_length=8)
    activity_id: str = Field(pattern=r"^[a-z0-9_:]{8,80}$")


INTAKE_TOOLS = frozenset({"extract_case", "send_followup", "fill_portal", "request_handoff"})


class IntakeEngine:
    SYSTEM = ("You are Soma's commercial insurance intake agent. Collect ACORD 125/126 facts across e-mail, SMS and "
              "voice, ask concise follow-ups, hand off to a human when confidence is low or risk is high, and submit "
              "complete applications to the carrier portal.")

    def __init__(self, store: EventStore, *, portal: PortalClient | None = None, llm=None, tracer: Tracer | None = None,
                 defenses: DefenseConfig = DefenseConfig(), crash_hook: Callable[[str, str], None] | None = None):
        self.store, self.portal, self.llm = store, portal or PortalClient(), llm
        self.crash_hook = crash_hook
        self._ext_cache: dict[str, CaseExtraction] = {}
        self._llm_budgets: dict[str, BudgetGuard] = {}
        self.runner = AgentRunner(
            AgentSpec("soma_intake", self.SYSTEM,
                      AgentPolicy(INTAKE_TOOLS, get_settings().data_dir, allowed_url_hosts=frozenset({"carrier.mock"}))),
            self._registry(), MockLLM("compliant"), defenses=defenses, tracer=tracer or default_tracer(),
            approver=self._policy)

    @staticmethod
    def _trusted(case_id: str) -> str:
        """Values the ENGINE owns (case id from routing, schema field keys, portal URL). The extract_case output
        echoes them, so without this L3's copied-argument rule would (rightly, from its view) block every action."""
        return " ".join([case_id, PORTAL_URL, *SPEC.keys()])

    # ---------------------------------------------------------------- state
    def state(self, case_id: str) -> CaseState:
        return fold(case_id, self.store.events(case_id))

    def extraction(self, st: CaseState) -> CaseExtraction:
        key = st.case_id + ":" + ",".join(m.message_id for m in st.messages)
        if key not in self._ext_cache:
            budget = None
            if is_real_llm(self.llm):   # ONE spend cap per case (HARNESS_MAX_RUN_USD), not a fresh one per call
                cfg = get_settings()
                budget = self._llm_budgets.setdefault(st.case_id, BudgetGuard(cfg.max_run_usd, cfg.max_run_tokens, 10_000))
            self._ext_cache[key] = extract(st.messages, self.llm, budget, tracer=self.runner.tracer)
        return self._ext_cache[key]

    # ---------------------------------------------------------------- policy approver (L3 gate)
    def _policy(self, req: ApprovalRequest) -> tuple[bool, str]:
        st = self.state(req.args.get("case_id", ""))
        if st.status in TERMINAL and req.tool != "request_handoff":
            return False, f"case is {st.status}"
        if req.tool == "send_followup":
            bad = [f for f in req.args.get("fields", []) if f not in SPEC]
            if bad:
                return False, f"unknown fields {bad}"
            return (st.followups < MAX_FOLLOWUPS), f"follow-up {st.followups + 1}/{MAX_FOLLOWUPS}"
        if req.tool == "fill_portal":
            ext = self.extraction(st)
            if ext.high_risk_hazards:
                return False, f"high-risk hazards {ext.high_risk_hazards} require underwriter review"
            return (ext.ready, "all required fields >= 0.80, no conflicts") if ext.ready else (False, f"not ready: {ext.gaps() + ext.conflicts()}")
        if req.tool == "request_handoff":
            return True, "handoff to a human is always allowed"
        return False, f"no policy for {req.tool}"

    # ---------------------------------------------------------------- tools
    def _t_extract(self, a: CaseArgs, ctx: ToolContext) -> dict:
        st = self.state(a.case_id)
        ext = self.extraction(st)
        return {"case_id": a.case_id, "ready": ext.ready, "gaps": ext.gaps(), "conflicts": ext.conflicts(),
                "hazards": ext.value("hazards"), "fields": {k: {"value": f.value, "confidence": round(f.confidence, 3)}
                                                            for k, f in ext.fields.items()}}

    def _t_followup(self, a: FollowupArgs, ctx: ToolContext) -> dict:
        st = self.state(a.case_id)
        channel = {"voice": "sms"}.get(st.last_channel, st.last_channel)
        to = st.contacts.get(channel) or st.contacts.get("email") or st.contacts.get("sms")
        qs = []
        for f in a.fields:
            q = QUESTIONS.get(f, f"Could you confirm the {SPEC[f].label.lower()}?")
            if f in self.extraction(st).conflicts():
                q = f"We received two different answers for {SPEC[f].label.lower()}. Which one is correct?"
            if q not in qs:
                qs.append(q)
        body = "Thanks for the details so far. " + " ".join(qs) if st.followups == 0 else \
               ("Just a quick reminder: " if ctx.extras.get("reminder") else "Thanks! ") + " ".join(qs)
        sent = self.store.send(a.activity_id, a.case_id, channel, to, body, ctx.extras.get("now", ""))
        return {"sent": sent, "channel": channel, "to": to, "fields": a.fields, "question": body,
                "remind_at": (datetime.fromisoformat(ctx.extras["now"]) + REMIND_AFTER).isoformat() if ctx.extras.get("now") else None}

    def _t_portal(self, a: PortalArgs, ctx: ToolContext) -> dict:
        st = self.state(a.case_id)
        rec = self.extraction(st).record()
        res = self.portal.fill(rec, client_ref=a.activity_id)
        return {"ok": res.ok, "submission_id": res.submission_id, "errors": res.errors, "duplicate": res.duplicate,
                "unmapped_required": res.unmapped_required, "status_code": res.status_code}

    def _t_handoff(self, a: HandoffArgs, ctx: ToolContext) -> dict:
        return {"reasons": a.reasons, "queue": "underwriting" if any(r.startswith("high_risk") for r in a.reasons) else "intake_specialist"}

    def _registry(self) -> ToolRegistry:
        R, W, E = Privilege.READ, Privilege.WRITE, Privilege.EGRESS
        return ToolRegistry([
            ToolSpec("extract_case", "ACORD extraction with confidence + provenance", CaseArgs, self._t_extract, R,
                     untrusted_output=True, timeout_s=90 if is_real_llm(self.llm) else 30),
            ToolSpec("send_followup", "Ask the customer for missing/low-confidence fields", FollowupArgs, self._t_followup, E),
            ToolSpec("fill_portal", "Submit the application to the carrier portal", PortalArgs, self._t_portal, E,
                     requires_approval=True, timeout_s=30),
            ToolSpec("request_handoff", "Route the case to a human", HandoffArgs, self._t_handoff, W),
        ])

    # ---------------------------------------------------------------- inbound
    def handle_inbound(self, msg: InboundMessage) -> str:
        case_id = self.store.case_for(msg.contact)
        if case_id is None:
            case_id = "case_" + hashlib.sha1(f"{msg.contact}:{msg.message_id}".encode()).hexdigest()[:10]
            self.store.append(case_id, "case_opened", {"channel": msg.channel, "contact": msg.contact, "at": msg.received_at},
                              idem=f"open:{case_id}", ts=msg.received_at)
            self.store.link(msg.contact, case_id)
        st = self.state(case_id)
        if msg.channel in ("sms", "email") and st.last_question:
            for s in msg.sentences:  # the reply answers our last question: keep it as context for extraction
                s.context = s.context or st.last_question
        fresh = self.store.append(case_id, "message_received", {"message": msg.to_dict()}, idem=f"msg:{msg.message_id}",
                                  ts=msg.received_at)
        if fresh:
            self.advance(case_id, now=msg.received_at)
        return case_id

    # ---------------------------------------------------------------- activities
    def _activity(self, wf, st: CaseState, kind: str, tool: str, args: dict, now: str, reminder: bool = False) -> dict | None:
        n = sum(1 for a in st.activities.values() if a["kind"] == kind)
        act_id = f"{st.case_id}:{kind}:{n}"
        return self._run_activity(wf, st.case_id, act_id, kind, tool, args, now, reminder)

    def _run_activity(self, wf, case_id: str, act_id: str, kind: str, tool: str, args: dict, now: str, reminder: bool = False):
        self.store.append(case_id, "activity_started", {"activity_id": act_id, "kind": kind, "args": args, "at": now},
                          idem=f"start:{act_id}", ts=now)
        if self.crash_hook:
            self.crash_hook(case_id, kind)       # tests/eval inject crashes here
        wf.ctx.extras.update(now=now, reminder=reminder)
        rec = wf.call_record(tool, **args, activity_id=act_id)
        if rec.decision and not rec.decision.allowed:
            result = {"blocked": f"{rec.decision.layer}:{rec.decision.rule}", "detail": rec.decision.detail}
            if kind == "portal":
                result.update(ok=False, errors=[f"gate: {rec.decision.detail}"])
        elif not rec.ok:
            result = {"error": "tool failed", "ok": False, "errors": ["tool failed"]} if kind == "portal" else {"error": "tool failed"}
        else:
            result = rec.raw
        self.store.append(case_id, "activity_completed", {"activity_id": act_id, "kind": kind, "result": result, "at": now},
                          idem=f"done:{act_id}", ts=now)
        return result

    # ---------------------------------------------------------------- main step
    def advance(self, case_id: str, now: str | None = None, reminder: bool = False) -> CaseState:
        st = self.state(case_id)
        now = now or st.now or datetime.now(timezone.utc).isoformat()
        with self.runner.workflow(f"intake:{case_id}", mode=READ_WRITE, trusted_context=self._trusted(case_id)) as wf:
            pend = st.pending_activity
            if pend:  # crash recovery: re-run the interrupted activity with the SAME id (exactly-once downstream)
                act_id = next(k for k, v in st.activities.items() if v is pend)
                tool = {"followup": "send_followup", "portal": "fill_portal", "handoff": "request_handoff"}[pend["kind"]]
                args = {k: v for k, v in pend["args"].items() if k != "activity_id"}
                self.store.append(case_id, "activity_resumed", {"activity_id": act_id, "at": now}, idem=f"resume:{act_id}", ts=now)
                self._run_activity(wf, case_id, act_id, pend["kind"], tool, args, now)
                st = self.state(case_id)
            wf.call("extract_case", case_id=case_id)          # taints the session: customer text is untrusted
            ext = self.extraction(st)
            for k, ch in (("contact_email", "email"), ("business_phone", "sms")):
                f = ext.fields[k]
                if f.value and f.confidence >= 0.8 and ch not in st.contacts:
                    self.store.link(f.value, case_id)
                    self.store.append(case_id, "contact_linked", {"channel": ch, "contact": f.value, "at": now},
                                      idem=f"link:{case_id}:{ch}:{f.value}", ts=now)
            st = self.state(case_id)
            d = decide(st, ext, reminder)
            if d.kind == "followup":
                self._activity(wf, st, "followup", "send_followup", {"case_id": case_id, "fields": d.fields}, now, reminder)
            elif d.kind == "submit":
                res = self._activity(wf, st, "portal", "fill_portal", {"case_id": case_id, "portal_url": PORTAL_URL}, now)
                if res and not res.get("ok"):
                    st = self.state(case_id)
                    self._activity(wf, st, "handoff", "request_handoff",
                                   {"case_id": case_id, "reasons": ["portal_rejected:" + "; ".join(res.get("errors", []))[:180]]}, now)
            elif d.kind == "handoff":
                self._activity(wf, st, "handoff", "request_handoff", {"case_id": case_id, "reasons": d.reasons}, now)
        return self.state(case_id)

    def tick(self, now: str) -> list[str]:
        """Durable timers: remind unanswered follow-ups, or hand off after the last one."""
        fired = []
        for cid in self.store.case_ids():
            st = self.state(cid)
            if st.status == "AWAITING_CUSTOMER" and st.remind_at and st.remind_at <= now:
                self.store.append(cid, "timer_fired", {"remind_at": st.remind_at, "at": now}, idem=f"timer:{cid}:{st.remind_at}", ts=now)
                if st.followups >= MAX_FOLLOWUPS:
                    with self.runner.workflow(f"intake:{cid}:timeout", mode=READ_WRITE, trusted_context=self._trusted(cid)) as wf:
                        self._activity(wf, self.state(cid), "handoff", "request_handoff",
                                       {"case_id": cid, "reasons": ["no_response_after_followups"]}, now)
                else:
                    self.advance(cid, now=now, reminder=True)
                fired.append(cid)
        return fired

    def recover(self, now: str | None = None) -> list[str]:
        """Process start-up: finish any activity a crashed process left half-done (started, never completed)."""
        resumed = []
        for cid in self.store.case_ids():
            if self.state(cid).pending_activity:
                self.advance(cid, now=now)
                resumed.append(cid)
        return resumed

    def case_view(self, case_id: str) -> dict:
        st = self.state(case_id)
        ext = self.extraction(st)
        return {"case_id": case_id, "status": st.status, "followups": st.followups, "turns": st.turns_in + st.turns_out,
                "submission_id": st.submission_id, "handoff_reasons": st.handoff_reasons, "contacts": st.contacts,
                "fields": ext.to_dict(), "outbound": self.store.outbound(case_id),
                "history": [{"seq": e["seq"], "type": e["type"], "ts": e["ts"]} for e in self.store.events(case_id)]}


# ============================================================================ HTTP surface
router = APIRouter(prefix="/soma", tags=["soma-intake"])
_engine: IntakeEngine | None = None
_engine_lock = threading.Lock()


def engine() -> IntakeEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            from core.harness.llm import get_llm  # real provider when a key is set, deterministic mock otherwise

            _engine = IntakeEngine(EventStore(get_settings().data_dir / "soma_intake.sqlite"), llm=get_llm())
        return _engine


@router.post("/inbound/email")
async def inbound_email(request: Request) -> dict:
    msg = normalize_email((await request.body()).decode("utf-8", "replace"))
    return {"case_id": engine().handle_inbound(msg)}


@router.post("/inbound/sms", response_class=PlainTextResponse)
async def inbound_sms(request: Request) -> str:
    form = dict(parse_qsl((await request.body()).decode(), keep_blank_values=True))
    token = os.environ.get("TWILIO_AUTH_TOKEN", "test-auth-token-not-a-secret")
    url = os.environ.get("TWILIO_WEBHOOK_URL", str(request.url))
    try:
        msg = normalize_sms(form, request.headers.get("X-Twilio-Signature", ""), url, token, datetime.now(timezone.utc))
    except SignatureError:
        raise HTTPException(403, "invalid Twilio signature")
    engine().handle_inbound(msg)
    return "<?xml version='1.0' encoding='UTF-8'?><Response></Response>"


@router.post("/inbound/voice")
async def inbound_voice(request: Request) -> dict:
    return {"case_id": engine().handle_inbound(normalize_voice(await request.json()))}


@router.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    if not re.fullmatch(r"case_[0-9a-f]{10}", case_id) or not engine().store.events(case_id):
        raise HTTPException(404, "unknown case")
    return engine().case_view(case_id)
