"""
================================================================================
BLOCK: Agent Common Utilities (approval ledger, audit log, simulated play)
--------------------------------------------------------------------------------
Role & contracts
  Shared by agents/liveops_copilot and agents/level_factory.
    * ApprovalLedger - wraps any decision function as a harness Approver and
      writes EVERY decision (approved or denied, by whom, why) to the
      `approvals` table. The L3 gate calls it; agents never write approvals.
    * audit()        - append-only `agent_audit` trail of agent actions.
    * inject_sim_attempts() - writes simulated level attempts (source='sim')
      for demos/tests of the live auto-rollback monitor.

Key engineering mechanism
  Decision functions are plain callables (ApprovalRequest -> (bool, reason)),
  so the same gate runs with an interactive human (CLI prompt), a scripted
  operator policy (evals), or a machine release policy (Level Factory).

Failure modes & 1-step mitigation
  * Ledger write fails -> the approval is DENIED (fail-closed) and the error
    is returned as the reason; an unrecorded approval is never acted on.
================================================================================
"""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from core.db import TelemetryDB
from core.harness.redteam_guard import ApprovalRequest

Decision = Callable[[ApprovalRequest], tuple[bool, str]]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def audit(db: TelemetryDB, agent: str, action: str, **detail) -> None:
    with db.cursor() as con:
        con.execute("INSERT INTO agent_audit VALUES (?, ?, ?, ?)",
                    [utcnow(), agent, action, json.dumps(detail, default=str)])


class ApprovalLedger:
    def __init__(self, db: TelemetryDB, agent: str, decide: Decision, decided_by: str):
        self.db, self.agent, self.decide, self.decided_by = db, agent, decide, decided_by
        self.log: list[dict] = []

    def __call__(self, req: ApprovalRequest) -> tuple[bool, str]:
        try:
            ok, reason = self.decide(req)
        except Exception as e:  # a crashing approver must never approve
            ok, reason = False, f"approver error: {e}"
        entry = {"approval_id": uuid.uuid4().hex[:12], "tool": req.tool, "args": req.args,
                 "decision": "approved" if ok else "denied", "decided_by": self.decided_by, "reason": reason}
        try:
            with self.db.cursor() as con:
                con.execute("INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            [entry["approval_id"], utcnow(), self.agent, req.tool, json.dumps(req.args, default=str),
                             entry["decision"], self.decided_by, reason])
        except Exception as e:
            return False, f"approval ledger unavailable ({e}); denied (fail-closed)"
        self.log.append(entry)
        return ok, reason


def interactive_decider(req: ApprovalRequest) -> tuple[bool, str]:
    """Real human in the loop for CLI runs."""
    print(f"\n[APPROVAL REQUIRED] {req.tool} {json.dumps(req.args, default=str)}")
    ans = input("approve? [y/N] ").strip().lower()
    return ans in ("y", "yes"), "operator answered " + (ans or "no")


def inject_sim_attempts(db: TelemetryDB, level_id: int, version: int, attempts: int, fail_rate: float,
                        seed: int = 0, start: datetime | None = None) -> int:
    """Simulated live attempts (level_start + complete/fail) tagged source='sim'."""
    rng = random.Random(seed)
    t = start or utcnow() - timedelta(hours=6)
    rows = []
    for i in range(attempts):
        uid, sid, att = f"sim_{seed}_{i}", f"simsess_{seed}_{i}", f"simatt_{seed}_{level_id}_{version}_{i}"
        fail = rng.random() < fail_rate
        for name, dt in (("level_start", 0), ("level_fail" if fail else "level_complete", 80)):
            rows.append({"event_id": uuid.uuid4().hex, "ts": t + timedelta(seconds=i * 30 + dt), "user_id": uid,
                         "session_id": sid, "event_name": name, "level_id": level_id, "level_version": version,
                         "attempt_id": att, "ab_group": "gate_30", "platform": "sim", "word": None,
                         "revenue_usd": None, "network": None, "placement": None, "source": "sim", "props": "{}"})
    db.insert_events(rows)
    return attempts
