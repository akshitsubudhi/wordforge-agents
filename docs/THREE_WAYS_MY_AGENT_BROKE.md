# Three ways my agent broke

*A blameless engineering postmortem.*

## Scope: what this document is, and isn't

**None of this ran in production.** These are three failures I actually hit while building and
evaluating this repository. Each one shipped in at least one working build and was caught by a test,
an eval run or a smoke run before anyone relied on it. Every incident has:

- its observable symptom,
- the root cause,
- a trace ID you can reproduce with `python scripts/repro_incidents.py`,
- the deterministic fix,
- the regression test that keeps it fixed.

**The four named guardrails.** `solver.py`, mSPRT, `PolicyGuard` and `auto_rollback` appear in two
places:

- **Section 4** covers the failure classes they *prevent*. Those classes never occurred here, because
  the guardrails were designed in from the start. Rather than present them as incidents, each one is
  proven by a fault drill you can run.
- **Incidents 1 and 2** are failures *inside* the guardrails themselves: a guard that blocked legitimate
  work, and an auto-mitigation that acted on its own side effects.

**"Permanently eliminated" would be an overclaim.** Each fix is pinned by a regression test, which is
the strongest honest statement available. One related failure class is still open; see Section 5.

---

## Incident 1: the guardrail that silenced the agent

**Component:** L3 `PrivilegeGate`, copied-argument rule (`core/harness/redteam_guard.py`), as used by
the Soma intake engine.
**Severity, had it shipped:** high. Every insurance intake stalls silently.

### Symptom

The first end-to-end run of the Soma workflow processed 8 cases. **All 8 ended in `COLLECTING` with 0
follow-ups sent**, even though every case had gaps. Nothing crashed and nothing was logged as an error:
the cases just stopped.

### Root cause

The L3 rule blocks any write or egress argument (8+ characters) that appears verbatim in untrusted text
but not in trusted text. It exists to stop an attacker's e-mail address or URL being copied out of a
document into an outbound call. The chain was:

1. The engine first calls `extract_case` (correctly marked `untrusted_output`, since it carries customer
   text). Its JSON output **echoed the `case_id` and the schema's field keys**.
2. `send_followup(case_id, fields)` therefore looked, to L3, like an action whose arguments were
   *copied from untrusted text*. It was blocked with `write_arg_from_untrusted`.
3. The workflow recorded the blocked activity as completed-with-no-send. `decide()` then waited for a
   customer reply that would never come.

The guard did exactly what it was told. The defect was that **identifiers the engine owns** (from
routing and the schema) were never declared trusted.

**Reproduction trace:** `58a0ddafff611550c1b09156ef67d395`. This opens the intake workflow *without*
the fix and gets `L3_privilege:write_arg_from_untrusted` on `'case_22c52117da'`. The fixed path sends
its follow-up in trace `d5b28c8a2f0d7a72b85380bf31c1e815`.

### Related earlier occurrence

In Phase 1's smoke test, the L2 directive pattern was case-insensitive, so ordinary prose like "call
the support team" was redacted as if it were a tool directive. It's the same failure class: a security
layer silently degrading legitimate content.

### Fix (deterministic)

- **Trusted context for engine-owned values.** `IntakeEngine._trusted(case_id)` declares the case ID,
  the schema keys and the portal URL as the workflow's trusted context. Customer-supplied strings still
  never travel in tool arguments: tools take only IDs and field keys, so the protection keeps full
  strength.
- **Narrower L2 pattern.** The directive pattern is now case-sensitive `CALL snake_case_tool`.
- **False positives fail the build.** The conformance suite carries **10 benign control cases** (base64
  images, Cyrillic text, `{{template}}` braces, "ignore my previous review", SQL literals containing
  "drop", and so on). CI fails on any false alarm. Current result: 0/10.

### Regression tests

- `tests/test_phase3.py::test_case_survives_restarts_and_dedupes_redelivery`
- `tests/test_phase1.py::test_l2_benign_text_untouched`
- `tests/test_phase1.py::test_conformance_suite_ablation` (asserts 0 false alarms in every config)

### Lesson

A security control that fails *closed and silently* is an availability incident. Every block now emits
a `guard.block` span event, and control cases are first-class test fixtures, not an afterthought.

---

## Incident 2: auto-mitigation that reacted to its own actions

**Component:** `AdOpsAgent.step` (`agents/ad_economy/bandit_agent.py`), the auto-rollback and quarantine
controller.
**Severity, had it shipped:** high. A healthy demand partner gets disabled, which costs revenue and
trust.

### Symptom

Running the 72-hour incident timeline across five seeds surfaced three distinct misbehaviours:

| Seed | What happened | Why it was wrong |
|---|---|---|
| 21 | applovin was **quarantined at h59**, one hour after admob was restored | applovin was healthy; it was only winning fewer auctions because admob was back |
| 5, 99 | During the market-wide −40% drop at h64, admob alone was **quarantined at h65** | Its bids happened to fall below the $6 floor more often than the others' |
| (earlier build) | The bad $17 floor push depressed fill on *every* network; the agent tried to quarantine them all, the "keep ≥ 2 networks" guard refused, and the **control loop crashed** with `RuntimeError: quarantine_network failed` | Per-network alarms were treated as independent outages |

### Root causes

1. **Self-inflicted baseline shift.** Per-network detectors learned "normal" win share while admob was
   quarantined. Restoring admob, the agent's own action, moved every share, and the detector read that
   as an outage.
2. **No systemic-vs-local distinction.** A share drop on one network while *total* revenue is also
   breaching is the market moving, not that network failing.
3. **Refused actions raised.** A guard refusal surfaced as an exception that killed the loop, instead
   of an escalation.

### Fix (deterministic): the guard rules in `AdOpsAgent.step`

| Rule | Behaviour |
|---|---|
| Re-baseline | Every config change (floor, quarantine, restore, rollback) re-baselines the per-network detectors with a short warm-up. The zero-fill "dark" rule stays armed throughout. |
| Systemic drop | If more than one network alarms in the same hour, no network is quarantined. The total-revenue logic decides instead. |
| Total breach | While total revenue is breaching, only a true zero-fill outage can quarantine. |
| Rollback | Rollback to last-known-good happens only if the active config is a *trial* changed within 8h, a known-good config exists, and the cooldown has passed. Otherwise the agent escalates to a human. |
| Refusals | Guard refusals use `call_record`, are audited, and escalate. The loop never crashes. |

**Reproduction traces** (seed 5, previously a false-quarantine seed), each with exactly the intended
actions:

- one rollback at h35: `442d19f7bfac2bb824c45e2e881135a3`;
- one admob quarantine at h44: `072a1ae09463a522a51a96e771eaf69b` (and the re-quarantine after the failed
  h50 half-open probe, at h51: `01496a387b6d015b9898892a00a762fe`);
- escalation, not quarantine, at h65;
- `false_quarantines: []`.

Across all five seeds, 0 false quarantines.

### Regression test

`tests/test_phase3.py::test_guarded_ops_scenario` asserts that the only quarantined network is admob,
that the rollback falls in h34–36, and that the market drop is escalated, not rolled back.

### Lesson

Any auto-mitigation must model its own actions as a regime change. A detector that can't tell "I
changed the system" from "the system broke" will eventually page on itself, or worse, act on itself.

---

## Incident 3: telemetry lost the trace of the incident it was recording

**Component:** `DuckDBSpanExporter` (`core/harness/telemetry.py`).
**Severity, had it shipped:** medium. It doesn't change behaviour, but it destroys the evidence.

### Symptom

While writing *this* document, one of the four drill trace IDs printed by `make faults`,
**`34349e7c556b4b3772ef6931f9c7666d`** (the h10 admob quarantine), **returned 0 spans** from
`traces.duckdb`. The h17 quarantine from the same run was present. Only 96 of roughly 120 drill spans
had been written, and the exporter reported nothing wrong.

### Root cause

Two defects combined:

1. **Concurrent flushers raced.** The background flush thread and a forced flush could both write at
   once, and both ran `CREATE TABLE IF NOT EXISTS` on first use. DuckDB raised `TransactionException:
   Catalog write-write conflict`. A stress test (2 producers × 3,000 spans plus 50 forced flushes)
   reproduced **1 flush failure per trial**. The batch was re-queued and an exponential backoff
   started.
2. **Shutdown gave up after one attempt.** At exit, `close()` ran one forced flush. If that attempt hit
   a transient failure, it returned with spans still buffered. The daemon thread then died with the
   process, and the spans were gone.

### Fix (deterministic)

- **Single flusher.** A dedicated flush lock means a background tick *skips* if a flush is already
  running, and a forced flush *waits* for it.
- **Shutdown retries:** five attempts with backoff.
- **Spill instead of drop.** Anything still unwritten goes to `<db>.spool.jsonl`, which the next
  exporter ingests before its first write. This is counted in `stats["spilled"]` and
  `stats["unspooled"]`.

**Evidence after the fix** (`scripts/repro_incidents.py`, incident 3):

- 6,000/6,000 spans in the database, with 0 flush failures, 0 dropped and 0 spilled.
- Every trace ID cited in the README resolves.

### Regression tests

- `tests/test_phase4.py::test_exporter_concurrent_flushes_never_conflict`
- `tests/test_phase4.py::test_exporter_spills_at_shutdown_and_reingests`

### Lesson

The observability path needs the same failure-mode design as the product path. A missing trace looks
exactly like "nothing happened".

---

## 4. Failure classes prevented by design, proven by drills

These are the four guardrails named in the brief. Run each one live in the Control Center (**⚡ Fault
drills**) or with `make faults`. Trace IDs are from the run in `docs/fault_drills.json`.

| Failure class | Deterministic guardrail | Drill evidence | Trace |
|---|---|---|---|
| LLM proposes an unsolvable level with hallucinated words | **`agents/level_factory/solver.py`** Trie+DFS proof, with every witness re-checked by the referee. Then the release-gate approver (no proof means no publish), then `publish_version` re-solving at the DB layer. | Claimed words `planet, zebra, quartz` absent; target 20 > 12 words; publish blocked by `L3_privilege:approval_denied`; rotation unchanged | `f6219563d74599478b99539d4b560dfb` |
| Shipping a null variant because someone peeked | **mSPRT** (`stats_engine.MSPRT`): always-valid p and intersected CIs, plus the copilot ship gate | A fixed-horizon test hits p = 0.033 on day 8; mSPRT p = 0.49; FPR 19% → 1.7% over 300 A/A runs | `f618fb7aa8af977a2d78ce747443ab82` |
| Untrusted text drives a write | **`PolicyGuard`** (L1: allowlist, 20% rollout cap, SQL parser, realpath, exact domains), with L2 redaction before it and L3 approval behind it | The attack works only with every layer off. It's redacted by L2, and blocked again by the L1 cap with L2 removed. | `893d446d72a7e7422bef80056ca47266` |
| A dead demand partner keeps the auction waiting | **auto_rollback / quarantine** (`AdOpsAgent.step`): EWMA dark rule → circuit open → half-open probes, plus guarded last-known-good rollback | Quarantined the same hour; +45% post-outage revenue vs unguarded | `5e528b6cd66a331ab7251b33efae73dd` |

## 5. Still open

- **Soma extraction on unseen phrasing.** With held-out wording, 5 of 38 cases that should submit reach
  the carrier portal **with a wrong full-time headcount at confidence ≥ 0.80**. This is the unsafe
  failure mode: the confidence gate didn't catch it.
- **Root cause:** narrow rule vocabulary ("12 people full time" doesn't match), plus context-based
  attribution leaking across multi-question replies.
- **Status: open, deliberately.** Tuning the rules against the held-out set would turn it into another
  training score. The next steps:
  1. enable the verified-quote LLM extractor;
  2. require two independent sources for headcount before it counts as confident;
  3. measure both on a **fresh** held-out set.
