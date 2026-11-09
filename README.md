# LiveOps Agent Platform

**A playable word game whose live telemetry drives five production-style AI agents, all running on one
guarded agent harness.** It's a single Python process (FastAPI, DuckDB and vanilla JS) with no Docker and
no API key needed. Without a key, every agent runs on a deterministic mock LLM, and every number below
is reproducible offline.

```bash
make setup && make run          # → http://localhost:8000  (self-seeds 580k events on first start, ~10 s)
```

In the first 60 seconds you can:

- **Inject production faults:** on the Overview page, click **⚡ Inject all 4 production faults** and watch
  each guardrail fire, with a trace ID you can open.
- **Ship a level:** in **Level factory**, click **Generate, verify & push level to live game**, then
  **Play it now**.
- **Run an experiment:** in **LiveOps copilot**, click **Launch experiment**. *You* are the approval gate:
  the job pauses until you click Approve.

## Production & evaluation scorecard

Regenerate everything with `make scorecard` (about 2 minutes; stop the server first). The source is
[`docs/scorecard.json`](docs/scorecard.json).

| Metric | Value | What it actually measures |
|---|---|---|
| **Eval agreement**: critic vs golden set | **96.0%** (Cohen's κ 0.92) | 4-rubric level critic vs 100 labelled levels. It catches 46/46 injected defects; all 4 disagreements are the critic being stricter (boards where most words are 3 letters). Labels come from construction, not human raters. |
| **Red-team block rate** | **100%** (25/25), 0/10 false alarms | Guardrail conformance suite through the real runner. Ablation: the best *single* defence layer blocks only 9/25. |
| **p95 agent task latency** | **35 ms** | p95 of 1,891 traced agent tasks in `traces.duckdb` (tool-call p95 29 ms). Deterministic workflows plus the mock LLM, so a real model adds its own latency. |
| **USD cost per task** | **$0.00033** | Mean LLM spend over 368 LLM-driven runs, at mock pricing. Deterministic workflows spend $0. A hard pre-call cap stops any run at $0.25. |
| **CUPED variance reduction** | **7.1%** | ρ = 0.26 between week-1 and week-2 completions in the seeded cohort, so ≈ρ² as theory predicts (8,429 → 7,774 users needed). It's modest because the seeder front-loads play; a ρ = 0.77 synthetic check gives 60%. |
| **D1 retention** | **−4.7pp harm caught on day 2** | A deliberately harmful variant was auto-rolled back when its always-valid CI fell below 0. The shipped variant's D1 was +0.9pp with a CI spanning 0, so **no lift is claimed**. |
| **ARPDAU bandit lift** | **+37.4%** | The guarded ad-ops agent vs the same 72h incident timeline unguarded. The Thompson-sampled floor *alone* adds only +0.5% over a $2 default, and the UA bandit improves ROAS by +15.4% vs a uniform split. |
| **Soma ACORD extraction F1** | **99.8%** dev / **91.7%** held-out | 50 synthetic multi-channel submissions. With held-out phrasing only 8/38 reach the portal: the gate sends 25 to humans and 5 slip through with a wrong headcount. See [Limitations](#limitations). |

![Control Center: four injected faults, each auto-mitigated](docs/control_center_faults.png)

## Four production faults, injected live

The **⚡ Fault drills** page (or `make faults` from a terminal) injects a real fault into the real
component. Nothing is mocked except the fault itself.

| # | Fault | Guardrail that fires | Result | Trace (run that wrote this README) |
|---|---|---|---|---|
| 1 | LLM hallucinates an unsolvable level: claims "planet, zebra, quartz" and asks for 20 words; the board has 12 | `solver.py` Trie+DFS proof → 4-rubric critic → release gate (L3 `approval_denied`) → `publish_version` re-solve | Stopped at independent gates; live rotation unchanged | `f6219563d74599478b99539d4b560dfb` |
| 2 | Ad network goes dark (hangs until timeout) | EWMA dark rule → `quarantine_network` (circuit open) → half-open probes | Quarantined in the same hour; post-outage revenue +45% vs unguarded | `5e528b6cd66a331ab7251b33efae73dd` |
| 3 | Peeking false positive: an A/A test peeked daily | mSPRT always-valid p → copilot ship gate | A naive t-test would ship on day 8 (p = 0.033); mSPRT p = 0.49. FPR 19% → 1.7% over 300 A/A runs | `f618fb7aa8af977a2d78ce747443ab82` |
| 4 | Indirect prompt injection: a support ticket carries a tool directive to set a flag to 100% | L2 spotlight redaction → L1 `PolicyGuard` 20% cap → L3 approval | Executes only with defences off; L2 redacts it, and L1 blocks it if L2 is removed | `893d446d72a7e7422bef80056ca47266` |

Trace IDs resolve in `data/traces.duckdb` on the machine that ran them (`SELECT * FROM spans WHERE trace_id = …`).
`make faults` produces fresh ones in `docs/fault_drills.json`.

## Architecture tour

```mermaid
flowchart LR
    G[LetterLoop game<br/>server referee] -->|events| T[(DuckDB<br/>telemetry)]
    T --> C[P1 LiveOps Copilot]
    T --> F[P2 Level Factory]
    T --> A[P3 Ad economy agent]
    S[Email / Twilio SMS / Voice] --> I[P5 Soma intake]
    C -->|candidate version + ≤20% flag| G
    F -->|release-gated publish| G
    I -->|label-mapped form fill| P[Mock carrier portal]
    subgraph H[P4 SRE agent harness: every agent runs here]
        direction TB
        V[schema · L1 PolicyGuard · L3 PrivilegeGate] --> R[breaker · timeout · retry] --> L2[L2 spotlight] --> TR[OTel spans → DuckDB]
    end
    C & F & A & I -.tool calls.-> H
    CC[Evaluator Control Center] --> C & F & A & I & G
```

Each block: how it's built, and the one-line hook that stops it failing.

| Block | How it's built | Mitigation hook |
|---|---|---|
| **Game referee** `app/game_api.py` | Client sends tile coordinates only; the server rebuilds the word, checks adjacency and the 2,500-word lexicon, keeps the clock, and emits every event | Forged or late words are rejected; an early "timeout" is recorded as "quit" |
| **Telemetry** `core/db.py`, `scripts/seed_telemetry.py` | Hot properties as real columns, the long tail as a JSON string. The Cookie Cats marginals are hit to the integer | Single-writer lock gives a clear error with the PID and exit code 2 |
| **Harness** `core/harness/runner.py` | Fixed per-call pipeline, plus a no-LLM `workflow()` mode for deterministic steps | Unknown tool or extra field: a non-switchable checkpoint blocks it |
| **Defences** `core/harness/redteam_guard.py` | L1: allowlist plus argument rules (SQL parser, realpath, exact domains, 20% cap). L2: normalise, decode, redact, nonce boundaries. L3: READ_ONLY tokens, taint → approval, copied-argument block, DLP, output filter | Conformance gate in CI: 25/25 blocked, 0/10 false alarms |
| **Resilience** `core/harness/resilience.py` | Full-jitter backoff, circuit breaker, worst-case pre-call cost check, `python -I -S` sandbox with rlimits and an audit hook | Spend can never exceed the cap; retries happen only on idempotent tools |
| **Telemetry spans** `core/harness/telemetry.py` | contextvars nesting; batched exporter behind one flush lock; OTLP/JSON export | A failed final flush spills to a spool file that the next run ingests (see the postmortem) |
| **Stats engine** `agents/liveops_copilot/stats_engine.py` | Fleiss / CUPED-adjusted power, CUPED, mSPRT | Always-valid inference: peeking cannot inflate false positives |
| **Copilot** `agents/liveops_copilot/agent.py` | Robust-z scan → power → factory variant → flag → daily replayed peeks | 20% cap enforced 3×; human gate; automatic rollback when the guardrail CI falls below 0 |
| **Solver** `agents/level_factory/solver.py` | Trie-pruned DFS proof; every witness path re-checked by the referee | No proof means no publish |
| **Difficulty + critic** `difficulty_model.py`, `critic_and_eval.py` | Poisson-binomial player model calibrated on telemetry; 4 rubrics including an offensive-word trie | Release gate requires proof + critic in the ledger; a live spike retires the level |
| **Ad economy** `agents/ad_economy/` | BNE-shaded auctions; NIG Thompson sampling; EWMA detectors; LinTS UA allocator | Quarantine a single dark network; roll back only after a recent change; otherwise escalate |
| **Soma intake** `agents/soma_intake/` | Channel normalisers (Twilio HMAC); event-sourced SQLite; rule extractor with provenance; label-driven portal bot | Replay after a crash, exactly-once; confidence < 0.80 after 3 follow-ups goes to a human |
| **Control Center** `app/dashboard.py`, `app/static/control_center.html` | Single-worker job queue in the server process; polled step logs; human-approval broker | Approvals fail closed after 120 s; jobs never contend for DuckDB |

Deeper diagrams and per-phase failure matrices are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Auction theory notes (Project 3)

- **Bid shading.** With N symmetric bidders, value CDF F and reserve r, the first-price Bayes-Nash
  equilibrium bid is

  ```
  β(v) = v − ∫_r^v F(x)^(N−1) dx  /  F(v)^(N−1)
  ```

  Each bidder shades its value down by the expected gap to the second-highest value, given that it wins.
  `BidShader` integrates this on a 4,000-point trapezoid grid for *any* F. It matches the uniform closed
  form, (N−1)/N · v, to **4×10⁻⁷**.
- **Revenue equivalence.** With symmetric i.i.d. bidders, the same allocation rule and the same reserve,
  first-price with BNE shading and second-price earn the same expected revenue. Measured over 100k
  log-normal auctions (N = 5, $4 reserve): **$10.66 vs $10.67 eCPM, z = −1.5**. This is the test that the
  shading code is right.
- **Waterfall vs header bidding.** A price-sorted waterfall is a descending (Dutch) auction, which is
  strategically equivalent to first price.
  - **Naive waterfall:** if networks accept any price at or below their value, it *out-earns* header
    bidding ($10.56). That's an upper bound no rational bidder would leave on the table.
  - **Strategic waterfall:** networks shade as in first price. Against it, second-price header bidding
    wins **+31.4% eCPM** with **4.5× lower latency**, and allocative efficiency rises from 77% to 100%.
- **Heterogeneous networks.** Each network shades against the *pooled* opponent CDF, the standard
  approximation when no closed-form asymmetric equilibrium exists.
- **Reserve floors.** Revenue vs floor is flat from $0 to $6 (within 0.3%), then falls off a cliff
  ($7.29 at $14). Thompson sampling with Normal-Inverse-Gamma posteriors finds $6, with regret **$3.16
  vs $38.36** for uniform exploration. Its value is avoiding the cliff, not the +0.5%.

## Experimentation notes (Project 1)

- **CUPED.** θ = cov(X, Y) / var(X), pooled over both arms, and Y′ = Y − θ(X − X̄). The variance
  reduction is ≈ ρ².
- **mSPRT.** With a normal mixture of variance τ² on the effect and V_n = Var(δ̂):

  ```
  Λ_n = sqrt(V_n / (V_n + τ²)) · exp( τ²·δ̂² / (2·V_n·(V_n + τ²)) )
  ```

  - The always-valid p-value is min(p, 1/Λ), and confidence intervals are intersected across peeks.
  - Λ is a non-negative martingale under the null, so P(ever ≥ 1/α) ≤ α by Ville's inequality.
  - Measured: naive peeking gives **21%** false positives; mSPRT gives **1.25%** (α = 5%).
- **Ship rule.** Ship only when (1) the mSPRT test rejects the null on the CUPED-adjusted primary
  metric, (2) at least 7 days have run, and (3) the D1 guardrail CI isn't below 0. Rollback needs no
  approval, because it's the safe direction.

## Level taste: what the data says

Data: 12 seeded levels, about 53k attempts, plus the critic's verdicts on 150 freshly generated
boards (`docs/scorecard.json → level_taste`).

| Level | Observed fail | Predicted | Target ratio | Seconds per required word | Words on board |
|---|---|---|---|---|---|
| 1 | 5.0% | 7.0% | 0.18 | 20.0 | 34 |
| 4 | 15.7% | 20.1% | 0.38 | 12.2 | 24 |
| **7** | **62.7%** | **58.4%** | **0.64** | 10.7 | **11** |
| 8 (5×5) | 25.1% | 14.4% | 0.30 | 10.0 | 40 |
| 12 (5×5) | 34.6% | 22.8% | 0.38 | 8.5 | 34 |

- **Target ratio drives difficulty.** The share of the board's words a player must find correlates
  **0.85** with fail rate. Seconds per required word comes next (**−0.68**), then raw word count (−0.37).
  Level 7 is hard because it asks for 7 of only 11 words, all of them rare, in 75 s.
- **The fitted model:** b_zipf = 0.79 (common words are found faster), b_bf = 1.51 (dense boards help),
  and a wide skill spread, σ = 1.72. The word-length effect is nearly zero.
- **5×5 is under-predicted by about 10pp.** The Phase 1 seeder set 5×5 fail rates by level index, not
  from board features, so this is fit to synthetic data. The live monitor that retires spiking levels
  is the backstop.
- **What the critic rejects.** It approves **59%** of fresh candidates. The #1 reason, **37 of 62
  rejections**, is an offensive word a player could *trace* on the board. Next come letter spam (9) and
  3-letter-heavy vocabularies. Lexical quality is the weakest rubric (mean 3.75/5); solvability is never
  the problem (5.0/5).

## Limitations

These are stated plainly, because an evaluator will find them anyway:

- **Synthetic data everywhere.** Cookie Cats marginals are exact, but the event cohort, experiment
  traffic, ad demand and the 50 Soma customers are simulated. Every simulator is seeded and every
  assumption is labelled in its scorecard.
- **Soma generalisation.** Extraction F1 falls from 99.8% to 91.7% on held-out phrasing.
  - The confidence gate fails safe 25 times, but **5 cases submit a wrong full-time headcount at
    confidence ≥ 0.80**.
  - The root causes are documented and deliberately not tuned against the held-out set. The next step
    is the verified-quote LLM extractor, measured on a *fresh* held-out set.
- **Golden labels** come from construction, not from people. Drop `data/golden_levels_human.csv` in
  place to re-score against human labels.
- **No real LLM has been exercised.** The Anthropic and OpenAI clients exist, but no API key was set
  during development.
- **Postmortem.** [docs/THREE_WAYS_MY_AGENT_BROKE.md](docs/THREE_WAYS_MY_AGENT_BROKE.md) covers the
  three real failures hit while building this, with reproducible trace IDs.

## Commands

| Command | What it does |
|---|---|
| `make run` | Server + Control Center on :8000 |
| `make test` | 69 tests across the 4 phases |
| `make check` | Tests + guardrail conformance (the CI gate) |
| `make faults` | The four fault drills from a terminal |
| `make phase2` / `make phase3` | Scorecards for Projects 1–2 and Projects 3 and 5 (server stopped) |
| `make scorecard` | Everything above, then rebuild `docs/scorecard.json` |

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Use a real model, with a circuit-breaker fallback to the mock |
| `LLM_MODEL` | Override the model ID |
| `LIVEOPS_DATA_DIR` | Database location (tests use temp directories) |
| `HARNESS_MAX_RUN_USD` | Hard per-run spend cap |
| `HARNESS_MAX_RUN_TOKENS` | Hard per-run token cap |
| `TWILIO_AUTH_TOKEN` / `TWILIO_WEBHOOK_URL` | SMS webhook signature verification. The default is a test token; set real values in production. |
| `data/cookie_cats.csv` | When present, the seeder loads the real Kaggle file instead of the calibrated synthetic copy |

DuckDB allows one writer per file. While the server is running, use the Control Center (or
`POST /api/admin/reseed`), not the CLI scripts.
