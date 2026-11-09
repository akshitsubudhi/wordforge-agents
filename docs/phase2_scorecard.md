# Phase 2 Scorecard

_Generated 2026-11-09, 39.7s, LLM: deterministic mock (set ANTHROPIC_API_KEY or OPENAI_API_KEY and LLM_PROVIDER=auto for a real provider)._

| check | result | detail |
|---|---|---|
| P1 scan flags level 7 | PASS | Level 7 has a 63% fail rate (median of other levels 23%, robust z 3.7) |
| P1 CUPED reduces required sample | PASS | 8429 -> 7774 |
| P1 50% exposure blocked by L1 | PASS |  |
| P1 unplanned flag denied at approval gate | PASS |  |
| P1 DB CHECK rejects 50% | PASS |  |
| P1 harmful variant auto-rolled back | PASS | day 2 |
| P1 good variant shipped after approval | PASS |  |
| P1 CUPED variance reduction > 0 | PASS | 7.06% |
| P1 mSPRT FPR <= alpha under peeking | PASS | 0.0125 |
| P1 naive peeking inflates FPR | PASS | 0.2125 |
| P2 difficulty model calibrated MAE < 0.10 | PASS | 0.0636 |
| P2 critic kappa >= 0.6 | PASS | 0.92 |
| P2 critic catches >= 95% of injected defects | PASS | 1.0 |
| P2 daily levels shipped | PASS | 3 levels |
| GAME level 7 now serves the shipped version | PASS | v3 |
| GAME daily levels in rotation | PASS |  |
| GAME daily level playable | PASS | EHSD/SRUS/IPTR/ILEV |
| P2 spiking daily level retired | PASS |  |

## Project 1 - LiveOps Experimentation Copilot

### 1. Funnel scan (active versions, robust z vs other levels)

| level | version | attempts | fail rate | robust z | anomaly |
|---|---|---|---|---|---|
| 1 | 1 | 4445 | 0.0499 | -1.81 |  |
| 2 | 1 | 4532 | 0.0759 | -1.57 |  |
| 3 | 1 | 4552 | 0.118 | -1.2 |  |
| 4 | 1 | 4429 | 0.1567 | -0.85 |  |
| 5 | 1 | 4406 | 0.1773 | -0.53 |  |
| 6 | 1 | 4392 | 0.2293 | -0.16 |  |
| 7 | 1 | 6441 | 0.6271 | 3.7 | **YES** |
| 8 | 1 | 3976 | 0.251 | 0.15 |  |
| 9 | 1 | 3985 | 0.274 | 0.31 |  |
| 10 | 1 | 3921 | 0.2948 | 0.46 |  |
| 11 | 1 | 3874 | 0.326 | 0.9 |  |
| 12 | 1 | 3905 | 0.3457 | 1.08 |  |

> Level 7 has a 63% fail rate (median of other levels 23%, robust z 3.7)

### 2. Baselines (measured from telemetry) and power analysis at 20% exposure

| quantity | value |
|---|---|
| primary metric | level completions in the 7 days after exposure |
| baseline mean / sigma | 3.9224 / 5.1414 |
| CUPED covariate rho (pre vs post) | 0.2787 |
| D1 retention baseline | 0.4452 |
| MDE (relative) / alpha / power | 10% / 0.05 / 0.80 |
| treatment share (cap) | 20% |
| users needed WITHOUT CUPED | 8429 |
| users needed WITH CUPED | 7774 |
| eligible users per day (assumed production) | 4000 |
| planned duration (days) | 2 |
| D1 guardrail: users to detect -2pp | 30180 |

### 3. Exposure cap and Human Approval Gate

| request | rollout | outcome | enforced by |
|---|---|---|---|
| level7_trivial (planned) | 20% | approved | approval ledger: simulated_operator |
| level7_trivial_wide | 50% | BLOCKED | L1_allowlist:rollout_cap |
| level7_unplanned | 15% | DENIED | L3_privilege:approval_denied |
| raw SQL bypass | 50% | rejected: ConstraintException | DuckDB CHECK (rollout_pct <= 20) |

### 4. Scenario B - too-easy variant hurts D1 retention -> automatic rollback

Candidate level 7 v2 (`HGYS/DEDN/LRAE/YTNO`, target 4/43). Simulated truth: D1 -6pp, primary -3%.

| day | n treat | n ctrl | D1 delta | D1 always-valid CI | guardrail breached |
|---|---|---|---|---|---|
| 1 | 839 | 3161 | -0.04196 | [-0.1028, +0.0189] | no |
| 2 | 1620 | 6380 | -0.04684 | [-0.0887, -0.0050] | **YES** |

**Action:** `rolled_back` on day 2 - D1 guardrail harmed: always-valid CI [-0.0886, -0.0050] below -0.0 on day 2

### 5. Scenario A - easier level 7 (goal fail 0.35) -> CUPED + mSPRT -> approved ship

Candidate level 7 v3 (`LDMR/RNOH/EICD/HAIN`, target 7/15, predicted fail 0.3521). Simulated truth: primary +12%, D1 +1pp.

| day | n treat | n ctrl | CUPED delta | always-valid 95% CI | likelihood ratio | always-valid p | D1 delta |
|---|---|---|---|---|---|---|---|
| 1 | 853 | 3147 | 0.1691 | [-0.4732, +0.8113] | 0.603 | 1 | -0.00865 |
| 2 | 1654 | 6346 | 0.3511 | [-0.1006, +0.8028] | 4.042 | 0.2474 | 0.01015 |
| 3 | 2514 | 9486 | 0.2979 | [-0.0640, +0.6598] | 5.095 | 0.1963 | 0.00572 |
| 4 | 3316 | 12684 | 0.3312 | [+0.0136, +0.6487] | 29.33 | 0.03409 | 0.0044 |
| 5 | 4150 | 15850 | 0.4166 | [+0.1281, +0.6487] | 2531 | 0.000395 | 0.01099 |
| 6 | 4931 | 19069 | 0.3961 | [+0.1290, +0.6487] | 4621 | 0.000216 | 0.01005 |
| 7 | 5745 | 22255 | 0.4538 | [+0.2018, +0.6487] | 5.985e+05 | 2e-06 | 0.0089 |

**CUPED at the final peek**

|  | raw | CUPED |
|---|---|---|
| effect (completions / user / 7d) | 0.4418 | 0.4539 |
| Var(effect estimator) | 0.007133 | 0.006629 |
| p-value (fixed horizon) | 1.683e-07 | 2.488e-08 |
| 95% CI | [+0.2763, +0.6074] | [+0.2943, +0.6134] |

theta = 0.14554, rho = 0.264406, **variance reduction 7.06%**, relative lift +11.49%.

**Action:** `shipped` on day 7 - primary lift +11.5% significant (always-valid p=1.671e-06) after 7 days; guardrail OK

### 6. Peeking safety (A/A simulation, 400 experiments x 14 daily peeks)

| procedure | false-positive rate | nominal alpha |
|---|---|---|
| fixed-horizon z-test, single look at the end | 0.0675 | 0.05 |
| fixed-horizon z-test, stop at first p<0.05 (peeking) | 0.2125 | 0.05 |
| mSPRT always-valid, stop at first p<=0.05 | 0.0125 | 0.05 |

_0.7s_

### 7. Approval ledger (`approvals` table)

| tool | args | decision | by | reason |
|---|---|---|---|---|
| configure_flag | {"flag": "level7_trivial", "level_id": 7, "treatment_version": 2, "rol | approved | simulated_operator | planned experiment within exposure policy |
| configure_flag | {"flag": "level7_unplanned", "level_id": 7, "treatment_version": 2, "r | denied | simulated_operator | flag not in the approved experiment plan |
| configure_flag | {"flag": "level7_easier", "level_id": 7, "treatment_version": 3, "roll | approved | simulated_operator | planned experiment within exposure policy |
| ship_experiment | {"flag": "level7_easier"} | approved | simulated_operator | primary lift +11.5% significant (always-valid p=1.671e-06) after 7 days; guardrail OK |

## Project 2 - Autonomous Level Factory

### 1. Difficulty model calibration (observed vs predicted per-attempt fail rate)

| level | version | attempts | observed | predicted | abs error |
|---|---|---|---|---|---|
| 1 | 1 | 4445 | 0.0499 | 0.07 | 0.0201 |
| 2 | 1 | 4532 | 0.0759 | 0.1068 | 0.0309 |
| 3 | 1 | 4552 | 0.118 | 0.1459 | 0.0279 |
| 4 | 1 | 4429 | 0.1567 | 0.2013 | 0.0446 |
| 5 | 1 | 4406 | 0.1773 | 0.2263 | 0.049 |
| 6 | 1 | 4392 | 0.2293 | 0.2599 | 0.0306 |
| 7 | 1 | 6441 | 0.6271 | 0.5837 | 0.0434 |
| 8 | 1 | 3976 | 0.251 | 0.1445 | 0.1065 |
| 9 | 1 | 3985 | 0.274 | 0.1672 | 0.1068 |
| 10 | 1 | 3921 | 0.2948 | 0.2057 | 0.0891 |
| 11 | 1 | 3874 | 0.326 | 0.2123 | 0.1137 |
| 12 | 1 | 3905 | 0.3457 | 0.2275 | 0.1182 |

Attempt-weighted MAE 0.2344 (priors) -> **0.0636** (calibrated). Params: `{"b0": -4.4635, "b_zipf": 0.7917, "b_len": 0.02, "b_bf": 1.5151, "sigma": 1.7167}`. Caveat: the seeded fail rates were set per level in Phase 1, not produced by board features, so this measures fit to that data, not real-player validity.

### 2. Critic vs 100-level golden dataset

Labels: **construction** (`data/golden_levels.jsonl`; add `data/golden_levels_human.csv` with `id,label` to evaluate against human labels instead).

| metric | value |
|---|---|
| agreement | 96.0% |
| Cohen's kappa | 0.92 |
| precision (reject = defect) | 0.92 |
| recall (reject = defect) | 1 |
| F1 (reject) | 0.9583 |
| precision (accept) | 1 |
| recall (accept) | 0.9259 |

|  | critic: reject | critic: accept |
|---|---|---|
| label: reject (defect) | 46 | 0 |
| label: accept (clean) | 4 | 50 |

| defect type | n | critic agrees |
|---|---|---|
| clean | 54 | 50/54 |
| duplicate_board | 6 | 6/6 |
| letter_spam | 6 | 6/6 |
| malformed_grid | 4 | 4/4 |
| missing_required_word | 3 | 3/3 |
| offensive_word | 6 | 6/6 |
| rare_letter_overload | 4 | 4/4 |
| too_easy | 6 | 6/6 |
| too_hard | 6 | 6/6 |
| unsolvable_target | 5 | 5/5 |

Disagreements (critic stricter than the construction label):

- G024 clean: lexical_quality: 80% of words are 3 letters
- G033 clean: lexical_quality: 100% of words are 3 letters
- G038 clean: lexical_quality: 83% of words are 3 letters
- G040 clean: lexical_quality: 75% of words are 3 letters

### 3. Daily ship (2026-11-09)

| stage | count |
|---|---|
| generated | 170 |
| solver verified | 170 |
| in difficulty band | 162 |
| critic approved | 80 |
| shipped | 3 |

| level id | slot | grid | target / words | time | goal | predicted fail | critic scores | source |
|---|---|---|---|---|---|---|---|---|
| 13 | daily-easy | `EHSD/SRUS/IPTR/ILEV` | 6/22 | 110 | 0.2 | 0.1982 | {'solvability_fairness': 5, 'difficulty_fit': 5, 'lexical_quality': 5, 'board_quality': 5} | heuristic |
| 14 | daily-medium | `WASN/AKAL/TEWI/EIID` | 8/18 | 100 | 0.32 | 0.3189 | {'solvability_fairness': 5, 'difficulty_fit': 5, 'lexical_quality': 5, 'board_quality': 5} | heuristic |
| 15 | daily-hard | `ULOTU/OOSOU/AESHU/DGPNN/TOHER` | 22/38 | 110 | 0.42 | 0.4005 | {'solvability_fairness': 5, 'difficulty_fit': 5, 'lexical_quality': 5, 'board_quality': 5} | heuristic |

_1.4s, every publish approved by the release gate (solver proof + critic in ledger)_

Release-gate decisions: {'approved': 5}

## Live game wiring (real FastAPI app, in-process)

| level | version | created by | daily | target | time | fail rate (this version) | predicted |
|---|---|---|---|---|---|---|---|
| 1 | 1 | seed | False | 6 | 120 | 0.05 |  |
| 2 | 1 | seed | False | 7 | 120 | 0.076 |  |
| 3 | 1 | seed | False | 8 | 120 | 0.118 |  |
| 4 | 1 | seed | False | 9 | 110 | 0.157 |  |
| 5 | 1 | seed | False | 9 | 105 | 0.177 |  |
| 6 | 1 | seed | False | 9 | 100 | 0.229 |  |
| 7 | 3 | level_factory | False | 7 | 90 |  | 0.3521 |
| 8 | 1 | seed | False | 12 | 120 | 0.251 |  |
| 9 | 1 | seed | False | 12 | 120 | 0.274 |  |
| 10 | 1 | seed | False | 12 | 115 | 0.295 |  |
| 11 | 1 | seed | False | 13 | 110 | 0.326 |  |
| 12 | 1 | seed | False | 13 | 110 | 0.346 |  |
| 13 | 1 | level_factory | True | 6 | 110 |  | 0.1982 |
| 14 | 1 | level_factory | True | 8 | 100 |  | 0.3189 |
| 15 | 1 | level_factory | True | 22 | 110 |  | 0.4005 |

### Auto-rollback on live fail-rate spike

| level | version | predicted | observed | attempts | spike | action |
|---|---|---|---|---|---|---|
| 7 | 3 | 0.3521 |  | 0 | False | none |
| 13 | 1 | 0.1982 |  | 0 | False | none |
| 14 | 1 | 0.3189 |  | 0 | False | none |
| 15 | 1 | 0.4005 | 0.7833 | 60 | True | retired |

Level 15 in rotation after monitor: **False**

## Harness telemetry for this run

19 workflows, 364 tool calls, all exported to `data/traces.duckdb` (`make metrics`).

| tool | calls | ok | blocked by guard | p50 ms | p95 ms |
|---|---|---|---|---|---|
| baseline_metrics | 1 | 1 | 0 | 39.17 | 39.17 |
| configure_flag | 4 | 2 | 2 | 9.93 | 10.02 |
| critique_level | 162 | 162 | 0 | 2.67 | 6.85 |
| evaluate_experiment | 9 | 9 | 0 | 69.44 | 158.9 |
| generate_candidates | 5 | 5 | 0 | 113.3 | 657.6 |
| live_level_health | 1 | 1 | 0 | 14.66 | 14.66 |
| power_analysis | 1 | 1 | 0 | 0.71 | 0.71 |
| propose_easier_level | 2 | 2 | 0 | 278.5 | 278.5 |
| prove_level | 170 | 170 | 0 | 1.07 | 2.03 |
| publish_level | 5 | 5 | 0 | 11.21 | 15.11 |
| retire_level | 1 | 1 | 0 | 8.12 | 8.12 |
| rollback_experiment | 1 | 1 | 0 | 6.69 | 6.69 |
| scan_funnel | 1 | 1 | 0 | 17.29 | 17.29 |
| ship_experiment | 1 | 1 | 0 | 12.2 | 12.2 |
