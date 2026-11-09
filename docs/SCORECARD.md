# Scorecard

| metric | value | what it means |
|---|---|---|
| Critic ↔ golden agreement | **96.0%** | Cohen's κ 0.92, defect recall 1.0; 100 levels, labels by construction |
| Red-team block rate | **100% (25/25)** | 0/10 false alarms; best single layer alone blocks 9/25 |
| p95 agent task latency | **47 ms** | 948 traced tasks of this scorecard run; tool-call p95 37.9 ms (traces.duckdb; wall-clock, so it varies with the machine) |
| LLM USD cost / task | **$0.00033** | mean over 185 LLM-driven runs at mock pricing; hard cap $0.25/run; workflows spend $0 |
| CUPED variance reduction | **7.1%** | ρ=0.26 on the seeded cohort (8,429→7,774 users); ≈ρ² as theory says |
| D1 retention: guardrail catch | **-4.7pp caught** | harmful variant rolled back on day 2; shipped variant D1 +0.9pp (always-valid CI spans 0: not claimed) |
| ARPDAU lift: guarded bandit ops | **+37.4%** | 72h incident timeline vs unguarded; TS floor alone +0.5% vs $2 default; UA ROAS +15.4% |
| Soma ACORD extraction F1 | **99.8%** | held-out phrasing 91.7% (portal fill 8/38); handoff recall 100% |

## Level taste

| feature | correlation with observed fail rate (12 seeded levels) |
|---|---|
| target_ratio | 0.846 |
| seconds_per_required_word | -0.679 |
| words_on_board | -0.374 |

Critic on 150 fresh candidates: 88 approved (59%).

| top rejection reason | count |
|---|---|
| lexical_quality: blocklisted words formable on board | 37 |
| board_quality: letter spam | 9 |
| lexical_quality: 71% of words are 3 letters | 3 |
| board_quality: vowel share 62% outside 20-60% | 3 |
| lexical_quality: 73% of words are 3 letters | 3 |
| board_quality: vowel share 69% outside 20-60% | 2 |
