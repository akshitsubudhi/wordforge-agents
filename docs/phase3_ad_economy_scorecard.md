# Project 3 - Ad Auction Simulator & Monetization/UA Bandit Agent


| check | result | detail |
|---|---|---|
| shading matches closed form | PASS | 4.2e-07 |
| revenue equivalence holds (|z| < 3) | PASS | z=-1.51 |
| header bidding beats strategic waterfall on eCPM | PASS | +31.4% |
| header bidding latency lower than waterfall | PASS |  |
| 1st-price shaded ~= 2nd price (within 3%) | PASS |  |
| TS finds the oracle floor | PASS |  |
| TS regret < 25% of uniform | PASS |  |
| bad floor rolled back within 2h | PASS | h35 |
| dark network quarantined within 1h | PASS | h44 |
| market drop escalated, not rolled back | PASS |  |
| no false quarantines | PASS |  |
| guarded revenue > unguarded | PASS |  |
| LinTS beats uniform | PASS |  |
| LinTS >= 90% of oracle | PASS |  |

_15.2s, seeded and deterministic._

## 1. Bid shading  beta(v) = v - integral_r^v F(x)^(N-1) dx / F(v)^(N-1)

| check | result |
|---|---|
| U[0,1], N=5: numeric beta vs closed form (N-1)/N v, max abs error | 4.17e-07 |
| beta(0.8) numeric / closed form | 0.64 / 0.64 |
| Revenue equivalence, log-normal, N=5, reserve $4: 1st-price shaded eCPM | 10.66 |
| ... 2nd-price eCPM | 10.67 |
| ... difference (z-score) | -0.0116 (z = -1.51, 100,000 auctions) |

## 2. Mediation mechanisms (5 networks, log-normal demand, 600 ms timeout)

| mechanism | requests | fill | eCPM (per request) | eCPM (filled) | latency mean ms | latency p95 ms | allocative efficiency | bidder surplus CPM |
|---|---|---|---|---|---|---|---|---|
| waterfall | 40000 | 0.9657 | 10.56 | 10.94 | 687.1 | 1500 | 0.8147 | 4.11 |
| waterfall_strategic | 40000 | 0.918 | 7.694 | 8.381 | 1081 | 1735 | 0.7657 | 6.459 |
| hb_second | 40000 | 0.9999 | 10.11 | 10.11 | 238.7 | 354.5 | 1 | 5.298 |
| hb_first_truthful | 40000 | 0.9997 | 15.4 | 15.4 | 239.2 | 354.6 | 1 | 0 |
| hb_first_shaded | 40000 | 0.9999 | 10.11 | 10.11 | 239 | 352.8 | 1 | 5.313 |

- `waterfall` lets networks accept any tier priced at or below their value. That is a coarse descending (Dutch) auction with truthful bidders, so its revenue is an UPPER BOUND that rational networks would not leave on the table.
- `waterfall_strategic`: networks shade exactly as in first price (Dutch is strategically equivalent to first price). This is the fair comparison.
- `hb_first_truthful` shows why bidders shade: the publisher captures all of the surplus, and the bidders keep none.

**Header bidding (2nd price) vs strategic waterfall: eCPM +31.4%, latency 4.5x lower, efficiency 100.0% vs 76.6%.**

## 3. Thompson-sampling reserve floor (NIG posterior, 150 rounds x 400 requests)

| floor CPM | true eCPM/request (oracle, 40k each) | posterior mean | requests allocated |
|---|---|---|---|
| 0 | 10.11 | 10.06 | 4800 |
| 2 | 10.1 | 10.13 | 12000 |
| 4 | 10.11 | 10.04 | 4400 |
| 6 | 10.14 | 10.15 | 36000 |
| 8 | 10.07 | 9.864 | 1200 |
| 10 | 9.593 | 9.575 | 800 |
| 12 | 8.622 | 8.877 | 400 |
| 14 | 7.293 | 7.163 | 400 |

Oracle floor **$6**, TS posterior best **$6**; 73% of the last 20% of rounds on the oracle arm. Regret $3.161 vs $38.361 for uniform exploration.

## 4. Guarded operations: 72h incident timeline (3,000 requests/hour)

| incident | hour | detected / acted | action | revenue in window: guarded | unguarded |
|---|---|---|---|---|---|
| bad floor push ($6 -> $17) | 34 | 35 | guarded rollback to last-known-good v2 | 276.4 | 158.3 |
| admob goes dark (hangs to timeout) | 44 | 44 | quarantine; half-open probes; restore on recovery | 353 | 137.4 |
| admob recovers | 56 | 58 | probe succeeded -> network restored |  |  |
| market-wide eCPM -40% | 64 | 65 | NO rollback (no recent change) -> escalate to human | 137.2 | 19.32 |

72h revenue: guarded **$1,981.46** vs unguarded $1,441.66 (**+37.4%**). Config history: `ad_mediation_config`; actions: `agent_audit`.

## 5. Contextual LTV bandit for UA budget (LinTS, 60 days x $20k/day)

D1 LTV per install is MEASURED from seeded telemetry (ad revenue in the first 48h per geo tier x platform, empirical-Bayes shrunk). CPI levels and channel-quality multipliers are stated ASSUMPTIONS, because the Phase 1 seeder assigned acquisition channels independently of player behaviour.

| context | players | measured D1 LTV $ |
|---|---|---|
| T1/ios | 439 | 0.0666 |
| T1/android | 944 | 0.07059 |
| T2/ios | 226 | 0.02382 |
| T2/android | 467 | 0.02165 |
| T3/ios | 137 | 0.01566 |
| T3/android | 287 | 0.01358 |

| policy | spend $ | installs | D1 revenue $ | D1 ROAS |
|---|---|---|---|---|
| lints | 1200000 | 1776821 | 4.633e+04 | 0.03861 |
| uniform | 1200000 | 1842296 | 4.017e+04 | 0.03347 |
| oracle | 1200000 | 1792469 | 4.739e+04 | 0.03949 |

| context | oracle-best channel | LinTS share of last-7-day spend on it |
|---|---|---|
| T1/ios | meta | 0.923 |
| T1/android | applovin | 0.811 |
| T2/ios | meta | 1 |
| T2/android | applovin | 0.844 |
| T3/ios | meta | 0.957 |
| T3/android | applovin | 0.879 |

LinTS ROAS is **+15.4% vs uniform** and reaches **97.8% of the oracle**; regret $1,054 over 60 days. Every allocation was committed through the harness tool `allocate_ua_budget` (L1 currency cap: $5,000 per call).
