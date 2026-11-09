"""
================================================================================
BLOCK: Project 3 Scorecard (auction mechanisms, floor TS, guarded ops, UA bandit)
--------------------------------------------------------------------------------
Role & contracts
  Runs every Project-3 component end to end and prints Markdown tables
  (also written to --out, default data/reports/ad_economy.md):
    1 bid-shading validation (closed form + revenue equivalence)
    2 mechanism comparison: waterfall (naive / strategic) vs header bidding
      2nd price / 1st price truthful / 1st price BNE-shaded
    3 Thompson-sampling floor tuning vs brute-force oracle
    4 72h incident timeline: guarded agent vs the same incidents unguarded
    5 contextual LTV bandit (LinTS) vs uniform vs oracle UA allocation
  Writes ad_mediation_config / ad_hourly_metrics / ua_allocations /
  agent_audit in data/telemetry.duckdb (single writer: stop the server first).

Key engineering mechanism
  Seeded and deterministic; every agent action goes through the core
  harness (spans, guards, audit).

Failure modes & 1-step mitigation
  * DB locked -> exit 2 with the fix. Invariant broken -> exit 1.
================================================================================
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.ad_economy.auction_sim import (  # noqa: E402
    MediationSimulator, revenue_equivalence_check, uniform_closed_form_check,
)
from agents.ad_economy.bandit_agent import AdOpsAgent, run_ops_scenario, run_ua_bandit, tune_floor  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.db import DatabaseLockedError, TelemetryDB  # noqa: E402
from core.harness.telemetry import DuckDBSpanExporter, InMemoryExporter, Tracer  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def table(headers, rows) -> str:
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] +
                     ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser(description="Project 3 ad economy scorecard")
    ap.add_argument("--impressions", type=int, default=40_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-trace", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    try:
        db = TelemetryDB(s.telemetry_db)
    except DatabaseLockedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    tracer = Tracer(InMemoryExporter() if a.no_trace else DuckDBSpanExporter(s.traces_db), service_name="ad-economy")
    md = ["# Project 3 - Ad Auction Simulator & Monetization/UA Bandit Agent", ""]

    # 1 shading validation
    u = uniform_closed_form_check(5)
    re_ = revenue_equivalence_check(n_bidders=5, reserve=4.0, auctions=100_000)
    md += ["## 1. Bid shading  beta(v) = v - integral_r^v F(x)^(N-1) dx / F(v)^(N-1)", "",
           table(["check", "result"], [
               ["U[0,1], N=5: numeric beta vs closed form (N-1)/N v, max abs error", f"{u['max_abs_error']:.2e}"],
               ["beta(0.8) numeric / closed form", f"{u['beta(0.8)']} / {u['closed_form(0.8)']}"],
               ["Revenue equivalence, log-normal, N=5, reserve $4: 1st-price shaded eCPM", re_["first_price_shaded_cpm"]],
               ["... 2nd-price eCPM", re_["second_price_cpm"]],
               ["... difference (z-score)", f"{re_['difference']} (z = {re_['z']}, {re_['auctions']:,} auctions)"]]), ""]
    check("shading matches closed form", u["max_abs_error"] < 1e-4, f"{u['max_abs_error']:.1e}")
    check("revenue equivalence holds (|z| < 3)", abs(re_["z"]) < 3, f"z={re_['z']}")

    # 2 mechanisms
    sim = MediationSimulator(seed=1)
    reps = [sim.run(m, a.impressions, **kw) for m, kw in [
        ("waterfall", {}), ("waterfall_strategic", {}), ("hb_second", {"reserve": 2.0}),
        ("hb_first_truthful", {"reserve": 2.0}), ("hb_first_shaded", {"reserve": 2.0})]]
    md += ["## 2. Mediation mechanisms (5 networks, log-normal demand, 600 ms timeout)", "",
           table(["mechanism", "requests", "fill", "eCPM (per request)", "eCPM (filled)", "latency mean ms", "latency p95 ms",
                  "allocative efficiency", "bidder surplus CPM"], [r.row() for r in reps]), "",
           "- `waterfall` lets networks accept any tier priced at or below their value. That is a coarse descending (Dutch) "
           "auction with truthful bidders, so its revenue is an UPPER BOUND that rational networks would not leave on the table.",
           "- `waterfall_strategic`: networks shade exactly as in first price (Dutch is strategically equivalent to first price). "
           "This is the fair comparison.",
           "- `hb_first_truthful` shows why bidders shade: the publisher captures all of the surplus, and the bidders keep none.", ""]
    by = {r.mechanism: r for r in reps}
    uplift = by["hb_second"].ecpm / by["waterfall_strategic"].ecpm - 1
    check("header bidding beats strategic waterfall on eCPM", uplift > 0, f"+{uplift:.1%}")
    check("header bidding latency lower than waterfall", by["hb_second"].latency_mean_ms < by["waterfall_strategic"].latency_mean_ms)
    check("1st-price shaded ~= 2nd price (within 3%)", abs(by["hb_first_shaded"].ecpm / by["hb_second"].ecpm - 1) < 0.03)
    md += [f"**Header bidding (2nd price) vs strategic waterfall: eCPM {uplift:+.1%}, latency "
           f"{by['waterfall_strategic'].latency_mean_ms / by['hb_second'].latency_mean_ms:.1f}x lower, efficiency "
           f"{by['hb_second'].efficiency:.1%} vs {by['waterfall_strategic'].efficiency:.1%}.**", ""]

    # 3 floor TS
    floors = [0, 2, 4, 6, 8, 10, 12, 14]
    ts = tune_floor(MediationSimulator(seed=5), floors, rounds=150, batch=400)
    md += ["## 3. Thompson-sampling reserve floor (NIG posterior, 150 rounds x 400 requests)", "",
           table(["floor CPM", "true eCPM/request (oracle, 40k each)", "posterior mean", "requests allocated"],
                 [[p["floor"], ts["truth_ecpm_by_floor"][p["floor"]], p["mean_cpm"], p["pulls"]] for p in ts["posterior"]]), "",
           f"Oracle floor **${ts['oracle_floor']}**, TS posterior best **${ts['ts_floor']}**; {ts['last20pct_on_oracle_arm']:.0%} of the last "
           f"20% of rounds on the oracle arm. Regret ${ts['regret_usd']} vs ${ts['uniform_regret_usd']} for uniform exploration.", ""]
    check("TS finds the oracle floor", ts["ts_floor"] == ts["oracle_floor"])
    check("TS regret < 25% of uniform", ts["regret_usd"] < 0.25 * ts["uniform_regret_usd"])

    # 4 ops scenario
    g = run_ops_scenario(db, guard=True, tracer=tracer)
    ug = run_ops_scenario(db, guard=False, tracer=tracer)
    inc = g["incidents"]
    ev = g["events"]
    first = lambda typ, after: next((e["hour"] for e in ev if e["type"] == typ and e["hour"] >= after), None)
    rb, q, esc = first("rollback", inc["bad_floor_push"]), first("quarantine", inc["network_dark"]), first("escalate", inc["market_drop"])
    probe_ok = next((t["hour"] for t in g["timeline"] if any(isinstance(x, dict) and "probe_ok" in x for x in t["actions"])), None)
    window = lambda run, h0, h1: sum(t["ecpm"] for t in run["timeline"][h0:h1]) * 3000 / 1000
    md += ["## 4. Guarded operations: 72h incident timeline (3,000 requests/hour)", "",
           table(["incident", "hour", "detected / acted", "action", "revenue in window: guarded", "unguarded"], [
               ["bad floor push ($6 -> $17)", inc["bad_floor_push"], rb, "guarded rollback to last-known-good v2",
                round(window(g, 34, 44), 2), round(window(ug, 34, 44), 2)],
               ["admob goes dark (hangs to timeout)", inc["network_dark"], q, "quarantine; half-open probes; restore on recovery",
                round(window(g, 44, 58), 2), round(window(ug, 44, 58), 2)],
               ["admob recovers", inc["network_recovers"], probe_ok, "probe succeeded -> network restored", "", ""],
               ["market-wide eCPM -40%", inc["market_drop"], esc, "NO rollback (no recent change) -> escalate to human",
                round(window(g, 64, 72), 2), round(window(ug, 64, 72), 2)]]), "",
           f"72h revenue: guarded **${g['revenue_usd']:,.2f}** vs unguarded ${ug['revenue_usd']:,.2f} "
           f"(**{g['revenue_usd'] / ug['revenue_usd'] - 1:+.1%}**). Config history: `ad_mediation_config`; actions: `agent_audit`.", ""]
    check("bad floor rolled back within 2h", rb is not None and rb - inc["bad_floor_push"] <= 2, f"h{rb}")
    check("dark network quarantined within 1h", q is not None and q - inc["network_dark"] <= 1, f"h{q}")
    check("market drop escalated, not rolled back", esc is not None and not any(e["type"] == "rollback" and e["hour"] >= inc["market_drop"] for e in ev))
    check("no false quarantines", all(e.get("network") == "admob" for e in ev if e["type"] == "quarantine"))
    check("guarded revenue > unguarded", g["revenue_usd"] > ug["revenue_usd"])

    # 5 UA bandit
    agent = AdOpsAgent(db, MediationSimulator(seed=0), tracer=tracer, run_id="ua_run")
    ua = run_ua_bandit(db, agent, days=60)
    T = ua["totals"]
    md += ["## 5. Contextual LTV bandit for UA budget (LinTS, 60 days x $20k/day)", "",
           "D1 LTV per install is MEASURED from seeded telemetry (ad revenue in the first 48h per geo tier x platform, "
           "empirical-Bayes shrunk). CPI levels and channel-quality multipliers are stated ASSUMPTIONS, because the Phase 1 "
           "seeder assigned acquisition channels independently of player behaviour.", "",
           table(["context", "players", "measured D1 LTV $"], [[k, v["users"], v["ltv"]] for k, v in ua["ltv_measured"].items()]), "",
           table(["policy", "spend $", "installs", "D1 revenue $", "D1 ROAS"],
                 [[k, round(v["spend"]), v["installs"], round(v["revenue"], 2), v["roas"]] for k, v in T.items()]), "",
           table(["context", "oracle-best channel", "LinTS share of last-7-day spend on it"],
                 [[k, ua["best_channel"][k], v] for k, v in ua["last_week_share_on_best"].items()]), "",
           f"LinTS ROAS is **{T['lints']['roas'] / T['uniform']['roas'] - 1:+.1%} vs uniform** and reaches "
           f"**{T['lints']['roas'] / T['oracle']['roas']:.1%} of the oracle**; regret ${ua['regret_usd_vs_oracle']:,.0f} over 60 days. "
           "Every allocation was committed through the harness tool `allocate_ua_budget` (L1 currency cap: $5,000 per call).", ""]
    check("LinTS beats uniform", T["lints"]["roas"] > T["uniform"]["roas"])
    check("LinTS >= 90% of oracle", T["lints"]["roas"] >= 0.9 * T["oracle"]["roas"])

    head = ["", table(["check", "result", "detail"], [[n, "PASS" if ok else "**FAIL**", d] for n, ok, d in CHECKS]), "",
            f"_{time.perf_counter() - t0:.1f}s, seeded and deterministic._", ""]
    report = "\n".join(md[:2] + head + md[2:])
    print(report)
    out = a.out or s.data_dir / "reports" / "ad_economy.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    failed = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed -> {out}")
    db.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
