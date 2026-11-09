"""
================================================================================
BLOCK: Monetization & UA Bandit Agent (Project 3)
--------------------------------------------------------------------------------
Role & contracts
  ThompsonFloorTuner  learns the reserve floor (CPM) that maximises revenue
                      per ad request in the header-bidding auction.
  EWMADetector        flags revenue / fill anomalies per network and in total.
  AdOpsAgent          harness-wired controller. Every hour it observes the
                      auction, keeps versioned mediation configs in DuckDB
                      (`ad_mediation_config`), quarantines dark networks
                      (circuit breaker with half-open probes) and performs a
                      GUARDED rollback to the last-known-good config.
  ContextualLTVBandit linear Thompson sampling that allocates the daily UA
                      budget across (channel x geo-tier x platform) using
                      predicted D1 LTV, with ground truth measured from
                      telemetry.duckdb.
  Downstream: ad_mediation_config, ad_hourly_metrics, ua_allocations,
              agent_audit, and spans in traces.duckdb.

Key engineering mechanism
  * Floor TS: Normal-Inverse-Gamma posterior per floor arm over revenue per
    request (unknown mean AND variance). Each round: tau ~ Gamma(a_n, b_n),
    mu ~ N(m_n, 1/(k_n tau)), play the argmax, update with the batch's
    (n, mean, sum of squares).
  * EWMA: m_t = lam x_t + (1 - lam) m_(t-1), with an exponentially weighted
    variance. z = (x - m)/max(sd, floor). The alarm needs `confirm`
    consecutive z < -k breaches, and the baseline FREEZES during a breach so
    an outage is never learned as normal. Every config change re-baselines
    the per-network detectors (the competitive mix moved), while the
    zero-fill "dark" rule stays armed during warmup.
  * Guarded rollback (all must hold, otherwise the agent escalates instead):
      1 confirmed alarm on TOTAL revenue
      2 the active config is a TRIAL changed within `rollback_window_h`
        (no recent change means rolling back cannot help; the cause is
        external, e.g. market-wide eCPM, so it pages a human)
      3 a last-known-good config exists (status 'good' = survived
        `promote_after_h` clean hours)
      4 cooldown since the previous rollback has elapsed
    Rollback is a pre-approved WRITE (the safe direction) and is audited.
  * Outage rollback: a network whose fill drops to exactly 0 (dark) while a
    TRIAL floor is active triggers rollback_config(cause="network_dark")
    BEFORE the quarantine: the floor reverts to the last-known-good config
    (or the initial baseline if nothing was promoted yet), then the dark
    network is disabled. The trial floor was tuned against the full bidder
    set, so it is not trusted through an outage.
  * Quarantine: a network whose fill collapses to 0 (or z < -k on fills)
    WHILE the other networks stay healthy is disabled (several networks
    dropping together is systemic and is never quarantined, and while total
    revenue is breaching only a zero-fill outage counts); after `probe_every_h` hours it is re-enabled for ONE probe
    hour (half-open) and either stays restored or is quarantined again.
  * LinTS for UA: y = D1 ROAS % for a $ slice, features = one-hot(channel,
    tier, platform) plus channel x platform and channel x tier interactions
    (29 dims). Posterior N(B^-1 f, v^2 B^-1) with B = lam I + sum x x^T.
    Budget is allocated slice by slice (probability matching) under two
    guards: <= 50% of the day on one channel, <= $5,000 per allocation call
    (L1 currency cap).

Failure modes & 1-step mitigation
  * Bad floor push -> rolled back within `confirm` hours to the LKG config.
  * Dark network -> quarantined; re-probed automatically, restored on recovery.
  * External demand drop -> no rollback; escalation recorded in agent_audit.
  * UA bandit over-concentrating -> per-channel daily cap re-routes spend.
================================================================================
"""
from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass, field
from statistics import fmean
from typing import Sequence

from pydantic import Field

from agents.ad_economy.auction_sim import MediationSimulator
from agents.common import audit, utcnow
from core.config import get_settings
from core.db import TelemetryDB
from core.harness.llm import MockLLM
from core.harness.redteam_guard import READ_WRITE, AgentPolicy, ApprovalRequest, DefenseConfig
from core.harness.runner import AgentRunner, AgentSpec
from core.harness.telemetry import Tracer, default_tracer
from core.harness.tools import ArgKind, Privilege, StrictArgs, ToolContext, ToolRegistry, ToolSpec


# ============================================================================ Thompson sampling for floors
@dataclass
class NIGArm:
    m: float = 10.0
    k: float = 0.01
    a: float = 2.0
    b: float = 50.0
    n: int = 0

    def update(self, xs: Sequence[float]) -> None:
        n = len(xs)
        if not n:
            return
        mean = fmean(xs)
        ss = sum((x - mean) ** 2 for x in xs)
        kn = self.k + n
        self.b += 0.5 * ss + self.k * n * (mean - self.m) ** 2 / (2 * kn)
        self.m = (self.k * self.m + n * mean) / kn
        self.k, self.a, self.n = kn, self.a + n / 2, self.n + n

    def sample(self, rng: random.Random) -> float:
        tau = rng.gammavariate(self.a, 1 / self.b)
        return rng.gauss(self.m, 1 / math.sqrt(self.k * tau))


class ThompsonFloorTuner:
    def __init__(self, floors: Sequence[float], seed: int = 0):
        self.floors = list(floors)
        self.arms = [NIGArm() for _ in floors]
        self.rng = random.Random(seed)
        self.history: list[int] = []

    def choose(self) -> int:
        i = max(range(len(self.arms)), key=lambda j: self.arms[j].sample(self.rng))
        self.history.append(i)
        return i

    def update(self, arm: int, rewards: Sequence[float]) -> None:
        self.arms[arm].update(rewards)

    def best(self) -> float:
        return self.floors[max(range(len(self.arms)), key=lambda j: self.arms[j].m)]

    def posterior(self) -> list[dict]:
        return [{"floor": f, "mean_cpm": round(a.m, 3), "pulls": a.n} for f, a in zip(self.floors, self.arms)]


def revenue_per_request_samples(sim: MediationSimulator, reserve: float, n: int, enabled: set[str] | None = None) -> list[float]:
    nets = [x for x in sim.networks if enabled is None or x.name in enabled]
    return [o.price if o.filled else 0.0 for o in (sim.header_impression("hb_second", reserve, nets) for _ in range(n))]


def tune_floor(sim: MediationSimulator, floors: Sequence[float], rounds: int = 150, batch: int = 400,
               oracle_n: int = 40_000, seed: int = 0) -> dict:
    """TS vs oracle (brute-force evaluation of every arm). Rewards in CPM per request."""
    oracle_sim = MediationSimulator(sim.networks, seed=seed + 999)
    truth = {f: fmean(revenue_per_request_samples(oracle_sim, f, oracle_n)) for f in floors}
    best_f = max(truth, key=truth.get)
    ts = ThompsonFloorTuner(floors, seed)
    regret, curve = 0.0, []
    for r in range(rounds):
        i = ts.choose()
        ts.update(i, revenue_per_request_samples(sim, floors[i], batch))
        regret += (truth[best_f] - truth[floors[i]]) * batch / 1000  # USD lost vs oracle
        if (r + 1) % max(1, rounds // 10) == 0:
            curve.append({"round": r + 1, "cum_regret_usd": round(regret, 3), "posterior_best": ts.best()})
    last = ts.history[-rounds // 5:]
    share_best = sum(floors[i] == best_f for i in last) / len(last)
    return {"truth_ecpm_by_floor": {f: round(v, 3) for f, v in truth.items()}, "oracle_floor": best_f,
            "ts_floor": ts.best(), "regret_usd": round(regret, 3), "curve": curve, "posterior": ts.posterior(),
            "last20pct_on_oracle_arm": round(share_best, 3), "impressions": rounds * batch,
            "uniform_regret_usd": round(sum(truth[best_f] - truth[f] for f in floors) / len(floors) * rounds * batch / 1000, 3)}


# ============================================================================ EWMA detector
@dataclass
class Signal:
    value: float
    mean: float | None
    z: float | None
    breach: bool
    alarm: bool
    reason: str = ""


class EWMADetector:
    def __init__(self, lam: float = 0.25, k: float = 3.0, warmup: int = 8, confirm: int = 2, sd_floor_frac: float = 0.04,
                 seen_positive: bool = False):
        self.lam, self.k, self.warmup, self.confirm, self.sd_floor_frac = lam, k, warmup, confirm, sd_floor_frac
        self.n, self.mean, self.var, self.streak = 0, 0.0, 0.0, 0
        self._warm: list[float] = []
        # carried across re-baselines: a series that has EVER been positive and reads exactly 0 is an outage, even
        # when the 0 is the very first sample of a fresh warmup (the hour of a config change) or the warmup was
        # all zeros (fixes: a network going dark in the same hour as a floor push was never detected)
        self.seen_positive = seen_positive

    def update(self, x: float) -> Signal:
        dark = x == 0 and self.seen_positive
        self.seen_positive = self.seen_positive or x > 0
        if self.n < self.warmup:
            if dark:  # outages are caught even while (re)warming; an outage sample never enters the baseline
                prior = [v for v in self._warm if v > 0]
                return Signal(x, fmean(prior) if prior else None, None, True, True, "dark")
            self._warm.append(x)
            self.n += 1
            if self.n == self.warmup:
                self.mean = fmean(self._warm)
                self.var = sum((v - self.mean) ** 2 for v in self._warm) / max(1, self.warmup - 1)
            return Signal(x, None, None, False, False, "warmup")
        sd = max(math.sqrt(self.var), self.sd_floor_frac * abs(self.mean), 1e-9)
        z = (x - self.mean) / sd
        breach = z < -self.k
        self.streak = self.streak + 1 if breach else 0
        alarm = self.streak >= self.confirm or dark
        if not breach and not dark:  # freeze the baseline while anomalous
            d = x - self.mean
            self.mean += self.lam * d
            self.var = (1 - self.lam) * (self.var + self.lam * d * d)
        self.n += 1
        return Signal(x, round(self.mean, 6), round(z, 2), breach, alarm, "dark" if dark else ("confirmed" if alarm else ""))


# ============================================================================ harness-wired ad ops agent
class HourArgs(StrictArgs):
    hour: int = Field(ge=0)
    requests: int = Field(ge=1, le=1_000_000)


class FloorArgs(StrictArgs):
    reserve_cpm: float = ArgKind("usd", ge=0, le=50)
    reason: str = Field(max_length=200)


class NetArgs(StrictArgs):
    network: str = Field(pattern=r"^[a-z_]{3,30}$")
    reason: str = Field(max_length=200)


class RollbackArgs(StrictArgs):
    reason: str = Field(max_length=300)
    # revenue_alarm: guarded by trial status + cooldown + an existing last-known-good config.
    # network_dark: an outage invalidates whatever floor is on trial (it was tuned against the full bidder set);
    #               reverts to the last-known-good config, or to the INITIAL baseline if nothing was promoted yet.
    cause: str = Field(default="revenue_alarm", pattern="^(revenue_alarm|network_dark)$")


class UAArgs(StrictArgs):
    day: int = Field(ge=0)
    channel: str = Field(pattern=r"^[a-z_]{3,30}$")
    context: str = Field(pattern=r"^[A-Za-z0-9_/]{3,30}$")
    usd: float = ArgKind("usd", ge=0)
    policy: str = Field(default="lints", max_length=20)


AD_TOOLS = frozenset({"observe_hour", "apply_floor", "quarantine_network", "restore_network", "rollback_config",
                      "allocate_ua_budget"})


class AdOpsAgent:
    SYSTEM = ("You are the Ad Monetization & UA agent. Tune floors, watch revenue anomalies, quarantine dark networks, "
              "roll back bad configs only when guarded conditions hold, and allocate UA budget within caps.")

    def __init__(self, db: TelemetryDB, sim: MediationSimulator, *, run_id: str | None = None, tracer: Tracer | None = None,
                 defenses: DefenseConfig = DefenseConfig(), max_floor_step: float = 12.0, promote_after_h: int = 6,
                 rollback_window_h: int = 8, cooldown_h: int = 3, probe_every_h: int = 6, guard_enabled: bool = True):
        self.db, self.sim = db, sim
        self.run_id = run_id or uuid.uuid4().hex[:10]
        self.max_floor_step, self.promote_after_h = max_floor_step, promote_after_h
        self.rollback_window_h, self.cooldown_h, self.probe_every_h = rollback_window_h, cooldown_h, probe_every_h
        self.guard_enabled = guard_enabled
        self.hour = 0
        self.clean_hours = 0
        self.last_change_h = -10**9
        self.last_rollback_h = -10**9
        self.quarantined: dict[str, int] = {}     # network -> hour quarantined
        self.probing: str | None = None
        self.events: list[dict] = []
        self.total_det = EWMADetector(k=3.0, confirm=2)
        self.net_det = {n.name: EWMADetector(k=3.5, confirm=2) for n in sim.networks}
        self.runner = AgentRunner(
            AgentSpec("ad_ops", self.SYSTEM, AgentPolicy(AD_TOOLS, get_settings().data_dir, max_usd_per_action=5_000)),
            self._registry(), MockLLM("compliant"), defenses=defenses, tracer=tracer or default_tracer(),
            approver=lambda req: (False, "no approver for ad ops"))
        with db.cursor() as con:
            con.execute("DELETE FROM ad_mediation_config")
            con.execute("INSERT INTO ad_mediation_config VALUES (1, ?, 'hb_second', 2.0, '[]', 'trial', NULL, 'initial')", [utcnow()])

    # ------------------------------------------------------------------ config store
    def active(self) -> dict:
        with self.db.cursor() as con:
            r = con.execute("""SELECT version, reserve_cpm, disabled, status FROM ad_mediation_config
                               WHERE status IN ('trial', 'good') ORDER BY version DESC LIMIT 1""").fetchone()
        return {"version": r[0], "reserve_cpm": r[1], "disabled": json.loads(r[2]), "status": r[3]}

    def baseline(self) -> dict:
        """The config a rollback restores: the last-known-good one, or the initial config if none was promoted."""
        with self.db.cursor() as con:
            r = con.execute("""SELECT version, reserve_cpm FROM ad_mediation_config
                               ORDER BY (status = 'good') DESC, CASE WHEN status = 'good' THEN -version ELSE version END
                               LIMIT 1""").fetchone()
        return {"version": r[0], "reserve_cpm": r[1]}

    def _rebaseline(self) -> None:
        """A config change shifts every network's share of wins: re-learn per-network baselines (short warmup)
        so our own action is never mistaken for an outage. The dark (zero-fill) rule stays armed throughout."""
        self.net_det = {n.name: EWMADetector(k=3.5, confirm=2, warmup=4,
                                             seen_positive=self.net_det[n.name].seen_positive if n.name in self.net_det else False)
                        for n in self.sim.networks}

    def _new_version(self, reserve: float, disabled: list[str], reason: str) -> int:
        self._rebaseline()
        cur = self.active()
        with self.db.cursor() as con:
            v = con.execute("SELECT max(version) + 1 FROM ad_mediation_config").fetchone()[0]
            con.execute("UPDATE ad_mediation_config SET status='superseded' WHERE version=? AND status='trial'", [cur["version"]])
            con.execute("INSERT INTO ad_mediation_config VALUES (?, ?, 'hb_second', ?, ?, 'trial', ?, ?)",
                        [v, utcnow(), reserve, json.dumps(sorted(disabled)), cur["version"], reason])
        self.clean_hours = 0
        return v

    # ------------------------------------------------------------------ tools
    def _t_observe(self, a: HourArgs, ctx: ToolContext) -> dict:
        cfg = self.active()
        enabled = {n.name for n in self.sim.networks} - set(cfg["disabled"])
        nets = [n for n in self.sim.networks if n.name in enabled]
        per = {n.name: {"requests": 0, "fills": 0, "revenue": 0.0, "timeouts": 0} for n in self.sim.networks}
        total_rev = 0.0
        filled = 0
        for _ in range(a.requests):
            o = self.sim.header_impression("hb_second", cfg["reserve_cpm"], nets)
            for n in nets:
                per[n.name]["requests"] += 1
            if o.filled:
                filled += 1
                total_rev += o.price / 1000
                per[o.winner]["fills"] += 1
                per[o.winner]["revenue"] += o.price / 1000
        with self.db.cursor() as con:
            con.executemany("INSERT INTO ad_hourly_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            [[self.run_id, a.hour, k, cfg["version"], v["requests"], v["fills"], round(v["revenue"], 6), v["timeouts"]]
                             for k, v in per.items()])
        return {"hour": a.hour, "config_version": cfg["version"], "reserve_cpm": cfg["reserve_cpm"],
                "disabled": cfg["disabled"], "requests": a.requests, "fill_rate": filled / a.requests,
                "ecpm": total_rev / a.requests * 1000, "revenue_usd": total_rev, "per_network": per}

    def _t_floor(self, a: FloorArgs, ctx: ToolContext) -> dict:
        cur = self.active()
        if abs(a.reserve_cpm - cur["reserve_cpm"]) > self.max_floor_step:
            raise ValueError(f"floor step {cur['reserve_cpm']} -> {a.reserve_cpm} exceeds max step {self.max_floor_step}")
        v = self._new_version(a.reserve_cpm, cur["disabled"], a.reason)
        self.last_change_h = self.hour
        audit(self.db, "ad_ops", "apply_floor", version=v, reserve=a.reserve_cpm, reason=a.reason, hour=self.hour)
        return {"version": v, "reserve_cpm": a.reserve_cpm}

    def _t_quarantine(self, a: NetArgs, ctx: ToolContext) -> dict:
        cur = self.active()
        if a.network in cur["disabled"]:
            return {"noop": True}
        if len(cur["disabled"]) + 1 >= len(self.sim.networks) - 1:
            raise ValueError("refusing to quarantine: fewer than 2 networks would remain")
        v = self._new_version(cur["reserve_cpm"], cur["disabled"] + [a.network], a.reason)
        audit(self.db, "ad_ops", "quarantine_network", network=a.network, version=v, reason=a.reason, hour=self.hour)
        return {"version": v, "disabled": cur["disabled"] + [a.network]}

    def _t_restore(self, a: NetArgs, ctx: ToolContext) -> dict:
        cur = self.active()
        v = self._new_version(cur["reserve_cpm"], [n for n in cur["disabled"] if n != a.network], a.reason)
        audit(self.db, "ad_ops", "restore_network", network=a.network, version=v, reason=a.reason, hour=self.hour)
        return {"version": v}

    def _t_rollback(self, a: RollbackArgs, ctx: ToolContext) -> dict:
        cur = self.active()
        if cur["status"] != "trial":
            raise ValueError("guard: active config is already known-good; nothing to roll back")
        outage = a.cause == "network_dark"
        if not outage and self.hour - self.last_rollback_h < self.cooldown_h:
            raise ValueError("guard: rollback cooldown active")
        with self.db.cursor() as con:
            lkg = con.execute("SELECT version, reserve_cpm, disabled FROM ad_mediation_config WHERE status='good' "
                              "ORDER BY version DESC LIMIT 1").fetchone()
            if not lkg and outage:  # nothing promoted yet: the initial config IS the baseline
                lkg = con.execute("SELECT version, reserve_cpm, disabled FROM ad_mediation_config "
                                  "ORDER BY version LIMIT 1").fetchone()
            if not lkg:
                raise ValueError("guard: no last-known-good config")
            # keep current quarantines: rolling back must not re-enable a dead network
            disabled = sorted(set(json.loads(lkg[2])) | set(cur["disabled"]))
            con.execute("UPDATE ad_mediation_config SET status='rolled_back', reason=concat(reason, ?) WHERE version=?",
                        [f" | rolled back: {a.reason}", cur["version"]])
            v = con.execute("SELECT max(version) + 1 FROM ad_mediation_config").fetchone()[0]
            con.execute("INSERT INTO ad_mediation_config VALUES (?, ?, 'hb_second', ?, ?, 'good', ?, ?)",
                        [v, utcnow(), lkg[1], json.dumps(disabled), cur["version"], f"rollback to v{lkg[0]}"])
        self.last_rollback_h = self.hour
        self.clean_hours = 0
        self._rebaseline()
        audit(self.db, "ad_ops", "rollback_config", from_version=cur["version"], to_version=v, lkg=lkg[0], reason=a.reason,
              cause=a.cause, hour=self.hour)
        return {"from_version": cur["version"], "to_version": v, "restored_reserve_cpm": lkg[1], "lkg": lkg[0],
                "from_reserve_cpm": cur["reserve_cpm"], "cause": a.cause}

    def _t_ua(self, a: UAArgs, ctx: ToolContext) -> dict:
        return {"day": a.day, "channel": a.channel, "context": a.context, "usd": a.usd, "accepted": True}

    def _registry(self) -> ToolRegistry:
        R, W = Privilege.READ, Privilege.WRITE
        return ToolRegistry([
            ToolSpec("observe_hour", "Run one hour of header-bidding traffic and record metrics", HourArgs, self._t_observe, R, timeout_s=60),
            ToolSpec("apply_floor", "Create a trial config with a new reserve floor", FloorArgs, self._t_floor, W, idempotent=False),
            ToolSpec("quarantine_network", "Disable a network (circuit open)", NetArgs, self._t_quarantine, W),
            ToolSpec("restore_network", "Re-enable a network (probe / recovery)", NetArgs, self._t_restore, W),
            ToolSpec("rollback_config", "Guarded rollback to the last-known-good config", RollbackArgs, self._t_rollback, W),
            ToolSpec("allocate_ua_budget", "Commit UA spend for a channel/context", UAArgs, self._t_ua, W, idempotent=False),
        ])

    # ------------------------------------------------------------------ hourly control loop
    def step(self, hour: int, requests: int) -> dict:
        self.hour = hour
        with self.runner.workflow(f"adops:h{hour}", mode=READ_WRITE) as wf:
            obs = wf.call("observe_hour", hour=hour, requests=requests)
            cfg_status = self.active()["status"]
            actions = []
            if not self.guard_enabled:
                return {**obs, "actions": actions, "signals": {}}
            tot = self.total_det.update(obs["ecpm"])
            signals = {"total": tot}
            # per-network: fill share among requests sent to it (only networks currently in the auction)
            dark_found = []
            for name, m in obs["per_network"].items():
                if name in obs["disabled"]:
                    continue
                s = self.net_det[name].update(m["fills"] / max(1, m["requests"]))
                signals[name] = s
                if s.alarm:
                    dark_found.append((name, s))
            if tot.breach:
                # total revenue is down too: a share-only (non-zero) drop on one network is the market moving,
                # not that network failing. Only a true zero-fill outage justifies quarantine now.
                dark_found = [d for d in dark_found if d[1].reason == "dark"]
            if len(dark_found) > 1:
                # several networks degrading in the same hour is SYSTEMIC (config or market), not a network
                # outage: never quarantine on it; the total-revenue logic below decides
                audit(self.db, "ad_ops", "systemic_network_drop", hour=hour, networks=[d[0] for d in dark_found])
                dark_found = []
            # half-open probe result
            if self.probing:
                name = self.probing
                self.probing = None
                if obs["per_network"][name]["fills"] == 0:
                    actions.append(wf.call("quarantine_network", network=name, reason=f"probe at h{hour} failed: still dark"))
                    self.quarantined[name] = hour
                else:
                    actions.append({"probe_ok": name})
                dark_found = [d for d in dark_found if d[0] != name]
            outage = [d for d in dark_found if d[1].reason == "dark"]
            if outage and cfg_status == "trial":
                base = self.baseline()
                cur_reserve = obs["reserve_cpm"]
                if abs(cur_reserve - base["reserve_cpm"]) > 1e-9:
                    # AUTOMATIC ROLLBACK TO THE BASELINE FLOOR: the floor on trial was tuned against the full bidder
                    # set; with a network dark it is unproven, so the outage is handled on the known-good floor.
                    rb = wf.call_record("rollback_config", cause="network_dark",
                                        reason=f"{outage[0][0]} went dark at h{hour} while floor ${cur_reserve:g} was on trial")
                    if rb.ok:
                        actions.append(rb.raw)
                        self.total_det = EWMADetector(k=3.0, confirm=2)
                        self.events.append({"hour": hour, "type": "rollback", "cause": "network_dark",
                                            "from_reserve": cur_reserve, "to_reserve": rb.raw["restored_reserve_cpm"]})
                    else:
                        audit(self.db, "ad_ops", "rollback_refused", hour=hour, cause="network_dark")
                        self.events.append({"hour": hour, "type": "escalate", "why": "outage rollback refused by guard"})
            for name, s in dark_found:
                base_txt = f"{s.mean:.3f}" if s.mean is not None else "n/a"
                rec = wf.call_record("quarantine_network", network=name,
                                     reason=f"fill share {s.value:.3f} vs baseline {base_txt} (z={s.z}, {s.reason})")
                if rec.ok:
                    actions.append(rec.raw)
                    self.quarantined[name] = hour
                    self.events.append({"hour": hour, "type": "quarantine", "network": name})
                else:  # guard refused (e.g. too few networks left): escalate instead of crashing the loop
                    audit(self.db, "ad_ops", "quarantine_refused", hour=hour, network=name)
                    self.events.append({"hour": hour, "type": "escalate", "why": f"quarantine of {name} refused by guard"})
            for name, qh in list(self.quarantined.items()):
                if hour - qh >= self.probe_every_h and self.probing is None and name not in [d[0] for d in dark_found]:
                    actions.append(wf.call("restore_network", network=name, reason=f"half-open probe after {hour - qh}h"))
                    self.probing = name
                    del self.quarantined[name]
                    self.events.append({"hour": hour, "type": "probe", "network": name})
            cfg_status = self.active()["status"]  # an outage rollback / quarantine above may have changed it
            if tot.alarm and not dark_found:
                recent = hour - self.last_change_h <= self.rollback_window_h
                if cfg_status == "trial" and recent and hour - self.last_rollback_h >= self.cooldown_h:
                    actions.append(wf.call("rollback_config", reason=f"eCPM {obs['ecpm']:.2f} vs baseline {tot.mean:.2f} (z={tot.z}) "
                                                                     f"{hour - self.last_change_h}h after config change"))
                    self.total_det = EWMADetector(k=3.0, confirm=2)
                    self.events.append({"hour": hour, "type": "rollback"})
                else:
                    why = ("no config change in the last %dh: external cause" % self.rollback_window_h) if not recent else \
                          ("active config already known-good" if cfg_status != "trial" else "rollback cooldown")
                    audit(self.db, "ad_ops", "escalate_to_human", hour=hour, ecpm=obs["ecpm"], z=tot.z, why=why)
                    if not self.events or self.events[-1].get("type") != "escalate":
                        self.events.append({"hour": hour, "type": "escalate", "why": why})
            elif not tot.breach:
                self.clean_hours += 1
                if cfg_status == "trial" and self.clean_hours >= self.promote_after_h:
                    with self.db.cursor() as con:
                        con.execute("UPDATE ad_mediation_config SET status='good' WHERE version=?", [obs["config_version"]])
            return {**obs, "actions": actions, "signals": {k: v.__dict__ for k, v in signals.items()}}

    def apply_floor(self, reserve: float, reason: str) -> dict:
        with self.runner.workflow(f"adops:floor:h{self.hour}", mode=READ_WRITE) as wf:
            rec = wf.call_record("apply_floor", reserve_cpm=reserve, reason=reason)
        self.events.append({"hour": self.hour, "type": "floor_change", "reserve": reserve, "ok": rec.ok})
        return {"ok": rec.ok, "result": rec.raw, "blocked": rec.decision.rule if rec.decision and not rec.decision.allowed else None}


# ============================================================================ contextual LTV bandit for UA
CHANNELS = ("google_uac", "meta", "tiktok", "applovin")
TIERS = ("T1", "T2", "T3")
PLATFORMS = ("ios", "android")
COUNTRY_TIER = {"US": "T1", "GB": "T1", "DE": "T1", "JP": "T1", "BR": "T2", "ZZ": "T2", "IN": "T3"}
CONTEXT_SHARE = {("T1", "ios"): .30, ("T1", "android"): .20, ("T2", "ios"): .05, ("T2", "android"): .15,
                 ("T3", "ios"): .05, ("T3", "android"): .25}
# ---- stated market ASSUMPTIONS (the seed assigned channels independently of behaviour) ----
BASE_CPI = {"T1": 2.40, "T2": 0.80, "T3": 0.30}
CHANNEL_CPI = {"google_uac": 1.00, "meta": 1.15, "tiktok": 0.70, "applovin": 0.90}
CHANNEL_QUALITY = {("google_uac", "ios"): 1.05, ("google_uac", "android"): 1.10, ("meta", "ios"): 1.30,
                   ("meta", "android"): 1.05, ("tiktok", "ios"): 0.55, ("tiktok", "android"): 0.75,
                   ("applovin", "ios"): 1.00, ("applovin", "android"): 1.25}


def measured_d1_ltv(db: TelemetryDB, prior_strength: float = 30.0) -> dict[tuple[str, str], dict]:
    """D1 LTV per install = ad revenue in the first 48h, per (tier, platform), measured from seeded telemetry,
    empirical-Bayes shrunk toward the platform mean."""
    with db.cursor() as con:
        rows = con.execute("""
            WITH first AS (SELECT user_id, min(ts) AS t0 FROM events WHERE source='seed' AND event_name='session_start' GROUP BY 1)
            SELECT p.country, p.platform, count(DISTINCT p.user_id) AS users,
                   coalesce(sum(e.revenue_usd), 0) AS rev
            FROM players p JOIN first f USING (user_id)
            LEFT JOIN events e ON e.user_id = p.user_id AND e.event_name = 'ad_impression' AND e.ts < f.t0 + INTERVAL 48 HOUR
            WHERE p.source = 'seed' GROUP BY 1, 2""").fetchall()
    cell: dict[tuple[str, str], list[float]] = {}
    for country, plat, users, rev in rows:
        c = cell.setdefault((COUNTRY_TIER.get(country, "T2"), plat), [0, 0.0])
        c[0] += users
        c[1] += rev
    plat_mean = {p: sum(v[1] for k, v in cell.items() if k[1] == p) / max(1, sum(v[0] for k, v in cell.items() if k[1] == p))
                 for p in PLATFORMS}
    out = {}
    for t in TIERS:
        for p in PLATFORMS:
            users, rev = cell.get((t, p), [0, 0.0])
            shrunk = (rev + prior_strength * plat_mean[p]) / (users + prior_strength)
            out[(t, p)] = {"users": users, "raw": round(rev / users, 5) if users else None, "ltv": round(shrunk, 5)}
    return out


def features(channel: str, tier: str, platform: str) -> list[float]:
    x = [1.0 if channel == c else 0.0 for c in CHANNELS]
    x += [1.0 if tier == t else 0.0 for t in TIERS]
    x += [1.0 if platform == p else 0.0 for p in PLATFORMS]
    x += [1.0 if (channel, platform) == (c, p) else 0.0 for c in CHANNELS for p in PLATFORMS]
    x += [1.0 if (channel, tier) == (c, t) else 0.0 for c in CHANNELS for t in TIERS]
    return x


def _cholesky(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(max(s, 1e-12)) if i == j else s / L[j][j]
    return L


def _inverse(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        pv = m[c][c]
        m[c] = [v / pv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c]
                m[r] = [rv - f * cv for rv, cv in zip(m[r], m[c])]
    return [row[n:] for row in m]


class ContextualLTVBandit:
    """Linear Thompson sampling over ROAS% with daily posterior refresh."""

    def __init__(self, dim: int, lam: float = 1.0, v: float = 1.0, seed: int = 0):
        self.d, self.v = dim, v
        self.B = [[lam if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        self.f = [0.0] * dim
        self.rng = random.Random(seed)
        self.refresh()

    def refresh(self) -> None:
        self.Binv = _inverse(self.B)
        self.mu = [sum(self.Binv[i][j] * self.f[j] for j in range(self.d)) for i in range(self.d)]
        self.L = _cholesky([[self.v ** 2 * x for x in row] for row in self.Binv])

    def sample_theta(self) -> list[float]:
        z = [self.rng.gauss(0, 1) for _ in range(self.d)]
        return [self.mu[i] + sum(self.L[i][k] * z[k] for k in range(i + 1)) for i in range(self.d)]

    def observe(self, x: list[float], y: float, weight: float = 1.0) -> None:
        for i in range(self.d):
            if x[i]:
                self.f[i] += weight * x[i] * y
                for j in range(self.d):
                    if x[j]:
                        self.B[i][j] += weight * x[i] * x[j]

    def predict(self, x: list[float]) -> float:
        return sum(m * xi for m, xi in zip(self.mu, x))


def run_ua_bandit(db: TelemetryDB, agent: AdOpsAgent | None = None, days: int = 60, daily_budget: float = 20_000,
                  slice_usd: float = 500, channel_cap: float = 0.5, seed: int = 0, ltv_cv: float = 2.5) -> dict:
    ltv = measured_d1_ltv(db)
    truth = {(c, t, p): ltv[(t, p)]["ltv"] * CHANNEL_QUALITY[(c, p)] / (BASE_CPI[t] * CHANNEL_CPI[c] * (1.35 if p == "ios" else 1.0))
             for c in CHANNELS for t in TIERS for p in PLATFORMS}  # true D1 ROAS (revenue per $)
    rng = random.Random(seed)
    contexts = list(CONTEXT_SHARE)
    slices_per_day = int(daily_budget / slice_usd)

    def outcome(c: str, t: str, p: str, spend: float) -> tuple[int, float]:
        cpi = BASE_CPI[t] * CHANNEL_CPI[c] * (1.35 if p == "ios" else 1.0) * math.exp(rng.gauss(0, 0.15))
        lam = spend / cpi
        installs = max(0, round(rng.gauss(lam, math.sqrt(lam))))
        mean_ltv = ltv[(t, p)]["ltv"] * CHANNEL_QUALITY[(c, p)]
        rev = max(0.0, rng.gauss(installs * mean_ltv, math.sqrt(max(installs, 1)) * ltv_cv * mean_ltv))
        return installs, rev

    def ctx_slices(day: int) -> list[tuple[str, str]]:
        r = random.Random(seed * 7919 + day)
        return [r.choices(contexts, weights=[CONTEXT_SHARE[c] for c in contexts])[0] for _ in range(slices_per_day)]

    policies = {"lints": ContextualLTVBandit(len(features(*("meta", "T1", "ios"))), lam=1.0, v=0.5, seed=seed),
                "uniform": None, "oracle": None}
    totals = {k: {"spend": 0.0, "revenue": 0.0, "installs": 0} for k in policies}
    alloc_last_week = {k: {} for k in policies}
    regret_curve = []
    cum_regret = 0.0
    for day in range(days):
        slices = ctx_slices(day)
        for pol, model in policies.items():
            spend_by: dict[tuple[str, str, str], float] = {}
            per_channel = {c: 0.0 for c in CHANNELS}
            for i, (t, p) in enumerate(slices):
                if pol == "oracle":
                    order = sorted(CHANNELS, key=lambda c: -truth[(c, t, p)])
                elif pol == "uniform":
                    order = [CHANNELS[(i + day) % len(CHANNELS)]]
                else:
                    th = model.sample_theta()
                    order = sorted(CHANNELS, key=lambda c: -sum(a * b for a, b in zip(th, features(c, t, p))))
                if pol != "uniform":  # portfolio guard: <= channel_cap of the day on one channel
                    order = [c for c in order if per_channel[c] + slice_usd <= channel_cap * daily_budget] or order
                c = order[0]
                per_channel[c] += slice_usd
                spend_by[(c, t, p)] = spend_by.get((c, t, p), 0.0) + slice_usd
            for (c, t, p), spend in spend_by.items():
                if pol == "lints" and agent is not None:  # commit through the harness in <= $5k calls
                    left = spend
                    while left > 0:
                        chunk = min(5_000.0, left)
                        with agent.runner.workflow(f"ua:d{day}", mode=READ_WRITE) as wf:
                            wf.call("allocate_ua_budget", day=day, channel=c, context=f"{t}/{p}", usd=chunk, policy=pol)
                        left -= chunk
                installs, rev = outcome(c, t, p, spend)
                totals[pol]["spend"] += spend
                totals[pol]["revenue"] += rev
                totals[pol]["installs"] += installs
                if pol == "lints":
                    model.observe(features(c, t, p), 100 * rev / spend, weight=spend / slice_usd)
                if day >= days - 7:
                    alloc_last_week[pol][(t, p, c)] = alloc_last_week[pol].get((t, p, c), 0.0) + spend
                with db.cursor() as con:
                    con.execute("INSERT INTO ua_allocations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                [agent.run_id if agent else "standalone", day, pol, c, f"{t}/{p}", spend, installs, round(rev, 4)])
        if policies["lints"] is not None:
            policies["lints"].refresh()
        cum_regret = totals["oracle"]["revenue"] - totals["lints"]["revenue"]
        if (day + 1) % 10 == 0:
            regret_curve.append({"day": day + 1, "lints_roas": round(totals["lints"]["revenue"] / totals["lints"]["spend"], 5),
                                 "uniform_roas": round(totals["uniform"]["revenue"] / totals["uniform"]["spend"], 5),
                                 "oracle_roas": round(totals["oracle"]["revenue"] / totals["oracle"]["spend"], 5)})
    best = {(t, p): max(CHANNELS, key=lambda c: truth[(c, t, p)]) for t in TIERS for p in PLATFORMS}
    on_best = {}
    for (t, p) in CONTEXT_SHARE:
        tot = sum(v for k, v in alloc_last_week["lints"].items() if k[:2] == (t, p))
        on_best[f"{t}/{p}"] = round(sum(v for k, v in alloc_last_week["lints"].items() if k == (t, p, best[(t, p)])) / tot, 3) if tot else None
    model = policies["lints"]
    learned = {f"{c}/{t}/{p}": round(model.predict(features(c, t, p)) / 100, 5) for c in CHANNELS for t in TIERS for p in PLATFORMS}
    return {"ltv_measured": {f"{t}/{p}": v for (t, p), v in ltv.items()},
            "true_roas": {f"{c}/{t}/{p}": round(v, 5) for (c, t, p), v in truth.items()}, "learned_roas": learned,
            "best_channel": {f"{t}/{p}": c for (t, p), c in best.items()},
            "totals": {k: {**v, "roas": round(v["revenue"] / v["spend"], 5)} for k, v in totals.items()},
            "regret_usd_vs_oracle": round(cum_regret, 2), "curve": regret_curve, "last_week_share_on_best": on_best,
            "days": days, "daily_budget": daily_budget}


# ============================================================================ ops scenario
def run_ops_scenario(db: TelemetryDB, *, hours: int = 72, requests: int = 3000, guard: bool = True, seed: int = 21,
                     tracer: Tracer | None = None) -> dict:
    """72h timeline: good floor push (h24), bad floor push (h34), network dark (h44-h55), market-wide drop (h64+)."""
    sim = MediationSimulator(seed=seed)
    agent = AdOpsAgent(db, sim, tracer=tracer, guard_enabled=guard, run_id=f"ops_{'guarded' if guard else 'unguarded'}_{seed}")
    timeline = []
    incidents = {"bad_floor_push": 34, "network_dark": 44, "network_recovers": 56, "market_drop": 64}
    for h in range(hours):
        if h == 24:
            agent.apply_floor(6.0, "TS-tuned floor")
        if h == incidents["bad_floor_push"]:
            agent.apply_floor(17.0, "exploratory floor push")
        if h == incidents["network_dark"]:
            sim.set_dark("admob")
        if h == incidents["network_recovers"]:
            sim.set_dark("admob", False)
        if h == incidents["market_drop"]:
            for n in sim.networks:
                sim.set_multiplier(n.name, 0.6)
        r = agent.step(h, requests)
        timeline.append({"hour": h, "ecpm": round(r["ecpm"], 3), "fill": round(r["fill_rate"], 4), "version": r["config_version"],
                         "reserve": r["reserve_cpm"], "disabled": r["disabled"], "actions": r["actions"]})
    rev = sum(t["ecpm"] * requests / 1000 for t in timeline)
    return {"guard": guard, "revenue_usd": round(rev, 2), "timeline": timeline, "events": agent.events, "run_id": agent.run_id,
            "incidents": incidents}
