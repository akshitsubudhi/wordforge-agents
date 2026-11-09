"""
================================================================================
BLOCK: Ad Mediation & Auction Simulator (Project 3)
--------------------------------------------------------------------------------
Role & contracts
  Stochastic, seedable simulator of a 5-network in-app ad mediation stack.
  Mechanisms:
    * sequential waterfall   - instances (network, fixed CPM price) called in
                               priority order, each adding latency; first
                               instance whose network accepts fills AT THAT PRICE.
                               naive: accept iff value >= price (upper bound);
                               strategic: accept iff beta(v) >= price (a
                               descending price ladder is a Dutch auction,
                               which is strategically equivalent to first price).
    * header bidding, 2nd price - parallel call, highest bid >= reserve wins,
                               pays max(2nd-highest bid, reserve).
    * header bidding, 1st price - winner pays its own bid, either truthful
                               (naive) or shaded to the Bayes-Nash equilibrium.
  Upstream : network configs (value distributions, fill, latency, timeouts)
             and incident controls (dark network, eCPM multiplier).
  Downstream: bandit_agent.py (floor tuning, anomaly detection, rollback)
             and scripts/run_ad_economy_sim.py (scorecard).
  Units: values, bids and prices are CPM in USD; revenue per impression = price/1000.

Key engineering mechanism
  * Analytical BNE bid shading for N symmetric bidders with value CDF F and
    reserve r (r = 0 gives the textbook formula):
        beta(v) = v - integral_r^v F(x)^(N-1) dx / F(v)^(N-1),   v >= r
    BidShader precomputes the integral on a grid with the trapezoid rule
    (any F, closed form not required) and interpolates linearly. Above the
    99.99th percentile beta is flat, which is exact because F(x) = 1 there.
    Validated two ways (see scorecard): the closed form (N-1)/N * v for
    U[0,1], and the Revenue Equivalence Theorem (shaded first-price revenue
    equals second-price revenue for symmetric i.i.d. bidders).
  * Heterogeneous networks shade against the POOLED opponent distribution (a
    mixture of every network's value CDF, including its no-demand mass), the
    standard approximation when an exact asymmetric equilibrium has no closed
    form.
  * Values are log-normal; per-network latency is log-normal; a network
    slower than the timeout is dropped from header bidding and costs the
    waterfall its full timeout. A DARK network hangs until the timeout, and an
    auction that had to wait out a timeout loses the placement with
    probability expire_prob_on_timeout (the user moved on). That is the real
    cost of leaving a dead network in the stack.

Failure modes & 1-step mitigation
  * Dark network / eCPM collapse -> set_dark() / set_multiplier() model the
    incident; the detector in bandit_agent.py catches it and quarantines
    the network or rolls the mediation config back.
  * Waterfall latency blow-up -> a waterfall-wide deadline stops the chain
    (counted as an unfilled impression, as in production SDKs).
================================================================================
"""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist, fmean
from typing import Callable, Literal, Sequence

_N = NormalDist()


@dataclass(frozen=True)
class Network:
    name: str
    mu: float            # log of median CPM value
    sigma: float         # log-normal shape
    fill_prob: float     # P(network has demand for this impression)
    latency_ms: float    # median response latency
    latency_sigma: float = 0.35

    def cdf(self, x: float) -> float:
        """CDF of the network's value INCLUDING the no-demand atom at 0."""
        if x <= 0:
            return 1 - self.fill_prob
        return (1 - self.fill_prob) + self.fill_prob * _N.cdf((math.log(x) - self.mu) / self.sigma)

    def quantile(self, q: float) -> float:
        return math.exp(self.mu + self.sigma * _N.inv_cdf(q))

    @property
    def mean_value_given_demand(self) -> float:
        return math.exp(self.mu + self.sigma ** 2 / 2)


DEFAULT_NETWORKS = (
    Network("applovin_max", math.log(9.0), 0.55, 0.92, 140),
    Network("admob", math.log(8.5), 0.45, 0.95, 120),
    Network("ironsource", math.log(7.5), 0.50, 0.90, 150),
    Network("meta_an", math.log(7.0), 0.70, 0.80, 180),
    Network("unity_ads", math.log(6.5), 0.60, 0.88, 160),
)


# ============================================================================ bid shading
class BidShader:
    """beta(v) = v - integral_r^v F(x)^(N-1) dx / F(v)^(N-1), on a precomputed grid."""

    def __init__(self, cdf: Callable[[float], float], n_bidders: int, reserve: float = 0.0, v_max: float = 1.0,
                 grid_n: int = 4000):
        if n_bidders < 2:
            raise ValueError("shading needs at least 2 bidders")
        if v_max <= reserve:
            raise ValueError("v_max must exceed the reserve")
        self.n, self.r, self.v_max = n_bidders, reserve, v_max
        h = (v_max - reserve) / grid_n
        self.xs = [reserve + i * h for i in range(grid_n + 1)]
        H = [cdf(x) ** (n_bidders - 1) for x in self.xs]
        integral = [0.0]
        for i in range(1, len(self.xs)):
            integral.append(integral[-1] + 0.5 * (H[i - 1] + H[i]) * h)
        self.beta = [x - (I / Hx if Hx > 1e-300 else 0.0) for x, I, Hx in zip(self.xs, integral, H)]
        self.beta[0] = reserve
        self.H_max, self.I_max = H[-1], integral[-1]

    def __call__(self, v: float) -> float | None:
        if v < self.r:
            return None  # below reserve: no bid
        if v >= self.v_max:
            return v - (self.I_max + (v - self.v_max) * self.H_max) / self.H_max
        i = bisect.bisect_right(self.xs, v) - 1
        x0, x1 = self.xs[i], self.xs[min(i + 1, len(self.xs) - 1)]
        if x1 == x0:
            return self.beta[i]
        t = (v - x0) / (x1 - x0)
        return self.beta[i] + t * (self.beta[i + 1] - self.beta[i])


def uniform_bne(v: float, n: int) -> float:
    """Closed form for U[0,1] values, reserve 0: beta(v) = (N-1)/N * v."""
    return (n - 1) / n * v


# ============================================================================ simulator
@dataclass
class ImpressionOutcome:
    filled: bool
    price: float            # CPM paid
    winner: str | None
    latency_ms: float
    efficient: bool         # winner had the highest value among networks with demand
    timed_out: int = 0
    winner_value: float = 0.0


@dataclass
class MechanismReport:
    mechanism: str
    impressions: int
    fill_rate: float
    ecpm: float             # revenue per 1000 REQUESTS (what the publisher earns)
    ecpm_filled: float      # revenue per 1000 FILLED impressions
    revenue_usd: float
    latency_mean_ms: float
    latency_p95_ms: float
    efficiency: float
    bidder_surplus_cpm: float   # mean (winner value - price) per REQUEST, CPM
    wins: dict[str, int] = field(default_factory=dict)
    revenue_by_network: dict[str, float] = field(default_factory=dict)
    timeouts: int = 0

    def row(self) -> list:
        return [self.mechanism, self.impressions, round(self.fill_rate, 4), round(self.ecpm, 3),
                round(self.ecpm_filled, 3), round(self.latency_mean_ms, 1), round(self.latency_p95_ms, 1),
                round(self.efficiency, 4), round(self.bidder_surplus_cpm, 3)]


Mechanism = Literal["waterfall", "waterfall_strategic", "hb_second", "hb_first_truthful", "hb_first_shaded"]


class MediationSimulator:
    def __init__(self, networks: Sequence[Network] = DEFAULT_NETWORKS, seed: int = 0, timeout_ms: float = 600.0,
                 waterfall_deadline_ms: float = 2500.0, expire_prob_on_timeout: float = 0.35):
        self.networks = list(networks)
        self.expire_prob_on_timeout = expire_prob_on_timeout
        self.rng = random.Random(seed)
        self.timeout_ms, self.waterfall_deadline_ms = timeout_ms, waterfall_deadline_ms
        self.dark: set[str] = set()
        self.multiplier: dict[str, float] = {}
        self._shaders: dict[tuple, BidShader] = {}

    # ---------------------------------------------------------------- incidents
    def set_dark(self, name: str, dark: bool = True) -> None:
        (self.dark.add if dark else self.dark.discard)(name)

    def set_multiplier(self, name: str, m: float) -> None:
        self.multiplier[name] = m

    # ---------------------------------------------------------------- draws
    def _draw(self, nets: Sequence[Network]) -> tuple[dict[str, float], dict[str, float]]:
        """Per network: value (CPM, absent if no demand) and response latency."""
        values, lat = {}, {}
        for n in nets:
            if n.name in self.dark:  # a dark endpoint hangs until the client times out
                lat[n.name] = self.timeout_ms * 1.5
                continue
            lat[n.name] = n.latency_ms * math.exp(self.rng.gauss(0, n.latency_sigma))
            if self.rng.random() > n.fill_prob:
                continue
            values[n.name] = math.exp(self.rng.gauss(n.mu, n.sigma)) * self.multiplier.get(n.name, 1.0)
        return values, lat

    def pooled_cdf(self, nets: Sequence[Network]) -> Callable[[float], float]:
        return lambda x: fmean(n.cdf(x) for n in nets)

    def shader(self, nets: Sequence[Network], reserve: float) -> BidShader:
        key = (tuple(n.name for n in nets), round(reserve, 4))
        if key not in self._shaders:
            v_max = max(n.quantile(0.9999) for n in nets)
            self._shaders[key] = BidShader(self.pooled_cdf(nets), len(nets), reserve, v_max)
        return self._shaders[key]

    # ---------------------------------------------------------------- mechanisms
    def default_waterfall(self, nets: Sequence[Network] | None = None) -> list[tuple[str, float]]:
        """Classic set-up: 2 instances per network at its p80 and p45 value, sorted by price."""
        nets = nets or self.networks
        tiers = [(n.name, round(n.quantile(q), 2)) for n in nets for q in (0.80, 0.45)]
        return sorted(tiers, key=lambda t: -t[1])

    def waterfall_impression(self, tiers: Sequence[tuple[str, float]], nets: Sequence[Network],
                             strategic: bool = False) -> ImpressionOutcome:
        """strategic=True: a network accepts price p only if p <= beta(v). A descending sequence of prices is a
        Dutch auction, strategically equivalent to first price, so rational networks shade exactly as in
        hb_first_shaded (declining keeps its lower-priced instance open)."""
        values, lat = self._draw(nets)
        sh = self.shader(nets, min(p for _, p in tiers)) if strategic else None
        best = max(values.values(), default=0.0)
        elapsed, timeouts = 0.0, 0
        for name, price in tiers:
            if name not in lat:
                continue
            step = min(lat[name], self.timeout_ms)
            timeouts += lat[name] > self.timeout_ms
            elapsed += step
            if elapsed > self.waterfall_deadline_ms:
                return ImpressionOutcome(False, 0.0, None, elapsed, False, timeouts)
            v = values.get(name, 0.0)
            accept_at = (sh(v) or 0.0) if strategic else v
            if lat[name] <= self.timeout_ms and v > 0 and accept_at >= price:
                return ImpressionOutcome(True, price, name, elapsed, v == best, timeouts, v)
        return ImpressionOutcome(False, 0.0, None, elapsed, False, timeouts)

    def header_impression(self, kind: Mechanism, reserve: float, nets: Sequence[Network]) -> ImpressionOutcome:
        values, lat = self._draw(nets)
        responded = {k: v for k, v in values.items() if lat[k] <= self.timeout_ms}
        timeouts = sum(l > self.timeout_ms for l in lat.values())
        latency = min(self.timeout_ms, max(lat.values())) if lat else 0.0
        best_value = max(values.values(), default=0.0)
        if timeouts and self.rng.random() < self.expire_prob_on_timeout:
            # the auction waited for a hung network until the timeout; the placement expired meanwhile
            return ImpressionOutcome(False, 0.0, None, latency, False, timeouts)
        if kind == "hb_first_shaded":
            sh = self.shader(nets, reserve)
            bids = {k: b for k, v in responded.items() if (b := sh(v)) is not None}
        else:
            bids = {k: v for k, v in responded.items() if v >= reserve}
        if not bids:
            return ImpressionOutcome(False, 0.0, None, latency, False, timeouts)
        ranked = sorted(bids.items(), key=lambda kv: -kv[1])
        winner, top = ranked[0]
        if kind == "hb_second":
            price = max(ranked[1][1] if len(ranked) > 1 else 0.0, reserve)
        else:
            price = top
        return ImpressionOutcome(True, price, winner, latency, values[winner] == best_value, timeouts, values[winner])

    def run(self, mechanism: Mechanism, n: int, *, reserve: float = 0.0, tiers: Sequence[tuple[str, float]] | None = None,
            enabled: set[str] | None = None) -> MechanismReport:
        nets = [x for x in self.networks if enabled is None or x.name in enabled]
        if mechanism.startswith("waterfall"):
            tiers = [t for t in (tiers or self.default_waterfall(nets)) if enabled is None or t[0] in enabled]
        lats, wins, rev_by = [], {}, {}
        filled = efficient = timeouts = 0
        revenue = surplus = 0.0
        for _ in range(n):
            if mechanism.startswith("waterfall"):
                o = self.waterfall_impression(tiers, nets, strategic=mechanism == "waterfall_strategic")
            else:
                o = self.header_impression(mechanism, reserve, nets)
            lats.append(o.latency_ms)
            timeouts += o.timed_out
            if o.filled:
                filled += 1
                efficient += o.efficient
                revenue += o.price / 1000
                surplus += o.winner_value - o.price
                wins[o.winner] = wins.get(o.winner, 0) + 1
                rev_by[o.winner] = rev_by.get(o.winner, 0.0) + o.price / 1000
        lats.sort()
        return MechanismReport(
            mechanism, n, filled / n, revenue / n * 1000, revenue / filled * 1000 if filled else 0.0, revenue,
            fmean(lats) if lats else 0.0, lats[min(len(lats) - 1, int(0.95 * len(lats)))] if lats else 0.0,
            efficient / filled if filled else 0.0, surplus / n, wins, {k: round(v, 4) for k, v in rev_by.items()}, timeouts)


# ============================================================================ validation helpers
def revenue_equivalence_check(n_bidders: int = 5, reserve: float = 4.0, auctions: int = 100_000, seed: int = 3,
                              mu: float = math.log(8.0), sigma: float = 0.5) -> dict:
    """Symmetric i.i.d. log-normal bidders: E[rev | 1st price + BNE shading] == E[rev | 2nd price]."""
    rng = random.Random(seed)
    net = Network("iid", mu, sigma, 1.0, 100)
    sh = BidShader(net.cdf, n_bidders, reserve, net.quantile(0.99999))
    fp, sp = [], []
    for _ in range(auctions):
        v = sorted((math.exp(rng.gauss(mu, sigma)) for _ in range(n_bidders)), reverse=True)
        if v[0] < reserve:
            fp.append(0.0)
            sp.append(0.0)
            continue
        fp.append(sh(v[0]))
        sp.append(max(v[1], reserve))
    m_fp, m_sp = fmean(fp), fmean(sp)
    se = math.sqrt(sum((a - b - (m_fp - m_sp)) ** 2 for a, b in zip(fp, sp)) / (auctions - 1) / auctions)
    return {"bidders": n_bidders, "reserve": reserve, "auctions": auctions, "first_price_shaded_cpm": round(m_fp, 4),
            "second_price_cpm": round(m_sp, 4), "difference": round(m_fp - m_sp, 4), "se": round(se, 4),
            "z": round((m_fp - m_sp) / se, 2) if se else 0.0}


def uniform_closed_form_check(n_bidders: int = 5) -> dict:
    sh = BidShader(lambda x: min(1.0, max(0.0, x)), n_bidders, 0.0, 1.0, grid_n=4000)
    pts = [i / 20 for i in range(1, 21)]
    err = max(abs(sh(v) - uniform_bne(v, n_bidders)) for v in pts)
    return {"bidders": n_bidders, "max_abs_error": err, "beta(0.8)": round(sh(0.8), 6),
            "closed_form(0.8)": round(uniform_bne(0.8, n_bidders), 6)}
