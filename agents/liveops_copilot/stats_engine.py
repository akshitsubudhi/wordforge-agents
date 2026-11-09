"""
================================================================================
BLOCK: Experimentation Stats Engine (power, CUPED, mSPRT) - pure stdlib
--------------------------------------------------------------------------------
Role & contracts
  Pure functions, no I/O. Inputs are plain Python sequences; outputs are
  frozen dataclasses. Used by agents/liveops_copilot/agent.py (power analysis
  before launch, CUPED + mSPRT at every daily peek, rollback guardrail) and by
  scripts/verify_phase2.py (peeking false-positive simulation).

Key engineering mechanism
  * Sample size, proportions (Fleiss, unequal allocation k = n_c / n_t):
        n_t = [ z_{1-a/2} sqrt(p_bar(1-p_bar)(1 + 1/k)) + z_{1-b} sqrt(p1(1-p1)/k + p2(1-p2)) ]^2 / d^2
        p_bar = (p2 + k p1) / (1 + k)
    Means (with optional CUPED correlation rho, treatment share s):
        N = sigma^2 (1 - rho^2) (z_{1-a/2} + z_{1-b})^2 (1/s + 1/(1-s)) / d^2
  * CUPED (Deng et al., 2013): theta = cov(X,Y)/var(X) pooled over both
    arms (so theta is independent of assignment); Y' = Y - theta (X - mean X).
    Reports raw vs adjusted estimator variance, reduction %, effects,
    Welch z-tests and CIs. Expected reduction ~ rho^2.
  * mSPRT (Johari, Pekelis & Walsh, 2017), normal mixture N(0, tau^2) on
    the effect. With V_n = Var(delta_hat_n):
        Lambda_n = sqrt(V_n / (V_n + tau^2)) * exp( tau^2 (delta_hat - theta0)^2 / (2 V_n (V_n + tau^2)) )
        always-valid p_n = min(p_{n-1}, 1 / Lambda_n)
        CI_n = delta_hat +- sqrt( V_n (V_n + tau^2) / tau^2 * (2 ln(1/alpha) + ln((V_n + tau^2) / V_n)) )
    intersected with every earlier CI. Lambda is a non-negative martingale
    under H0, so by Ville's inequality P(ever Lambda >= 1/alpha) <= alpha:
    peeking after every batch NEVER inflates the false-positive rate.

Failure modes & 1-step mitigation
  * Degenerate inputs (n < 2, zero variance) -> ValueError with the cause,
    never a silent NaN decision. Callers that must keep running (the daily
    monitor) fall back explicitly: unadjusted() when the covariate is
    constant, and "no decision" when an outcome has no variance yet.
  * Mis-chosen tau -> only costs power, never validity; default tau = MDE
    (the effect size the test is designed to detect).
  * Plug-in variance estimate early in the test -> min_samples floor before
    any decision (default 100 per arm).
================================================================================
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist, fmean
from typing import Sequence

_N = NormalDist()


def _z(p: float) -> float:
    return _N.inv_cdf(p)


def _two_sided_p(z: float) -> float:
    return 2.0 * (1.0 - _N.cdf(abs(z)))


def _var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 observations")
    m = fmean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _cov(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx, my = fmean(xs), fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)


# ============================================================================ power analysis
@dataclass(frozen=True)
class PowerResult:
    metric: str
    baseline: float
    mde_abs: float
    alpha: float
    power: float
    treatment_share: float
    n_treatment: int
    n_control: int
    n_total: int
    cuped_rho: float = 0.0
    n_total_without_cuped: int | None = None
    days: int | None = None
    daily_eligible_users: int | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def sample_size_proportions(p_control: float, mde_abs: float, *, alpha: float = 0.05, power: float = 0.8,
                            treatment_share: float = 0.5, daily_eligible_users: int | None = None) -> PowerResult:
    if not (0 < p_control < 1 and 0 < p_control + mde_abs < 1) or mde_abs == 0:
        raise ValueError("need 0 < p < 1 for both arms and a non-zero MDE")
    if not 0 < treatment_share < 1:
        raise ValueError("treatment_share must be in (0, 1)")
    p1, p2 = p_control, p_control + mde_abs
    k = (1 - treatment_share) / treatment_share            # n_control / n_treatment
    pbar = (p2 + k * p1) / (1 + k)
    za, zb = _z(1 - alpha / 2), _z(power)
    n_t = (za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / k)) + zb * math.sqrt(p1 * (1 - p1) / k + p2 * (1 - p2))) ** 2 / mde_abs ** 2
    n_t = math.ceil(n_t)
    n_c = math.ceil(n_t * k)
    days = math.ceil((n_t + n_c) / daily_eligible_users) if daily_eligible_users else None
    return PowerResult("proportion", p_control, mde_abs, alpha, power, treatment_share, n_t, n_c, n_t + n_c,
                       days=days, daily_eligible_users=daily_eligible_users)


def sample_size_means(sigma: float, mde_abs: float, *, baseline: float = 0.0, alpha: float = 0.05, power: float = 0.8,
                      treatment_share: float = 0.5, cuped_rho: float = 0.0,
                      daily_eligible_users: int | None = None) -> PowerResult:
    if sigma <= 0 or mde_abs == 0:
        raise ValueError("sigma must be > 0 and MDE non-zero")
    if not 0 < treatment_share < 1 or not 0 <= abs(cuped_rho) < 1:
        raise ValueError("treatment_share in (0,1), |rho| < 1")
    za, zb = _z(1 - alpha / 2), _z(power)
    alloc = 1 / treatment_share + 1 / (1 - treatment_share)

    def total(var: float) -> int:
        return math.ceil(var * (za + zb) ** 2 * alloc / mde_abs ** 2)

    n_raw = total(sigma ** 2)
    n = total(sigma ** 2 * (1 - cuped_rho ** 2))
    n_t = math.ceil(n * treatment_share)
    days = math.ceil(n / daily_eligible_users) if daily_eligible_users else None
    return PowerResult("mean", baseline, mde_abs, alpha, power, treatment_share, n_t, n - n_t, n, cuped_rho, n_raw,
                       days, daily_eligible_users)


# ============================================================================ CUPED
@dataclass(frozen=True)
class CupedResult:
    n_treatment: int
    n_control: int
    theta: float
    rho: float
    effect_raw: float
    effect_cuped: float
    var_raw: float              # Var(effect estimator), raw
    var_cuped: float            # Var(effect estimator), CUPED-adjusted
    variance_reduction_pct: float
    se_raw: float
    se_cuped: float
    p_raw: float
    p_cuped: float
    ci_raw: tuple[float, float]
    ci_cuped: tuple[float, float]
    mean_control: float
    relative_lift_cuped: float

    def to_dict(self) -> dict:
        # non-finite floats (relative lift when the control mean is 0) become None: strict JSON for the API
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in self.__dict__.items()}


def _welch(yt: Sequence[float], yc: Sequence[float]) -> tuple[float, float]:
    return fmean(yt) - fmean(yc), _var(yt) / len(yt) + _var(yc) / len(yc)


@dataclass(frozen=True)
class Moments:
    """Sufficient statistics of one arm: n, sum x, sum y, sum x^2, sum y^2, sum xy (x = covariate, y = outcome).
    They add across days, so a daily peek costs O(1) instead of re-reading every exposure so far."""
    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0

    @staticmethod
    def of(y: Sequence[float], x: Sequence[float] | None = None) -> "Moments":
        x = [0.0] * len(y) if x is None else x
        if len(x) != len(y):
            raise ValueError("outcome and covariate lengths differ")
        return Moments(len(y), math.fsum(x), math.fsum(y), math.fsum(v * v for v in x), math.fsum(v * v for v in y),
                       math.fsum(a * b for a, b in zip(x, y)))

    def __add__(self, o: "Moments") -> "Moments":
        return Moments(self.n + o.n, self.sx + o.sx, self.sy + o.sy, self.sxx + o.sxx, self.syy + o.syy, self.sxy + o.sxy)

    def _ss(self, sq: float, a: float, b: float) -> float:
        return (sq - a * b / self.n) / (self.n - 1)

    @property
    def var_x(self) -> float:
        return max(0.0, self._ss(self.sxx, self.sx, self.sx))

    @property
    def var_y(self) -> float:
        return max(0.0, self._ss(self.syy, self.sy, self.sy))

    @property
    def cov(self) -> float:
        return self._ss(self.sxy, self.sx, self.sy)


def _result(mt: Moments, mc: Moments, theta: float, rho: float, alpha: float) -> CupedResult:
    eff_raw = mt.sy / mt.n - mc.sy / mc.n
    eff_adj = eff_raw - theta * (mt.sx / mt.n - mc.sx / mc.n)
    v_raw = mt.var_y / mt.n + mc.var_y / mc.n
    adj = lambda m: max(0.0, m.var_y - 2 * theta * m.cov + theta * theta * m.var_x)   # Var(y - theta x) in one arm
    v_adj = adj(mt) / mt.n + adj(mc) / mc.n
    se_r, se_a = math.sqrt(v_raw), math.sqrt(v_adj)
    zc = _z(1 - alpha / 2)
    mean_c = mc.sy / mc.n
    return CupedResult(
        mt.n, mc.n, round(theta, 6), round(rho, 6), eff_raw, eff_adj, v_raw, v_adj,
        round(100 * (1 - v_adj / v_raw), 2) if v_raw > 0 else 0.0, se_r, se_a,
        _two_sided_p(eff_raw / se_r) if se_r > 0 else 1.0, _two_sided_p(eff_adj / se_a) if se_a > 0 else 1.0,
        (eff_raw - zc * se_r, eff_raw + zc * se_r), (eff_adj - zc * se_a, eff_adj + zc * se_a),
        mean_c, eff_adj / mean_c if mean_c else float("nan"),
    )


def cuped_from_moments(mt: Moments, mc: Moments, alpha: float = 0.05) -> CupedResult:
    """CUPED from per-arm sufficient statistics. theta = cov(X,Y)/var(X) pooled over both arms."""
    if min(mt.n, mc.n) < 2:
        raise ValueError("need >= 2 units per arm")
    pooled = mt + mc
    vx = pooled.var_x
    if vx <= 1e-12 * max(1.0, pooled.sxx / pooled.n):
        raise ValueError("covariate has zero variance - CUPED undefined")
    theta = pooled.cov / vx
    vy = pooled.var_y
    return _result(mt, mc, theta, pooled.cov / math.sqrt(vx * vy) if vy > 0 else 0.0, alpha)


def cuped(y_t: Sequence[float], x_t: Sequence[float], y_c: Sequence[float], x_c: Sequence[float],
          alpha: float = 0.05) -> CupedResult:
    if len(y_t) != len(x_t) or len(y_c) != len(x_c):
        raise ValueError("outcome and covariate lengths differ")
    return cuped_from_moments(Moments.of(y_t, x_t), Moments.of(y_c, x_c), alpha)


def unadjusted_from_moments(mt: Moments, mc: Moments, alpha: float = 0.05) -> CupedResult:
    """Plain Welch estimate in the CupedResult shape (theta = 0): the documented fallback when CUPED is undefined
    because the covariate has zero variance (e.g. an experiment on brand-new players with no pre-period)."""
    if min(mt.n, mc.n) < 2:
        raise ValueError("need >= 2 units per arm")
    return _result(mt, mc, 0.0, 0.0, alpha)


def unadjusted(y_t: Sequence[float], y_c: Sequence[float], alpha: float = 0.05) -> CupedResult:
    return unadjusted_from_moments(Moments.of(y_t), Moments.of(y_c), alpha)


# ============================================================================ mSPRT
def mixture_likelihood_ratio(delta: float, var_delta: float, tau2: float, theta0: float = 0.0) -> float:
    if var_delta <= 0 or tau2 <= 0:
        raise ValueError("variance and tau^2 must be positive")
    log_l = 0.5 * math.log(var_delta / (var_delta + tau2)) + tau2 * (delta - theta0) ** 2 / (2 * var_delta * (var_delta + tau2))
    return math.exp(min(log_l, 700.0))


def always_valid_ci(delta: float, var_delta: float, tau2: float, alpha: float) -> tuple[float, float]:
    half = math.sqrt(var_delta * (var_delta + tau2) / tau2 * (2 * math.log(1 / alpha) + math.log((var_delta + tau2) / var_delta)))
    return delta - half, delta + half


@dataclass(frozen=True)
class MsprtStep:
    peek: int
    n_treatment: int
    n_control: int
    delta: float
    var_delta: float
    likelihood_ratio: float
    p_always_valid: float
    ci_low: float
    ci_high: float
    reject_null: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class MSPRT:
    """Always-valid sequential test of H0: effect == theta0. Peek as often as you like."""

    def __init__(self, alpha: float = 0.05, tau: float = 1.0, theta0: float = 0.0, min_samples: int = 100):
        if not 0 < alpha < 1 or tau <= 0:
            raise ValueError("alpha in (0,1), tau > 0")
        self.alpha, self.tau2, self.theta0, self.min_samples = alpha, tau * tau, theta0, min_samples
        self.p = 1.0
        self.lo, self.hi = -math.inf, math.inf
        self.history: list[MsprtStep] = []

    def update(self, delta: float, var_delta: float, n_t: int, n_c: int) -> MsprtStep:
        lam = mixture_likelihood_ratio(delta, var_delta, self.tau2, self.theta0)
        lo, hi = always_valid_ci(delta, var_delta, self.tau2, self.alpha)
        enough = min(n_t, n_c) >= self.min_samples
        if enough:  # running min / running intersection keep the guarantees monotone
            self.p = min(self.p, 1.0 / lam if lam > 0 else 1.0)
            self.lo, self.hi = max(self.lo, lo), min(self.hi, hi)
        step = MsprtStep(len(self.history) + 1, n_t, n_c, delta, var_delta, lam, self.p,
                         self.lo if enough else lo, self.hi if enough else hi, enough and self.p <= self.alpha)
        self.history.append(step)
        return step

    def update_from_samples(self, y_t: Sequence[float], y_c: Sequence[float]) -> MsprtStep:
        d, v = _welch(y_t, y_c)
        return self.update(d, v, len(y_t), len(y_c))


def guardrail_breached(step: MsprtStep, tolerance: float = 0.0) -> bool:
    """Harm is PROVEN when the whole always-valid CI sits below -tolerance."""
    return step.n_treatment > 0 and step.ci_high < -tolerance and step.ci_high != math.inf


# ============================================================================ peeking simulation
@dataclass(frozen=True)
class PeekingReport:
    sims: int
    peeks: int
    n_per_arm_per_peek: int
    alpha: float
    naive_false_positive_rate: float
    msprt_false_positive_rate: float
    naive_single_look_rate: float


def peeking_false_positive_simulation(sims: int = 400, peeks: int = 14, n_per_peek: int = 200, alpha: float = 0.05,
                                      tau: float = 0.1, seed: int = 7) -> PeekingReport:
    """A/A tests (true effect 0), N(0,1) outcomes, peek after every batch.

    naive  = reject the first time a fixed-horizon z-test gives p < alpha
    mSPRT  = reject the first time the always-valid p <= alpha"""
    rng = random.Random(seed)
    zc = _z(1 - alpha / 2)
    naive_hits = msprt_hits = final_hits = 0
    for _ in range(sims):
        st = ss_t = sc = ss_c = 0.0
        n = 0
        naive = msprt = False
        test = MSPRT(alpha=alpha, tau=tau, min_samples=1)
        for _ in range(peeks):
            for _ in range(n_per_peek):
                a, b = rng.gauss(0, 1), rng.gauss(0, 1)
                st += a; ss_t += a * a; sc += b; ss_c += b * b
            n += n_per_peek
            mt, mc = st / n, sc / n
            vt, vc = (ss_t - n * mt * mt) / (n - 1), (ss_c - n * mc * mc) / (n - 1)
            var_d = vt / n + vc / n
            z = (mt - mc) / math.sqrt(var_d)
            naive = naive or abs(z) > zc
            msprt = msprt or test.update(mt - mc, var_d, n, n).reject_null
        final_hits += abs(z) > zc
        naive_hits += naive
        msprt_hits += msprt
    return PeekingReport(sims, peeks, n_per_peek, alpha, naive_hits / sims, msprt_hits / sims, final_hits / sims)
