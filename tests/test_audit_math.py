"""
================================================================================
BLOCK: Audit Suite 1/3 - "No Fake Math"
--------------------------------------------------------------------------------
Role & contracts
  Proves that every statistical function computes its formula instead of
  returning a stub or a hardcoded constant. Nothing here re-states the
  implementation's own formula and compares it to itself: every check uses
  an INDEPENDENT reference:
    * a Monte Carlo experiment (achieved power, false-positive rate,
      posterior moments, revenue equivalence),
    * a numerical integral (the mSPRT mixture likelihood ratio),
    * a different closed form (uniform BNE with a reserve, Student-t
      marginal of the Normal-Inverse-Gamma posterior), or
    * a hand-worked textbook table (Cohen's kappa).
  Each block ends with a "responds to its inputs" check, which a hardcoded
  return value cannot pass.

Failure modes & 1-step mitigation
  * Monte Carlo flakiness -> fixed seeds; tolerances are >= 3 standard errors
    of the estimate, stated next to each assertion.
================================================================================
"""
from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist

import pytest

from agents.ad_economy.auction_sim import BidShader, revenue_equivalence_check, uniform_bne
from agents.ad_economy.bandit_agent import (
    ContextualLTVBandit, NIGArm, ThompsonFloorTuner, _cholesky, _inverse, features,
)
from agents.level_factory.critic_and_eval import cohens_kappa
from agents.liveops_copilot.stats_engine import (
    MSPRT, always_valid_ci, cuped, mixture_likelihood_ratio, peeking_false_positive_simulation, sample_size_means,
    sample_size_proportions,
)

N01 = NormalDist()


# ============================================================================ sample size / power
def _achieved_power_proportions(p_c: float, p_t: float, n_t: int, n_c: int, sims: int, seed: int) -> float:
    """Fraction of simulated experiments in which a pooled two-proportion z-test rejects at alpha = 0.05."""
    rng = random.Random(seed)
    zc = N01.inv_cdf(0.975)
    hits = 0
    for _ in range(sims):
        # normal approximation to the binomial counts (n is in the thousands) keeps this fast
        xt = rng.gauss(n_t * p_t, math.sqrt(n_t * p_t * (1 - p_t)))
        xc = rng.gauss(n_c * p_c, math.sqrt(n_c * p_c * (1 - p_c)))
        pt, pc = xt / n_t, xc / n_c
        pool = (xt + xc) / (n_t + n_c)
        z = (pt - pc) / math.sqrt(pool * (1 - pool) * (1 / n_t + 1 / n_c))
        hits += abs(z) > zc
    return hits / sims


@pytest.mark.parametrize("p,mde,share,power", [(0.10, 0.02, 0.5, 0.8), (0.445, -0.02, 0.2, 0.8), (0.30, 0.03, 0.2, 0.9)])
def test_proportion_sample_size_delivers_the_requested_power(p, mde, share, power):
    r = sample_size_proportions(p, mde, power=power, treatment_share=share)
    achieved = _achieved_power_proportions(p, p + mde, r.n_treatment, r.n_control, sims=6000, seed=11)
    assert abs(achieved - power) < 0.025, (achieved, r)          # SE of the estimate ~0.005
    assert r.n_control == pytest.approx(r.n_treatment * (1 - share) / share, abs=1)   # allocation honoured


def test_proportion_sample_size_scales_like_the_formula_says():
    base = sample_size_proportions(0.10, 0.02).n_total
    assert sample_size_proportions(0.10, 0.01).n_total == pytest.approx(4 * base, rel=0.08)   # ~1/d^2
    assert sample_size_proportions(0.10, 0.02, power=0.9).n_total > base
    assert sample_size_proportions(0.10, 0.02, alpha=0.01).n_total > base
    assert sample_size_proportions(0.10, 0.02, treatment_share=0.2).n_total > base           # unbalanced costs users
    with pytest.raises(ValueError):
        sample_size_proportions(0.10, 0.0)
    with pytest.raises(ValueError):
        sample_size_proportions(0.99, 0.02)


def test_mean_sample_size_delivers_power_and_cuped_shrinks_it_by_one_minus_rho_squared():
    sigma, d, share = 5.0, 0.4, 0.2
    r = sample_size_means(sigma, d, treatment_share=share)
    rng = random.Random(5)
    zc = N01.inv_cdf(0.975)
    se = sigma * math.sqrt(1 / r.n_treatment + 1 / r.n_control)
    hits = sum(abs(rng.gauss(d, se) / se) > zc for _ in range(20_000)) / 20_000   # exact sampling law of the z-stat
    assert abs(hits - 0.80) < 0.012
    for rho in (0.3, 0.6, 0.9):
        c = sample_size_means(sigma, d, treatment_share=share, cuped_rho=rho)
        assert c.n_total == pytest.approx(r.n_total * (1 - rho ** 2), abs=2)
        assert c.n_total_without_cuped == r.n_total


# ============================================================================ CUPED
def _correlated_arms(rng: random.Random, n: int, rho: float, effect: float, beta: float = 2.0):
    """y = beta*x + noise with corr(x, y) = rho; treatment adds `effect`."""
    noise_sd = beta * math.sqrt(1 - rho ** 2) / rho
    arm = lambda shift: tuple(zip(*[((x := rng.gauss(10, 1)) * beta + rng.gauss(0, noise_sd) + shift, x) for _ in range(n)]))
    (yt, xt), (yc, xc) = arm(effect), arm(0.0)
    return list(yt), list(xt), list(yc), list(xc)


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.85])
def test_cuped_theta_is_the_ols_slope_and_variance_drops_by_rho_squared(rho):
    yt, xt, yc, xc = _correlated_arms(random.Random(1), 6000, rho, effect=0.5)
    r = cuped(yt, xt, yc, xc)
    slope = statistics.linear_regression(xt + xc, yt + yc).slope       # independent stdlib OLS
    assert r.theta == pytest.approx(slope, rel=1e-4)
    assert r.rho == pytest.approx(statistics.correlation(xt + xc, yt + yc), abs=1e-5)
    assert r.variance_reduction_pct / 100 == pytest.approx(rho ** 2, abs=0.03)
    assert r.var_cuped < r.var_raw and r.se_cuped == pytest.approx(math.sqrt(r.var_cuped))


def test_cuped_is_unbiased_and_its_ci_has_nominal_coverage():
    true, cover_raw, cover_adj, effects = 0.5, 0, 0, []
    for seed in range(300):
        yt, xt, yc, xc = _correlated_arms(random.Random(seed), 400, 0.7, effect=true)
        r = cuped(yt, xt, yc, xc)
        effects.append(r.effect_cuped)
        cover_adj += r.ci_cuped[0] <= true <= r.ci_cuped[1]
        cover_raw += r.ci_raw[0] <= true <= r.ci_raw[1]
    assert statistics.fmean(effects) == pytest.approx(true, abs=0.02)   # SE of the mean ~0.006
    assert 0.92 <= cover_adj / 300 <= 0.98 and 0.92 <= cover_raw / 300 <= 0.98


def test_cuped_with_a_useless_covariate_changes_nothing_and_degenerate_input_raises():
    rng = random.Random(3)
    y_t, y_c = [rng.gauss(1, 1) for _ in range(3000)], [rng.gauss(0, 1) for _ in range(3000)]
    noise = [rng.gauss(0, 1) for _ in range(6000)]
    r = cuped(y_t, noise[:3000], y_c, noise[3000:])
    assert abs(r.variance_reduction_pct) < 1.0 and r.effect_cuped == pytest.approx(r.effect_raw, abs=0.02)
    with pytest.raises(ValueError):
        cuped([1.0, 2.0], [3.0, 3.0], [1.0, 2.0], [3.0, 3.0])          # zero-variance covariate
    with pytest.raises(ValueError):
        cuped([1.0, 2.0], [1.0], [1.0, 2.0], [1.0, 2.0])


# ============================================================================ mSPRT
def _mixture_lr_by_quadrature(delta: float, v: float, tau2: float, theta0: float = 0.0) -> float:
    """Lambda = integral of N(delta; theta, v) / N(delta; theta0, v) against the mixing density N(theta0, tau2)."""
    like = lambda th: math.exp(-(delta - th) ** 2 / (2 * v))
    prior = lambda th: math.exp(-(th - theta0) ** 2 / (2 * tau2)) / math.sqrt(2 * math.pi * tau2)
    lo, hi, n = theta0 - 12 * math.sqrt(tau2), theta0 + 12 * math.sqrt(tau2), 40_000
    h = (hi - lo) / n
    total = sum((0.5 if i in (0, n) else 1.0) * like(lo + i * h) * prior(lo + i * h) for i in range(n + 1)) * h
    return total / like(theta0)


@pytest.mark.parametrize("delta,v,tau,theta0", [(0.0, 0.01, 0.1, 0.0), (0.25, 0.01, 0.1, 0.0), (-0.4, 0.04, 0.3, 0.0),
                                                 (0.9, 0.002, 0.05, 0.5), (0.05, 0.0004, 0.5, 0.0)])
def test_msprt_closed_form_equals_the_mixture_integral(delta, v, tau, theta0):
    closed = mixture_likelihood_ratio(delta, v, tau * tau, theta0)
    assert closed == pytest.approx(_mixture_lr_by_quadrature(delta, v, tau * tau, theta0), rel=1e-6)


def test_msprt_ci_is_exactly_the_set_of_nulls_the_test_does_not_reject():
    delta, v, tau2, alpha = 0.31, 0.004, 0.02, 0.05
    lo, hi = always_valid_ci(delta, v, tau2, alpha)
    for edge in (lo, hi):   # on the boundary the likelihood ratio against that null is exactly 1/alpha
        assert mixture_likelihood_ratio(delta, v, tau2, theta0=edge) == pytest.approx(1 / alpha, rel=1e-9)
    assert mixture_likelihood_ratio(delta, v, tau2, theta0=(lo + hi) / 2) < 1 / alpha
    assert mixture_likelihood_ratio(delta, v, tau2, theta0=hi + 0.01) > 1 / alpha
    assert (hi - lo) / 2 > N01.inv_cdf(0.975) * math.sqrt(v)   # always-valid is wider than a fixed-horizon CI


def _sequential_run(rng: random.Random, effect: float, peeks: int, n_per_peek: int, tau: float, alpha: float = 0.05):
    test = MSPRT(alpha=alpha, tau=tau, min_samples=1)
    zc = N01.inv_cdf(1 - alpha / 2)
    st = sc = sst = ssc = 0.0
    n, naive, step = 0, False, None
    for _ in range(peeks):
        for _ in range(n_per_peek):
            a, b = rng.gauss(effect, 1), rng.gauss(0, 1)
            st += a; sst += a * a; sc += b; ssc += b * b
        n += n_per_peek
        mt, mc = st / n, sc / n
        var = ((sst - n * mt * mt) + (ssc - n * mc * mc)) / (n - 1) / n
        naive = naive or abs((mt - mc) / math.sqrt(var)) > zc
        step = test.update(mt - mc, var, n, n)
    return naive, test, step


def test_msprt_type_one_error_stays_below_alpha_under_heavy_peeking_while_naive_explodes():
    rng = random.Random(2026)
    sims, naive_hits, msprt_hits = 1000, 0, 0
    for _ in range(sims):
        naive, test, _ = _sequential_run(rng, effect=0.0, peeks=40, n_per_peek=50, tau=0.15)
        naive_hits += naive
        msprt_hits += any(s.reject_null for s in test.history)
    assert msprt_hits / sims <= 0.05            # Ville's inequality: P(ever reject | H0) <= alpha
    assert naive_hits / sims > 0.25             # 40 peeks: the fixed-horizon test is wrong > 5x as often as claimed


def test_msprt_has_power_and_its_guarantees_are_monotone():
    rng = random.Random(7)
    rejected = 0
    for _ in range(200):
        _, test, last = _sequential_run(rng, effect=0.2, peeks=20, n_per_peek=100, tau=0.2)
        rejected += last.reject_null
        ps = [s.p_always_valid for s in test.history]
        assert all(a >= b for a, b in zip(ps, ps[1:]))                                   # p never goes back up
        assert all(a.ci_low <= b.ci_low and a.ci_high >= b.ci_high for a, b in zip(test.history, test.history[1:]))
        if last.reject_null:
            assert last.ci_low > 0 or last.ci_high < 0                                   # reject <=> CI excludes 0
    assert rejected / 200 > 0.9                 # n = 2000/arm, d = 0.2 sigma: a real effect is found


def test_peeking_simulation_is_computed_not_hardcoded():
    a = peeking_false_positive_simulation(sims=200, peeks=10, n_per_peek=100, seed=1)
    b = peeking_false_positive_simulation(sims=200, peeks=10, n_per_peek=100, seed=2)
    one = peeking_false_positive_simulation(sims=200, peeks=1, n_per_peek=1000, seed=1)
    assert a != b                                                   # different seed, different sample
    assert a.msprt_false_positive_rate <= 0.05 < a.naive_false_positive_rate
    assert b.msprt_false_positive_rate <= 0.05 < b.naive_false_positive_rate
    assert one.naive_false_positive_rate == one.naive_single_look_rate < 0.10   # 1 peek == fixed horizon: no inflation
    with pytest.raises(ValueError):
        mixture_likelihood_ratio(0.1, 0.0, 0.01)


# ============================================================================ Thompson sampling (Normal-Inverse-Gamma)
def test_nig_update_is_order_free_and_matches_the_conjugate_closed_form():
    rng = random.Random(4)
    xs = [rng.gauss(7.0, 3.0) for _ in range(500)]
    batch, split = NIGArm(), NIGArm()
    batch.update(xs)
    split.update(xs[:137]); split.update(xs[137:])
    for f in ("m", "k", "a", "b"):
        assert getattr(batch, f) == pytest.approx(getattr(split, f), rel=1e-9)   # sufficient statistics compose
    k0, m0, a0, b0, n, xbar = 0.01, 10.0, 2.0, 50.0, len(xs), statistics.fmean(xs)
    assert batch.k == pytest.approx(k0 + n) and batch.a == pytest.approx(a0 + n / 2)
    assert batch.m == pytest.approx((k0 * m0 + n * xbar) / (k0 + n))
    ss = sum((x - xbar) ** 2 for x in xs)
    assert batch.b == pytest.approx(b0 + ss / 2 + k0 * n * (xbar - m0) ** 2 / (2 * (k0 + n)))
    assert batch.b / (batch.a - 1) == pytest.approx(9.0, rel=0.15)   # posterior mean of sigma^2 recovers the truth


def test_nig_sampler_draws_from_the_student_t_marginal():
    arm = NIGArm(m=3.0, k=4.0, a=6.0, b=10.0)
    rng = random.Random(0)
    draws = [arm.sample(rng) for _ in range(60_000)]
    assert statistics.fmean(draws) == pytest.approx(3.0, abs=0.01)
    # marginal of mu is Student-t(2a) with variance b / (k (a - 1))
    assert statistics.pvariance(draws) == pytest.approx(10.0 / (4.0 * 5.0), rel=0.04)


def test_thompson_sampling_concentrates_on_the_best_arm_and_explores_first():
    truth = {2.0: 9.0, 6.0: 10.0, 14.0: 6.0}
    ts = ThompsonFloorTuner(list(truth), seed=1)
    rng = random.Random(1)
    for _ in range(300):
        i = ts.choose()
        ts.update(i, [rng.gauss(truth[ts.floors[i]], 4.0) for _ in range(50)])
    assert ts.best() == 6.0
    assert len(set(ts.history[:30])) == 3                               # every arm tried early
    assert sum(ts.floors[i] == 6.0 for i in ts.history[-100:]) >= 85    # exploitation late
    # swap the truth: the same code must find the other arm (it is not returning a constant)
    truth2 = {2.0: 11.0, 6.0: 10.0, 14.0: 6.0}
    ts2 = ThompsonFloorTuner(list(truth2), seed=1)
    for _ in range(300):
        i = ts2.choose()
        ts2.update(i, [rng.gauss(truth2[ts2.floors[i]], 4.0) for _ in range(50)])
    assert ts2.best() == 2.0


def test_linear_thompson_sampling_linear_algebra_and_posterior():
    rng = random.Random(9)
    a = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(6)]
    spd = [[sum(a[i][k] * a[j][k] for k in range(6)) + (2.0 if i == j else 0.0) for j in range(6)] for i in range(6)]
    inv, chol = _inverse(spd), _cholesky(spd)
    for i in range(6):
        for j in range(6):
            assert sum(spd[i][k] * inv[k][j] for k in range(6)) == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)
            assert sum(chol[i][k] * chol[j][k] for k in range(6)) == pytest.approx(spd[i][j], abs=1e-9)
    # one-hot design => ridge posterior mean has the closed form sum(y) / (lam + n) per arm
    b = ContextualLTVBandit(3, lam=1.0, v=0.5, seed=0)
    for arm, ys in enumerate(([10.0] * 4, [20.0] * 9, [])):
        for y in ys:
            b.observe([1.0 if k == arm else 0.0 for k in range(3)], y)
    b.refresh()
    assert b.mu == pytest.approx([40 / 5, 180 / 10, 0.0])
    th = [b.sample_theta() for _ in range(20_000)]
    for arm, n in enumerate((4, 9, 0)):   # posterior variance v^2 / (lam + n): the unseen arm is the widest
        assert statistics.pvariance([t[arm] for t in th]) == pytest.approx(0.25 / (1 + n), rel=0.06)
    assert len(features("meta", "T1", "ios")) == 29 and sum(features("meta", "T1", "ios")) == 5.0


# ============================================================================ bid shading (first-price BNE)
@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_bid_shading_matches_uniform_closed_form_without_and_with_reserve(n):
    cdf = lambda x: min(1.0, max(0.0, x))
    sh = BidShader(cdf, n, 0.0, 1.0)
    for v in (0.1, 0.35, 0.8, 1.0):
        assert sh(v) == pytest.approx(uniform_bne(v, n), abs=1e-5) == pytest.approx((n - 1) / n * v, abs=1e-5)
    r = 0.3
    shr = BidShader(cdf, n, r, 1.0)
    for v in (0.3, 0.45, 0.8, 1.0):   # U[0,1] with reserve r: beta(v) = v - (v^n - r^n) / (n v^(n-1))
        assert shr(v) == pytest.approx(v - (v ** n - r ** n) / (n * v ** (n - 1)), abs=1e-5)
    assert shr(r) == pytest.approx(r) and shr(r - 0.01) is None       # bid the reserve at the reserve; no bid below


def test_bid_shading_matches_an_exponential_closed_form_and_is_a_valid_strategy():
    # Exp(1) values, N = 2: beta(v) = v - (v - (1 - e^-v)) / (1 - e^-v) = 1 - v e^-v / (1 - e^-v)
    sh = BidShader(lambda x: 1 - math.exp(-x) if x > 0 else 0.0, 2, 0.0, 12.0, grid_n=20_000)
    for v in (0.5, 1.0, 3.0, 7.0):
        assert sh(v) == pytest.approx(1 - v * math.exp(-v) / (1 - math.exp(-v)), abs=2e-4)
    bids = [sh(0.05 * i) for i in range(1, 200)]
    assert all(0 <= b <= 0.05 * i + 1e-12 for i, b in enumerate(bids, 1))     # never bid above value
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(bids, bids[1:]))             # monotone in value
    more = BidShader(lambda x: 1 - math.exp(-x) if x > 0 else 0.0, 6, 0.0, 12.0, grid_n=20_000)
    assert more(2.0) > sh(2.0)                                               # more competition, less shading
    with pytest.raises(ValueError):
        BidShader(lambda x: x, 1)


def test_revenue_equivalence_holds_for_the_shaded_first_price_auction():
    for kw in ({"n_bidders": 5, "reserve": 4.0, "seed": 3}, {"n_bidders": 3, "reserve": 0.5, "seed": 8}):
        r = revenue_equivalence_check(auctions=40_000, **kw)
        assert abs(r["z"]) < 3.0, r                                    # same expected revenue (paired z-test)
        assert r["second_price_cpm"] > kw["reserve"]
    # truthful first-price bidding must NOT be revenue-equivalent: the test above can fail
    rng = random.Random(0)
    truthful = second = 0.0
    for _ in range(20_000):
        v = sorted((math.exp(rng.gauss(math.log(8.0), 0.5)) for _ in range(5)), reverse=True)
        truthful += v[0]; second += v[1]
    assert truthful / second > 1.15


# ============================================================================ Cohen's kappa
def test_cohens_kappa_textbook_tables():
    def expand(a, b, c, d):   # a = both yes, b = rater1 yes / rater2 no, c = rater1 no / rater2 yes, d = both no
        return (["yes"] * (a + b) + ["no"] * (c + d), ["yes"] * a + ["no"] * b + ["yes"] * c + ["no"] * d)
    y1, y2 = expand(20, 5, 10, 15)                                    # po = 0.70, pe = 0.50
    assert cohens_kappa(y1, y2) == pytest.approx(0.4)
    y1, y2 = expand(45, 15, 25, 15)                                   # po = 0.60, pe = 0.54
    assert cohens_kappa(y1, y2) == pytest.approx((0.60 - 0.54) / (1 - 0.54))
    assert cohens_kappa(y1, y2) == pytest.approx(cohens_kappa(y2, y1))             # symmetric
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohens_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == pytest.approx(-1.0)
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)   # chance-level agreement
    # 90% raw agreement on a 90/10 class split is NOT good agreement: kappa corrects for prevalence
    assert cohens_kappa(["a"] * 90 + ["b"] * 10, ["a"] * 100) == pytest.approx(0.0)
    three = (["x", "y", "z", "x", "y", "z", "x", "x"], ["x", "y", "z", "y", "y", "x", "x", "x"])
    po, pe = 6 / 8, (4 / 8) * (4 / 8) + (2 / 8) * (3 / 8) + (2 / 8) * (1 / 8)
    assert cohens_kappa(*three) == pytest.approx((po - pe) / (1 - pe))             # multi-class
    with pytest.raises(ValueError):
        cohens_kappa([], [])
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])
