"""
================================================================================
BLOCK: Difficulty Model (predicts per-attempt fail rate of a level)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : SolverProof (word list, branching factor) + level config
             (target word count, time limit).
  Downstream: generator (pick a target count that hits a difficulty goal),
             critic rubric 2 (difficulty fit), the publish ledger (stored in
             levels.predicted_fail_rate), and the live auto-rollback monitor
             (observed vs predicted).

Key engineering mechanism: mechanistic player model, calibrated on telemetry
  * Each word w is discovered as a Poisson process with rate
        lambda_w = exp(b0 + b_zipf (zipf_w - 5) - b_len (len_w - 4) + b_bf (bf - 0.9) + s)
    where s ~ N(0, sigma^2) is player skill. Common words are found fast,
    long words slowly, and dense boards (high branching) help.
  * P(found by T | s) = 1 - exp(-lambda_w T). The number of words found
    is Poisson-binomial, computed EXACTLY by dynamic programming (truncated
    at the target). Then
        fail(s) = P(found < target | s)
        fail    = E_s[fail(s)]   (15-node equal-mass skill quadrature)
  * calibrate(): Nelder-Mead on the 5 parameters, minimising the
    attempt-weighted squared error between predicted and OBSERVED fail
    rates from data/telemetry.duckdb (one point per level version), with a
    weak ridge pull toward the priors so a 12-point fit stays stable.
  * Deterministic and fast (~0.2 ms per level), so the generator can score
    hundreds of candidate (grid, target) pairs per second.

Failure modes & 1-step mitigation
  * Too little telemetry to calibrate (< 3 level versions with >= 50
    attempts) -> keep the prior parameters and report calibrated=False.
  * Model drift on new level styles -> the live monitor compares observed
    vs predicted per shipped version and retires or rolls back outliers;
    re-running calibrate() picks up the new evidence.
================================================================================
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Callable, Sequence

from core.wordlist import Lexicon

_SKILL_Q = [NormalDist().inv_cdf((i + 0.5) / 15) for i in range(15)]


@dataclass
class DifficultyParams:
    b0: float = -3.2
    b_zipf: float = 1.1
    b_len: float = 0.45
    b_bf: float = 1.5
    sigma: float = 0.9

    def vector(self) -> list[float]:
        return [self.b0, self.b_zipf, self.b_len, self.b_bf, self.sigma]

    @classmethod
    def from_vector(cls, v: Sequence[float]) -> "DifficultyParams":
        return cls(v[0], v[1], v[2], v[3], abs(v[4]))


@dataclass
class LevelFeatures:
    zipfs: list[float]
    lengths: list[int]
    target: int
    time_limit_s: float
    branching_factor: float

    @classmethod
    def from_words(cls, words: Sequence[str], lex: Lexicon, target: int, time_limit_s: float, bf: float) -> "LevelFeatures":
        return cls([lex.zipf.get(w, 3.0) for w in words], [len(w) for w in words], target, time_limit_s, bf)

    def describe(self) -> dict:
        easy = sorted(self.zipfs, reverse=True)[: self.target]
        return {
            "words": len(self.zipfs), "target": self.target, "time_limit_s": self.time_limit_s,
            "target_ratio": round(self.target / len(self.zipfs), 3) if self.zipfs else None,
            "seconds_per_required_word": round(self.time_limit_s / max(1, self.target), 1),
            "required_words_mean_zipf": round(sum(easy) / len(easy), 3) if easy else None,
            "branching_factor": self.branching_factor,
        }


def _poisson_binomial_below(ps: Sequence[float], target: int) -> float:
    """P(sum Bernoulli(ps) < target), DP truncated at `target`."""
    if target <= 0:
        return 0.0
    dp = [1.0] + [0.0] * target  # dp[k] = P(count == k), dp[target] = P(count >= target)
    for p in ps:
        for k in range(target, 0, -1):
            if k == target:
                dp[k] = dp[k] + dp[k - 1] * p
            else:
                dp[k] = dp[k] * (1 - p) + dp[k - 1] * p
        dp[0] *= 1 - p
    return sum(dp[:target])


class DifficultyModel:
    def __init__(self, params: DifficultyParams | None = None):
        self.params = params or DifficultyParams()
        self.calibration: dict = {"calibrated": False}

    def predict(self, f: LevelFeatures) -> float:
        if f.target > len(f.zipfs):
            return 1.0
        p = self.params
        base = [p.b0 + p.b_zipf * (z - 5) - p.b_len * (n - 4) + p.b_bf * (f.branching_factor - 0.9)
                for z, n in zip(f.zipfs, f.lengths)]
        fails = []
        for q in _SKILL_Q:
            s = p.sigma * q
            probs = [1.0 - math.exp(-math.exp(min(b + s, 30.0)) * f.time_limit_s) for b in base]
            fails.append(_poisson_binomial_below(probs, f.target))
        return min(1.0, max(0.0, sum(fails) / len(fails)))

    def best_target(self, f: LevelFeatures, goal: float, lo: int = 3, hi: int | None = None) -> tuple[int, float]:
        """Integer target word count whose predicted fail rate is closest to `goal`."""
        hi = min(hi or len(f.zipfs), len(f.zipfs))
        best = (lo, 1.0, math.inf)
        for t in range(lo, hi + 1):
            pred = self.predict(LevelFeatures(f.zipfs, f.lengths, t, f.time_limit_s, f.branching_factor))
            if abs(pred - goal) < best[2]:
                best = (t, pred, abs(pred - goal))
            if pred > goal + 0.25:
                break  # fail rate is monotone in target
        return best[0], best[1]

    # ------------------------------------------------------------------ calibration
    def calibrate(self, levels: Sequence[LevelFeatures], observed: Sequence[float], weights: Sequence[float],
                  ridge: float = 0.02, iters: int = 600) -> dict:
        prior = DifficultyParams().vector()
        wsum = sum(weights)

        def loss(v: Sequence[float]) -> float:
            m = DifficultyModel(DifficultyParams.from_vector(v))
            err = sum(w * (m.predict(f) - o) ** 2 for f, o, w in zip(levels, observed, weights)) / wsum
            return err + ridge * sum((a - b) ** 2 for a, b in zip(v, prior)) / len(v)

        before = [self.predict(f) for f in levels]
        best_v, best_loss = nelder_mead(loss, self.params.vector(), step=0.3, iters=iters)
        self.params = DifficultyParams.from_vector(best_v)
        after = [self.predict(f) for f in levels]
        mae = lambda preds: sum(w * abs(p - o) for p, o, w in zip(preds, observed, weights)) / wsum
        self.calibration = {"calibrated": True, "points": len(levels), "mae_before": round(mae(before), 4),
                            "mae_after": round(mae(after), 4), "loss": round(best_loss, 6),
                            "params": {k: round(v, 4) for k, v in asdict(self.params).items()}}
        return self.calibration

    def to_json(self) -> str:
        return json.dumps({"params": asdict(self.params), "calibration": self.calibration})


def nelder_mead(f: Callable[[Sequence[float]], float], x0: Sequence[float], step: float = 0.5, iters: int = 500,
                tol: float = 1e-9) -> tuple[list[float], float]:
    """Plain Nelder-Mead (reflection 1, expansion 2, contraction 0.5, shrink 0.5)."""
    n = len(x0)
    simplex = [list(x0)] + [[x0[j] + (step if j == i else 0.0) for j in range(n)] for i in range(n)]
    vals = [f(x) for x in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=vals.__getitem__)
        simplex, vals = [simplex[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = [sum(x[j] for x in simplex[:-1]) / n for j in range(n)]
        refl = [c + (c - w) for c, w in zip(centroid, simplex[-1])]
        fr = f(refl)
        if fr < vals[0]:
            exp_ = [c + 2 * (c - w) for c, w in zip(centroid, simplex[-1])]
            fe = f(exp_)
            simplex[-1], vals[-1] = (exp_, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = refl, fr
        else:
            con = [c + 0.5 * (w - c) for c, w in zip(centroid, simplex[-1])]
            fc = f(con)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = con, fc
            else:
                best = simplex[0]
                simplex = [best] + [[b + 0.5 * (x - b) for b, x in zip(best, s)] for s in simplex[1:]]
                vals = [vals[0]] + [f(s) for s in simplex[1:]]
    i = min(range(n + 1), key=vals.__getitem__)
    return simplex[i], vals[i]


def observed_level_stats(con, min_attempts: int = 50) -> list[dict]:
    """Per (level_id, version): attempts + observed fail rate + its config, from DuckDB."""
    rows = con.execute("""
        WITH f AS (
            SELECT level_id, level_version AS version, count(*) AS attempts,
                   avg((event_name = 'level_fail')::INT) AS fail_rate
            FROM events WHERE event_name IN ('level_fail', 'level_complete') AND level_version IS NOT NULL
            GROUP BY 1, 2)
        SELECT l.level_id, l.version, l.valid_words, l.target_words, l.time_limit_s, l.branching_factor,
               f.attempts, f.fail_rate
        FROM levels l JOIN f USING (level_id, version)
        WHERE f.attempts >= ? ORDER BY 1, 2""", [min_attempts]).fetchall()
    return [{"level_id": r[0], "version": r[1], "words": json.loads(r[2]), "target": r[3], "time_limit_s": r[4],
             "branching_factor": r[5], "attempts": r[6], "fail_rate": r[7]} for r in rows]


def calibrated_model(con, lex: Lexicon, min_attempts: int = 50) -> tuple[DifficultyModel, list[dict]]:
    stats = observed_level_stats(con, min_attempts)
    model = DifficultyModel()
    if len(stats) >= 3:
        feats = [LevelFeatures.from_words(s["words"], lex, s["target"], s["time_limit_s"], s["branching_factor"]) for s in stats]
        model.calibrate(feats, [s["fail_rate"] for s in stats], [math.sqrt(s["attempts"]) for s in stats])
        for s, f in zip(stats, feats):
            s["predicted"] = round(model.predict(f), 4)
    return model, stats
