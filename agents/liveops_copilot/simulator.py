"""
================================================================================
BLOCK: Experiment Traffic Simulator (bootstrapped from seeded players)
--------------------------------------------------------------------------------
Role & contracts
  Produces realistic daily exposure batches for a feature-flagged experiment
  when real traffic is too small (the seeded cohort is a 2,500-player
  sample; production DAU is a parameter). Rows land in
  `experiment_exposures` with source='sim', next to live exposures logged by
  app/game_api.py, and the Copilot's evaluator reads both identically.

Key engineering mechanism
  * Joint (pre, post, D1) triples are BOOTSTRAPPED from real seeded players
    (pre = level completions on days 0-6, post = days 7-13, D1 retention),
    so the CUPED correlation and the outcome distribution are the data's own,
    not assumed.
  * Treatment effects are applied as count thinning or augmentation
    (Binomial(Y, 1 + lift) for a drop, Y + Poisson(lift * Y) for a lift) and
    as a D1 retention flip with exact marginal shift delta.
  * Deterministic per (seed, flag, day).

Failure modes & 1-step mitigation
  * Seeded DB missing the cohort -> clear error naming `make seed`.
  * Simulated rows polluting real analysis -> always tagged source='sim';
    scripts/verify_phase2.py --reset deletes them.
================================================================================
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta

from core.db import TelemetryDB, bulk_insert

EXPOSURE_COLUMNS = ["flag", "user_id", "variant", "day", "exposed_at", "pre_metric", "outcome", "retained_d1", "source"]

COHORT_SQL = """
WITH first AS (SELECT user_id, min(ts) AS t0 FROM events WHERE source = 'seed' AND event_name = 'session_start' GROUP BY 1)
SELECT f.user_id,
       count(*) FILTER (WHERE e.event_name = 'level_complete' AND e.ts <  f.t0 + INTERVAL 7 DAY) AS pre,
       count(*) FILTER (WHERE e.event_name = 'level_complete' AND e.ts >= f.t0 + INTERVAL 7 DAY
                                                              AND e.ts <  f.t0 + INTERVAL 14 DAY) AS post,
       any_value(p.retention_1) AS d1
FROM first f JOIN events e USING (user_id) JOIN players p USING (user_id)
GROUP BY 1 ORDER BY 1"""


def load_cohort(db: TelemetryDB) -> list[tuple[float, float, bool]]:
    with db.cursor() as con:
        rows = con.execute(COHORT_SQL).fetchall()
    if len(rows) < 100:
        raise RuntimeError("seeded cohort missing - run `make seed` first")
    return [(float(r[1]), float(r[2]), bool(r[3])) for r in rows]


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


@dataclass(frozen=True)
class Effect:
    primary_lift: float = 0.0   # relative change in 7-day level completions
    d1_delta: float = 0.0       # absolute change in D1 retention probability


class ExperimentSimulator:
    def __init__(self, db: TelemetryDB, seed: int = 0):
        self.db, self.seed = db, seed
        self.cohort = load_cohort(db)
        self.p_d1 = sum(c[2] for c in self.cohort) / len(self.cohort)

    def simulate_day(self, flag: str, day: int, users: int, treatment_share: float, effect: Effect,
                     start: datetime | None = None) -> dict:
        rng = random.Random(zlib.crc32(f"{self.seed}:{flag}:{day}".encode()))
        t0 = (start or datetime(2026, 11, 10)) + timedelta(days=day - 1)
        rows, n_t = [], 0
        for i in range(users):
            pre, post, d1 = rng.choice(self.cohort)
            variant = "treatment" if rng.random() < treatment_share else "control"
            if variant == "treatment":
                n_t += 1
                if effect.primary_lift >= 0:
                    post = post + _poisson(rng, effect.primary_lift * post)
                else:
                    keep = 1 + effect.primary_lift
                    post = float(sum(rng.random() < keep for _ in range(int(post))))
                if effect.d1_delta < 0 and d1 and rng.random() < -effect.d1_delta / self.p_d1:
                    d1 = False
                elif effect.d1_delta > 0 and not d1 and rng.random() < effect.d1_delta / (1 - self.p_d1):
                    d1 = True
            rows.append([flag, f"x_{flag}_{day}_{i}", variant, day, t0 + timedelta(seconds=i), pre, post, d1, "sim"])
        with self.db.cursor() as con:
            bulk_insert(con, "experiment_exposures", EXPOSURE_COLUMNS, rows)
        return {"day": day, "users": users, "treatment": n_t, "control": users - n_t}
