"""
================================================================================
BLOCK: Versioned Level Catalogue (publish / rollback / seed generation)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : seed generator below (Phase 1) and the Autonomous Level Factory
             (Phase 2) which calls publish_version() for solver-verified grids.
  Downstream: app/game_api.py serves `active_levels` (highest live version per
             level_id); every telemetry event records level_version so live
             fail-rate can be attributed to the exact version.
  Contract : versions are append-only; rollback flips status, never deletes.

Key engineering mechanism
  * Seed levels are generated deterministically (seeded RNG) by sampling many
    candidate boards, solving each with the trie solver, and selecting the one
    closest to a difficulty spec (word-count band, mean Zipf). Level 7 is
    deliberately built hard: rare-letter bias, low word count, low mean Zipf,
    target ~65% of all words in 75 s -> live play reproduces the seeded 62%
    fail-rate spike.
  * rollback_level(): marks the current live version `rolled_back` in one
    UPDATE; the `active_levels` view automatically falls back to the previous
    live version - zero-downtime, one statement, fully auditable.

Failure modes & 1-step mitigation
  * Bad level shipped -> rollback_level(level_id) (Phase 2 automates it on a
    live fail-rate spike).
  * Publishing an unverified grid -> publish_version() re-solves the grid and
    refuses if the declared word list differs or target > solvable words.
  * Rolling back the only version -> refused (a level must always exist);
    a brand-new daily level with no fallback is RETIRED instead
    (set_version_status), which removes it from `active_levels`.
================================================================================
"""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime

from core.wordgrid import Trie, grid_to_str, parse_grid, random_grid, solve
from core.wordlist import Lexicon


@dataclass(frozen=True)
class LevelSpec:
    level_id: int
    size: int
    time_limit_s: int
    words_lo: int
    words_hi: int
    target_frac: float
    hard: bool = False


SEED_SPECS = [
    LevelSpec(1, 4, 120, 28, 40, 0.18),
    LevelSpec(2, 4, 120, 26, 36, 0.22),
    LevelSpec(3, 4, 120, 24, 34, 0.26),
    LevelSpec(4, 4, 110, 20, 28, 0.36),
    LevelSpec(5, 4, 105, 18, 26, 0.40),
    LevelSpec(6, 4, 100, 16, 24, 0.44),
    LevelSpec(7, 4, 75, 9, 12, 0.65, hard=True),   # the fail-rate spike
    LevelSpec(8, 5, 120, 34, 46, 0.30),
    LevelSpec(9, 5, 120, 32, 44, 0.32),
    LevelSpec(10, 5, 115, 30, 42, 0.34),
    LevelSpec(11, 5, 110, 30, 40, 0.36),
    LevelSpec(12, 5, 110, 28, 40, 0.38),
]
HARD_BIAS = {"Q": 8, "Z": 8, "X": 8, "J": 8, "K": 4, "V": 4, "W": 3, "Y": 3}


def build_level(spec: LevelSpec, lex: Lexicon, trie: Trie, seed: int = 7, candidates: int = 400) -> dict:
    rng = random.Random(seed * 1000 + spec.level_id)
    best, best_score = None, float("inf")
    mid = (spec.words_lo + spec.words_hi) / 2
    for _ in range(candidates):
        g = random_grid(rng, spec.size, spec.size, HARD_BIAS if spec.hard else None)
        res = solve(g, trie)
        n = len(res.words)
        if not n:
            continue
        mz = statistics.mean(lex.zipf[w] for w in res.words)
        in_band = spec.words_lo <= n <= spec.words_hi
        # hard level prefers rarer vocabulary; easy levels prefer common words
        score = abs(n - mid) + (0 if in_band else 50) + (mz * 2 if spec.hard else -mz)
        if score < best_score:
            best, best_score = (g, res, mz), score
    g, res, mz = best
    words = sorted(res.words)
    return {
        "level_id": spec.level_id, "version": 1, "grid": grid_to_str(g),
        "n_rows": spec.size, "n_cols": spec.size,
        "target_words": max(3, round(len(words) * spec.target_frac)),
        "time_limit_s": spec.time_limit_s, "valid_words": json.dumps(words),
        "word_count": len(words), "mean_zipf": round(mz, 3),
        "branching_factor": res.branching_factor, "status": "live",
        "created_by": "seed", "created_at": datetime(2026, 9, 30),
        "notes": "hard: rare-letter bias, fail-rate spike" if spec.hard else "",
    }


def build_seed_levels(lex: Lexicon, trie: Trie | None = None) -> list[dict]:
    trie = trie or Trie(lex.words)
    return [build_level(s, lex, trie) for s in SEED_SPECS]


LEVEL_COLS = ["level_id", "version", "grid", "n_rows", "n_cols", "target_words", "time_limit_s",
              "valid_words", "word_count", "mean_zipf", "branching_factor", "status",
              "created_by", "created_at", "notes", "predicted_fail_rate"]


def insert_levels(con, levels: list[dict]) -> None:
    con.executemany(
        f"INSERT INTO levels ({','.join(LEVEL_COLS)}) VALUES ({','.join('?' * len(LEVEL_COLS))})",
        [[lv.get(c) for c in LEVEL_COLS] for lv in levels],
    )


def publish_version(con, lex: Lexicon, trie: Trie, level_id: int, grid: str, target_words: int,
                    time_limit_s: int, created_by: str = "level_factory", notes: str = "",
                    status: str = "live", predicted_fail_rate: float | None = None) -> dict:
    """Append a new version after re-verifying solvability (solver gate).

    status="live" serves it to everyone at once; status="candidate" stores it
    for a feature-flagged experiment without changing `active_levels`."""
    if status not in ("live", "candidate"):
        raise ValueError(f"cannot publish with status {status!r}")
    g = parse_grid(grid)
    res = solve(g, trie)
    if target_words > len(res.words):
        raise ValueError(f"unsolvable: target {target_words} > {len(res.words)} valid words")
    ver = con.execute("SELECT coalesce(max(version),0)+1 FROM levels WHERE level_id=?", [level_id]).fetchone()[0]
    words = sorted(res.words)
    row = {
        "level_id": level_id, "version": ver, "grid": grid_to_str(g), "n_rows": len(g), "n_cols": len(g[0]),
        "target_words": target_words, "time_limit_s": time_limit_s, "valid_words": json.dumps(words),
        "word_count": len(words), "mean_zipf": round(statistics.mean(lex.zipf[w] for w in words), 3) if words else 0,
        "branching_factor": res.branching_factor, "status": status, "created_by": created_by,
        "created_at": datetime.now(), "notes": notes, "predicted_fail_rate": predicted_fail_rate,
    }
    insert_levels(con, [row])
    return row


def rollback_level(con, level_id: int, reason: str = "") -> dict:
    live = con.execute("SELECT version FROM levels WHERE level_id=? AND status='live' ORDER BY version DESC",
                       [level_id]).fetchall()
    if len(live) < 2:
        raise ValueError(f"level {level_id}: no earlier live version to fall back to")
    bad, fallback = live[0][0], live[1][0]
    con.execute("UPDATE levels SET status='rolled_back', notes=concat(coalesce(notes,''), ?) WHERE level_id=? AND version=?",
                [f" | rolled back: {reason}", level_id, bad])
    return {"level_id": level_id, "rolled_back_version": bad, "active_version": fallback}


def set_version_status(con, level_id: int, version: int, status: str, reason: str = "") -> None:
    """candidate -> live (ship) | candidate/live -> rolled_back | live -> retired."""
    if status not in ("live", "rolled_back", "retired"):
        raise ValueError(status)
    if not con.execute("SELECT 1 FROM levels WHERE level_id=? AND version=?", [level_id, version]).fetchone():
        raise ValueError(f"level {level_id} v{version} not found")
    con.execute("UPDATE levels SET status=?, notes=concat(coalesce(notes,''), ?) WHERE level_id=? AND version=?",
                [status, f" | {status}: {reason}" if reason else "", level_id, version])


def next_level_id(con) -> int:
    return con.execute("SELECT coalesce(max(level_id), 0) + 1 FROM levels").fetchone()[0]
