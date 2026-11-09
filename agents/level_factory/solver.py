"""
================================================================================
BLOCK: Level Solver & Solvability Prover (Level Factory gate #1)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : any grid string from the generator (LLM or heuristic) plus an
             optional list of target words the generator claims are present.
  Downstream: SolverProof -> difficulty_model (features), critic (rubric 1),
             publish gate (a level without ok=True proof can't be shipped).
  Contract : prove() never raises on bad input. A malformed grid returns
             ok=False with a reason, because generator output is untrusted.

Key engineering mechanism
  * Exhaustive Trie-pruned backtracking DFS (core.wordgrid.solve): the
    exact set of dictionary words on the board, each with a witness path.
    This is the same solver the live game referee relies on.
  * Independent re-verification: every witness path is re-checked by the
    REFEREE (validate_path: bounds, adjacency, no reuse, dictionary), a
    second implementation. A proof is only valid if both agree, so a
    solver bug cannot silently ship an unsolvable level.
  * Structural and lexical features: branching factor (mean live trie
    children per DFS node), nodes visited, Zipf rarity distribution, length
    histogram, share of rare words (Zipf < 5).

Failure modes & 1-step mitigation
  * Generator claims words that are not on the board -> targets_missing is
    non-empty, so ok=False and the candidate is rejected automatically.
  * Target count larger than the solvable word count -> ok=False.
  * Malformed grid (ragged rows, digits, empty) -> ok=False with the reason.
================================================================================
"""
from __future__ import annotations

import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from core.wordgrid import Trie, parse_grid, solve, validate_path
from core.wordlist import Lexicon, load_lexicon


@dataclass
class SolverProof:
    grid: str
    ok: bool
    reason: str
    rows: int = 0
    cols: int = 0
    valid_words: list[str] = field(default_factory=list)
    word_count: int = 0
    target_count: int | None = None
    targets: list[str] = field(default_factory=list)
    targets_missing: list[str] = field(default_factory=list)
    witness: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    witness_rechecked: bool = False
    branching_factor: float = 0.0
    nodes_visited: int = 0
    mean_zipf: float = 0.0
    min_zipf: float = 0.0
    rare_share: float = 0.0
    mean_len: float = 0.0
    by_length: dict[int, int] = field(default_factory=dict)
    solve_ms: float = 0.0

    @property
    def all_targets_present(self) -> bool:
        return not self.targets_missing

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in ("grid", "ok", "reason", "word_count", "target_count", "targets_missing",
                                              "witness_rechecked", "branching_factor", "mean_zipf", "rare_share",
                                              "solve_ms")}


class LevelSolver:
    def __init__(self, lex: Lexicon | None = None, trie: Trie | None = None):
        self.lex = lex or load_lexicon()
        self.trie = trie or Trie(self.lex.words)

    def prove(self, grid: str, targets: Sequence[str] | None = None, target_count: int | None = None) -> SolverProof:
        t0 = time.perf_counter()
        try:
            g = parse_grid(grid)
        except (ValueError, AttributeError) as e:
            return SolverProof(str(grid), False, f"malformed_grid: {e}")
        res = solve(g, self.trie)
        words = sorted(res.words)
        targets = [t.lower() for t in (targets or [])]
        missing = [t for t in targets if t not in res.words]
        to_check = targets if targets else words
        witness = {w: res.words[w] for w in to_check if w in res.words}
        rechecked = all(validate_path(g, p, self.lex.words).ok and validate_path(g, p, self.lex.words).word == w
                        for w, p in witness.items())
        z = [self.lex.zipf[w] for w in words]
        proof = SolverProof(
            grid="/".join("".join(r) for r in g), ok=True, reason="ok", rows=len(g), cols=len(g[0]),
            valid_words=words, word_count=len(words), target_count=target_count, targets=targets,
            targets_missing=missing, witness=witness, witness_rechecked=rechecked,
            branching_factor=res.branching_factor, nodes_visited=res.nodes_visited,
            mean_zipf=round(statistics.fmean(z), 3) if z else 0.0, min_zipf=min(z) if z else 0.0,
            rare_share=round(sum(v < 5.0 for v in z) / len(z), 3) if z else 0.0,
            mean_len=round(statistics.fmean(len(w) for w in words), 2) if words else 0.0,
            by_length=dict(sorted(Counter(len(w) for w in words).items())),
            solve_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
        if not words:
            proof.ok, proof.reason = False, "no_words_on_board"
        elif missing:
            proof.ok, proof.reason = False, f"targets_missing: {missing}"
        elif target_count is not None and target_count > len(words):
            proof.ok, proof.reason = False, f"unsolvable: target {target_count} > {len(words)} words"
        elif not rechecked:
            proof.ok, proof.reason = False, "witness_recheck_failed"
        return proof
