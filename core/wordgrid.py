"""
================================================================================
BLOCK: Word-Grid Engine (Trie + backtracking solver + path referee)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : Lexicon (core/wordlist.py), grid strings "ABCD/EFGH/IJKL/MNOP".
  Downstream: app/game_api.py (server-side referee: validate_path), the level
             seeder (solve -> exact valid word list per level), and Phase-2
             Level Factory solvability gate (same solver = one source of truth).
  Contract : 8-neighbour adjacency (incl. diagonals), each tile used at most
             once per word, "Q" tile is a literal Q (no auto-"QU").

Key engineering mechanism
  * Prefix Trie over the lexicon; DFS from every tile prunes as soon as the
    current path is not a trie prefix -> visits only live prefixes, so a 5x5
    board solves in ~1 ms instead of 25*8^6 naive paths.
  * Referee validate_path(): bounds -> adjacency -> no-reuse -> dictionary,
    with a machine-readable reason code for every rejection (telemetry-ready).
  * branching_factor(): mean live-trie children per DFS node - a structural
    difficulty signal reused by the Phase-2 difficulty model.

Failure modes & 1-step mitigation
  * Client forges a word not on the board -> server recomputes the word from
    the submitted coordinates; client-sent text is never trusted.
  * Malformed grid from a generator (ragged rows, non-letters) -> parse_grid
    raises ValueError; the Level Factory treats it as a hard reject.
  * Pathological input (huge path) -> length capped at rows*cols before work.
================================================================================
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

# English letter frequencies (Lewand) used for board generation.
LETTER_FREQ = {
    "E": 12.7, "T": 9.1, "A": 8.2, "O": 7.5, "I": 7.0, "N": 6.7, "S": 6.3,
    "H": 6.1, "R": 6.0, "D": 4.3, "L": 4.0, "C": 2.8, "U": 2.8, "M": 2.4,
    "W": 2.4, "F": 2.2, "G": 2.0, "Y": 2.0, "P": 1.9, "B": 1.5, "V": 1.0,
    "K": 0.8, "J": 0.15, "X": 0.15, "Q": 0.1, "Z": 0.07,
}
VOWELS = set("AEIOU")


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    terminal: bool = False


class Trie:
    def __init__(self, words: Iterable[str] = ()):
        self.root = TrieNode()
        self.size = 0
        for w in words:
            self.insert(w)

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word.upper():
            node = node.children.setdefault(ch, TrieNode())
        if not node.terminal:
            node.terminal = True
            self.size += 1


Grid = list[list[str]]


def parse_grid(grid: str) -> Grid:
    rows = [r.strip().upper() for r in grid.strip().split("/")]
    if not rows or any(not r.isalpha() for r in rows) or len({len(r) for r in rows}) != 1:
        raise ValueError(f"malformed grid {grid!r}: rows must be equal-length A-Z strings")
    return [list(r) for r in rows]


def grid_to_str(g: Grid) -> str:
    return "/".join("".join(r) for r in g)


def neighbours(r: int, c: int, rows: int, cols: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc


@dataclass
class SolveResult:
    words: dict[str, list[tuple[int, int]]]  # word -> one witness path
    nodes_visited: int
    branching_factor: float


def solve(grid: Grid, trie: Trie, min_len: int = 3) -> SolveResult:
    """Exhaustive trie-pruned DFS. Returns every dictionary word on the board."""
    rows, cols = len(grid), len(grid[0])
    found: dict[str, list[tuple[int, int]]] = {}
    visited = [[False] * cols for _ in range(rows)]
    stats = {"nodes": 0, "children": 0}

    def dfs(r: int, c: int, node: TrieNode, path: list[tuple[int, int]], letters: list[str]):
        stats["nodes"] += 1
        if node.terminal and len(letters) >= min_len:
            w = "".join(letters).lower()
            found.setdefault(w, list(path))
        live = 0
        for nr, nc in neighbours(r, c, rows, cols):
            if visited[nr][nc]:
                continue
            child = node.children.get(grid[nr][nc])
            if child is None:
                continue
            live += 1
            visited[nr][nc] = True
            path.append((nr, nc)); letters.append(grid[nr][nc])
            dfs(nr, nc, child, path, letters)
            path.pop(); letters.pop()
            visited[nr][nc] = False
        stats["children"] += live

    for r in range(rows):
        for c in range(cols):
            child = trie.root.children.get(grid[r][c])
            if child is None:
                continue
            visited[r][c] = True
            dfs(r, c, child, [(r, c)], [grid[r][c]])
            visited[r][c] = False
    bf = stats["children"] / stats["nodes"] if stats["nodes"] else 0.0
    return SolveResult(words=found, nodes_visited=stats["nodes"], branching_factor=round(bf, 4))


@dataclass(frozen=True)
class PathVerdict:
    ok: bool
    word: str
    reason: str  # "ok" | "too_short" | "out_of_bounds" | "not_adjacent" | "tile_reused" | "not_in_dictionary"


def validate_path(grid: Grid, path: list[tuple[int, int]] | list[list[int]], lexicon_words: frozenset[str] | set[str], min_len: int = 3) -> PathVerdict:
    """Server-side referee. The word is derived from coordinates only."""
    rows, cols = len(grid), len(grid[0])
    pts = [tuple(p) for p in path][: rows * cols + 1]
    if len(pts) > rows * cols:
        return PathVerdict(False, "", "tile_reused")
    for p in pts:
        if len(p) != 2 or not (0 <= p[0] < rows and 0 <= p[1] < cols):
            return PathVerdict(False, "", "out_of_bounds")
    word = "".join(grid[r][c] for r, c in pts).lower()
    if len(pts) < min_len:
        return PathVerdict(False, word, "too_short")
    if len(set(pts)) != len(pts):
        return PathVerdict(False, word, "tile_reused")
    for (r1, c1), (r2, c2) in zip(pts, pts[1:]):
        if max(abs(r1 - r2), abs(c1 - c2)) != 1:
            return PathVerdict(False, word, "not_adjacent")
    if word not in lexicon_words:
        return PathVerdict(False, word, "not_in_dictionary")
    return PathVerdict(True, word, "ok")


def random_grid(rng: random.Random, rows: int, cols: int, letter_bias: dict[str, float] | None = None,
                min_vowels: int | None = None) -> Grid:
    """Frequency-weighted letters; guarantees a vowel floor so boards are playable."""
    weights = dict(LETTER_FREQ)
    for k, v in (letter_bias or {}).items():
        weights[k] = weights.get(k, 0.1) * v
    letters, w = zip(*weights.items())
    need = min_vowels if min_vowels is not None else max(2, (rows * cols) // 4)
    while True:
        cells = rng.choices(letters, weights=w, k=rows * cols)
        if sum(ch in VOWELS for ch in cells) >= need:
            return [cells[i * cols:(i + 1) * cols] for i in range(rows)]
