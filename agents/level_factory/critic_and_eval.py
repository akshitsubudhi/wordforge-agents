"""
================================================================================
BLOCK: Level Factory - Generator, 4-Rubric Critic, Golden Eval, Daily Shipper
--------------------------------------------------------------------------------
Role & contracts
  LevelGenerator  proposes grids (LLM when a real provider is configured,
                  otherwise a deterministic heuristic) and chooses the target
                  word count that hits a difficulty goal.
  CriticAgent     grades each candidate on 4 rubrics (1-5) and approves only
                  if every rubric scores >= 3.
  Golden eval     builds and loads a 100-level labelled dataset and scores the
                  critic against it (agreement %, precision, recall, F1,
                  Cohen's kappa, per-defect recall).
  LevelFactory    the harness-wired pipeline: generate -> prove (solver) ->
                  predict (difficulty model) -> critique -> publish through a
                  requires_approval tool behind a machine RELEASE GATE ->
                  monitor live fail rate -> retire / roll back.
  Downstream: the `levels` table. app/game_api.py serves `active_levels`, so
              a published level appears in game.html's level rotation on the
              next GET /api/game/levels. No code deploy is needed.

Key engineering mechanism
  * Every pipeline step is a tool call through AgentRunner.workflow(): full
    Pydantic validation, L1 allowlist, L3 gate, timeouts, spans in
    traces.duckdb. publish_level is requires_approval; the release-gate
    approver approves ONLY a grid whose solver proof and critic approval are
    in this run's ledger with the same target. The model cannot ship around
    the gates.
  * Rubrics: (1) solvability & fairness (2) difficulty fit vs the slot goal
    (3) lexical quality, including an OFFENSIVE-word trie solved over the board
    so profanity a player could trace along tile paths is caught (4) board quality: vowel
    balance, letter spam, rare-letter overload, novelty (Jaccard similarity
    of word sets vs every live level).
  * Golden labels are ground truth BY CONSTRUCTION: 54 clean levels (which
    satisfy the hard board spec: letter repeats, vowel share, rare letters)
    plus 46 with one injected defect each, across 9 defect types. Soft
    judgements (difficulty fit, vocabulary) are NOT pre-filtered, so the
    critic can and does disagree with the construction label. To evaluate against
    real human judgement, put `id,label` rows in
    data/golden_levels_human.csv; they override the constructed labels and
    the report says labels=human.

Failure modes & 1-step mitigation
  * LLM returns junk grids -> parse_grid + solver proof reject them; the
    heuristic generator fills the quota (never fewer candidates).
  * Critic too lenient or too strict -> golden kappa and per-defect recall
    are printed on every verify run; a drop is visible immediately.
  * Shipped level plays harder than predicted -> monitor_and_rollback()
    retires it (new daily level) or rolls back to the previous version.
  * Re-running the daily ship on the same date -> idempotent (returns the
    levels already shipped for that date).
================================================================================
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pydantic import Field

from agents.common import ApprovalLedger, audit
from agents.level_factory.difficulty_model import DifficultyModel, LevelFeatures, calibrated_model
from agents.level_factory.solver import LevelSolver, SolverProof
from core.config import get_settings
from core.db import TelemetryDB
from core.harness.llm import LLMClient, Message, MockLLM, get_llm, is_real_llm
from core.harness.redteam_guard import READ_WRITE, AgentPolicy, ApprovalRequest, DefenseConfig
from core.harness.resilience import BudgetExceeded, BudgetGuard
from core.harness.runner import AgentRunner, AgentSpec, guarded_completion
from core.harness.telemetry import Tracer, default_tracer
from core.harness.tools import Privilege, StrictArgs, ToolContext, ToolRegistry, ToolSpec
from core.levels import build_seed_levels, next_level_id, publish_version, set_version_status
from core.wordgrid import Trie, grid_to_str, neighbours, parse_grid, random_grid, solve
from core.wordlist import Lexicon, load_lexicon
from scripts.build_wordlist import BLOCKLIST

# Words a player must never be able to TRACE on a board (profanity / sexual).
# Deliberately narrower than build_wordlist.BLOCKLIST, which also excludes
# harmless non-dictionary tokens ("ins", "hey", "etc") from the answer list.
OFFENSIVE = frozenset({
    "ass", "asses", "asshole", "damn", "damned", "hell", "sex", "sexy", "porn", "nude", "naked", "rape", "raped",
    "slut", "whore", "bitch", "crap", "piss", "pissed", "fuck", "fucked", "fucking", "shit", "dick", "cock", "tit",
    "tits", "nazi", "boob", "boobs", "horny", "pimp", "kinky", "lust", "suck", "sucks", "poop",
}) & BLOCKLIST

RUBRICS = ("solvability_fairness", "difficulty_fit", "lexical_quality", "board_quality")
RARE_LETTERS = set("QZXJ")
VOWELS = set("AEIOU")
DAILY_SLOTS = [("daily-easy", 0.20, 4, 110), ("daily-medium", 0.32, 4, 100), ("daily-hard", 0.42, 5, 110)]
GOLDEN_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_levels.jsonl"


@dataclass
class Candidate:
    grid: str
    time_limit_s: int
    target_words: int
    goal: float
    slot: str = ""
    source: str = "heuristic"
    required_words: list[str] = field(default_factory=list)
    predicted_fail_rate: float | None = None


# ============================================================================ generator
class LevelGenerator:
    """LLM proposes raw grids when available; the heuristic always backs it up."""

    PROMPT = ("Propose {n} {size}x{size} word-search letter grids for a casual English word game. Mix common "
              "letters with enough vowels, avoid rare letters (Q, Z, X, J). Reply ONLY with JSON: "
              '{{"grids": ["ABCD/EFGH/IJKL/MNOP", ...]}} using "/" between rows.')

    def __init__(self, solver: LevelSolver, model: DifficultyModel, llm: LLMClient | None = None, seed: int = 0,
                 llm_budget_usd: float = 0.05, tracer: Tracer | None = None):
        self.solver, self.model, self.llm, self.tracer = solver, model, llm, tracer
        self.rng = random.Random(seed)
        self.budget = BudgetGuard(max_usd=llm_budget_usd, max_tokens=50_000, max_steps=1_000)
        self.stats = {"llm_proposed": 0, "llm_valid": 0, "heuristic": 0, "llm_calls": 0, "llm_errors": 0,
                      "llm_budget_stops": 0, "last_llm_error": ""}

    @property
    def uses_llm(self) -> bool:
        return is_real_llm(self.llm)

    def _llm_grids(self, size: int, n: int) -> list[str]:
        prompt = self.PROMPT.format(n=n, size=size)
        try:  # through the harness: budget pre-check, breaker, retry, llm.call span
            resp = guarded_completion(self.llm, [Message("system", "You design word puzzles."), Message("user", prompt)],
                                      budget=self.budget, tracer=self.tracer, max_output_tokens=600,
                                      purpose="level_generator")
            self.stats["llm_calls"] += 1
            m = re.search(r"\{.*\}", resp.text, re.S)
            grids = json.loads(m.group())["grids"] if m else []
            if not isinstance(grids, list):
                grids = []
        except BudgetExceeded as e:  # spend cap reached -> heuristic fills the quota
            self.stats["llm_budget_stops"] += 1
            self.stats["last_llm_error"] = str(e)[:160]
            return []
        except Exception as e:  # provider / parse failure -> heuristic fills the quota (recorded, never silent)
            self.stats["llm_errors"] += 1
            self.stats["last_llm_error"] = f"{type(e).__name__}: {str(e)[:140]}"
            return []
        self.stats["llm_proposed"] += len(grids)
        valid = []
        for g in grids:
            try:
                rows = parse_grid(str(g))
            except ValueError:
                continue
            if len(rows) == size and len(rows[0]) == size:
                valid.append(grid_to_str(rows))
        self.stats["llm_valid"] += len(valid)
        return valid

    def candidate_for(self, grid: str, goal: float, time_limit_s: int, slot: str, source: str) -> Candidate | None:
        proof = self.solver.prove(grid)
        if not proof.ok or proof.word_count < 6:
            return None
        f = LevelFeatures.from_words(proof.valid_words, self.solver.lex, 3, time_limit_s, proof.branching_factor)
        target, pred = self.model.best_target(f, goal, lo=3, hi=max(3, int(proof.word_count * 0.6)))
        return Candidate(proof.grid, time_limit_s, target, goal, slot, source, predicted_fail_rate=round(pred, 4))

    def propose(self, goal: float, size: int, time_limit_s: int, n: int, slot: str = "") -> list[Candidate]:
        out: list[Candidate] = []
        if self.uses_llm:
            for g in self._llm_grids(size, n):
                c = self.candidate_for(g, goal, time_limit_s, slot, "llm")
                if c:
                    out.append(c)
        attempts = 0
        while len(out) < n and attempts < n * 20:
            attempts += 1
            g = grid_to_str(random_grid(self.rng, size, size))
            c = self.candidate_for(g, goal, time_limit_s, slot, "heuristic")
            if c:
                out.append(c)
                self.stats["heuristic"] += 1
        return out[:n]


# ============================================================================ critic
@dataclass
class RubricScore:
    rubric: str
    score: int
    reason: str


@dataclass
class CriticReport:
    grid: str
    approve: bool
    scores: list[RubricScore]
    predicted_fail_rate: float | None
    llm_review: dict | None = None

    @property
    def score_map(self) -> dict[str, int]:
        return {s.rubric: s.score for s in self.scores}

    def to_dict(self) -> dict:
        return {"grid": self.grid, "approve": self.approve, "predicted_fail_rate": self.predicted_fail_rate,
                "scores": self.score_map, "reasons": {s.rubric: s.reason for s in self.scores},
                "llm_review": self.llm_review}


def blocklist_trie() -> Trie:
    return Trie(w for w in OFFENSIVE if len(w) >= 3)


class CriticAgent:
    def __init__(self, solver: LevelSolver, model: DifficultyModel, existing_word_sets: Sequence[set[str]] = (),
                 llm: LLMClient | None = None, budget: BudgetGuard | None = None, tracer: Tracer | None = None):
        self.solver, self.model, self.llm, self.tracer = solver, model, llm, tracer
        self.budget = budget or BudgetGuard(max_usd=0.05, max_tokens=50_000, max_steps=1_000)
        self.llm_stats = {"reviews": 0, "errors": 0, "budget_stops": 0, "last_error": ""}
        self.existing = [s for s in existing_word_sets if s]
        self.block_trie = blocklist_trie()

    def review(self, c: Candidate, proof: SolverProof | None = None) -> CriticReport:
        proof = proof or self.solver.prove(c.grid, c.required_words, c.target_words)
        if proof.reason.startswith("malformed"):
            s = [RubricScore(r, 1, proof.reason) for r in RUBRICS]
            return CriticReport(c.grid, False, s, None)
        g = parse_grid(proof.grid)
        letters = [ch for row in g for ch in row]
        n_cells = len(letters)
        scores: list[RubricScore] = []

        # 1. solvability & fairness
        ratio = c.target_words / proof.word_count if proof.word_count else math.inf
        if proof.targets_missing:
            scores.append(RubricScore(RUBRICS[0], 1, f"required words not on board: {proof.targets_missing}"))
        elif not proof.witness_rechecked or c.target_words > proof.word_count:
            scores.append(RubricScore(RUBRICS[0], 1, f"unsolvable: target {c.target_words} > {proof.word_count} words"))
        elif ratio > 0.75:
            scores.append(RubricScore(RUBRICS[0], 2, f"needs {ratio:.0%} of all words - no slack for players"))
        elif ratio > 0.6:
            scores.append(RubricScore(RUBRICS[0], 3, f"tight: needs {ratio:.0%} of all words"))
        else:
            scores.append(RubricScore(RUBRICS[0], 5, f"solvable with slack ({c.target_words}/{proof.word_count})"))

        # 2. difficulty fit
        pred = None
        if proof.word_count and c.target_words <= proof.word_count:
            pred = self.model.predict(LevelFeatures.from_words(proof.valid_words, self.solver.lex, c.target_words,
                                                               c.time_limit_s, proof.branching_factor))
            gap = abs(pred - c.goal)
            per_word = c.time_limit_s / max(1, c.target_words)
            if not 0.05 <= pred <= 0.70:
                sc, why = 1, f"predicted fail {pred:.2f} outside playable band [0.05, 0.70]"
            elif per_word > 60:
                sc, why = 2, f"no time pressure: {per_word:.0f}s per required word"
            else:
                sc = 5 if gap <= 0.08 else 4 if gap <= 0.12 else 3 if gap <= 0.17 else 2 if gap <= 0.25 else 1
                why = f"predicted fail {pred:.2f} vs goal {c.goal:.2f}"
            scores.append(RubricScore(RUBRICS[1], sc, why))
        else:
            scores.append(RubricScore(RUBRICS[1], 1, "no prediction possible (unsolvable)"))

        # 3. lexical quality
        offensive = sorted(solve(g, self.block_trie).words)
        easy = sorted((self.solver.lex.zipf[w] for w in proof.valid_words), reverse=True)[: max(1, c.target_words)]
        easy_z = sum(easy) / len(easy) if easy else 0
        short_share = proof.by_length.get(3, 0) / proof.word_count if proof.word_count else 1
        if offensive:
            scores.append(RubricScore(RUBRICS[2], 1, f"blocklisted words formable on board: {offensive}"))
        elif easy_z < 4.6:
            scores.append(RubricScore(RUBRICS[2], 2, f"required vocabulary too obscure (mean zipf {easy_z:.2f})"))
        elif short_share > 0.7:
            scores.append(RubricScore(RUBRICS[2], 2, f"{short_share:.0%} of words are 3 letters"))
        else:
            scores.append(RubricScore(RUBRICS[2], 5 if easy_z >= 5.0 else 4, f"clean vocabulary (mean zipf {easy_z:.2f})"))

        # 4. board quality
        counts = {ch: letters.count(ch) for ch in set(letters)}
        top_letter, top = max(counts.items(), key=lambda kv: kv[1])
        rare = sum(counts.get(ch, 0) for ch in RARE_LETTERS)
        vowel_share = sum(counts.get(ch, 0) for ch in VOWELS) / n_cells
        words = set(proof.valid_words)
        jac = max((len(words & e) / len(words | e) for e in self.existing), default=0.0) if words else 0.0
        limit = 4 if n_cells <= 16 else 5
        if jac >= 0.6:
            sc, why = 1, f"near-duplicate of a live level (word-set Jaccard {jac:.2f})"
        elif top > limit:
            sc, why = 2, f"letter spam: {top}x '{top_letter}'"
        elif rare >= 3:
            sc, why = 2, f"{rare} rare letters (Q/Z/X/J)"
        elif not 0.2 <= vowel_share <= 0.6:
            sc, why = 2, f"vowel share {vowel_share:.0%} outside 20-60%"
        else:
            sc = 5 if rare == 0 and jac < 0.3 else 4 if rare <= 1 else 3
            why = f"vowels {vowel_share:.0%}, max repeat {top}, rare letters {rare}, novelty J={jac:.2f}"
        scores.append(RubricScore(RUBRICS[3], sc, why))

        llm_review = self._llm_review(c, proof) if is_real_llm(self.llm) else None
        if llm_review:  # risk-averse ensemble: an LLM can only LOWER a rubric score (clamped to the 1-5 scale)
            for s in scores:
                v = llm_review.get(s.rubric)
                if isinstance(v, int) and not isinstance(v, bool) and max(1, v) < s.score:
                    s.score, s.reason = max(1, v), s.reason + f" | llm: {str(llm_review.get('reason', ''))[:80]}"
        return CriticReport(proof.grid, all(s.score >= 3 for s in scores), scores, round(pred, 4) if pred is not None else None,
                            llm_review)

    def _llm_review(self, c: Candidate, proof: SolverProof) -> dict | None:
        prompt = (f"Grade this word-game level 1-5 on {list(RUBRICS)}. Grid {proof.grid}, target {c.target_words} of "
                  f"{proof.word_count} words in {c.time_limit_s}s, words: {proof.valid_words[:40]}. Reply ONLY JSON "
                  '{"solvability_fairness":n,"difficulty_fit":n,"lexical_quality":n,"board_quality":n,"reason":"..."}')
        try:  # through the harness: budget pre-check, breaker, retry, llm.call span
            resp = guarded_completion(self.llm, [Message("user", prompt)], budget=self.budget, tracer=self.tracer,
                                      max_output_tokens=300, purpose="level_critic")
            m = re.search(r"\{.*\}", resp.text, re.S)
            out = json.loads(m.group()) if m else None
            if not isinstance(out, dict):
                return None  # e.g. the breaker served the mock, which has no opinion
            self.llm_stats["reviews"] += 1
            return out
        except BudgetExceeded as e:  # cap reached: the deterministic rubrics stand on their own
            self.llm_stats["budget_stops"] += 1
            self.llm_stats["last_error"] = str(e)[:160]
            return None
        except Exception as e:
            self.llm_stats["errors"] += 1
            self.llm_stats["last_error"] = f"{type(e).__name__}: {str(e)[:140]}"
            return None


# ============================================================================ golden dataset
GOLDEN_PLAN = [("clean", 54), ("unsolvable_target", 5), ("missing_required_word", 3), ("too_hard", 6),
               ("too_easy", 6), ("offensive_word", 6), ("duplicate_board", 6), ("letter_spam", 6),
               ("rare_letter_overload", 4), ("malformed_grid", 4)]
EMBED_WORDS = ["damn", "crap", "piss", "hell"]


def _embed_word(rng: random.Random, size: int, word: str) -> str:
    while True:
        g = random_grid(rng, size, size)
        r, c = rng.randrange(size), rng.randrange(size)
        path = [(r, c)]
        for _ in word[1:]:
            opts = [p for p in neighbours(path[-1][0], path[-1][1], size, size) if p not in path]
            if not opts:
                break
            path.append(rng.choice(opts))
        if len(path) == len(word):
            for (pr, pc), ch in zip(path, word.upper()):
                g[pr][pc] = ch
            return grid_to_str(g)


def build_golden_dataset(solver: LevelSolver, model: DifficultyModel, seed: int = 2026,
                         seed_levels: list[dict] | None = None) -> list[dict]:
    rng = random.Random(seed)
    gen = LevelGenerator(solver, model, seed=seed)
    block = blocklist_trie()
    seeds = seed_levels or build_seed_levels(solver.lex, solver.trie)
    items: list[dict] = []

    def meets_board_spec(grid: str) -> bool:
        """Hard design-spec constraints a 'clean' level must satisfy (soft judgements are left to the critic)."""
        letters = grid.replace("/", "")
        limit = 4 if len(letters) <= 16 else 5
        vowels = sum(ch in VOWELS for ch in letters) / len(letters)
        return (max(letters.count(ch) for ch in set(letters)) <= limit and 0.2 <= vowels <= 0.6
                and sum(ch in RARE_LETTERS for ch in letters) <= 1)

    def clean(size: int | None = None) -> Candidate:
        while True:
            size_ = size or (4 if rng.random() < 0.7 else 5)
            c = gen.propose(round(rng.uniform(0.18, 0.48), 2), size_, rng.choice([90, 100, 110, 120]), 1)[0]
            if meets_board_spec(c.grid) and not solve(parse_grid(c.grid), block).words:
                return c

    for defect, count in GOLDEN_PLAN:
        for _ in range(count):
            note = ""
            if defect == "clean":
                c = clean()
            elif defect == "unsolvable_target":
                c = clean()
                c.target_words = solver.prove(c.grid).word_count + rng.randint(1, 4)
            elif defect == "missing_required_word":
                c = clean()
                words = solver.prove(c.grid).valid_words
                absent = next(w for w in ("zone", "jazz", "quiz", "wolf", "zero", "joke", "fox") if w not in words)
                c.required_words = rng.sample(words, 2) + [absent]
                note = f"requires '{absent}'"
            elif defect == "too_hard":
                c = clean()
                c.target_words, c.time_limit_s = max(3, math.ceil(0.95 * solver.prove(c.grid).word_count)), 30
            elif defect == "too_easy":
                c = clean()
                c.target_words, c.time_limit_s = 2, 180
            elif defect == "offensive_word":
                w = rng.choice(EMBED_WORDS)
                size = rng.choice([4, 5])
                c = gen.candidate_for(_embed_word(rng, size, w), 0.3, 100, "", "golden") or clean(size)
                note = f"embeds '{w}'"
            elif defect == "duplicate_board":
                src = rng.choice(seeds)
                g = parse_grid(src["grid"])
                if rng.random() < 0.5:
                    g = [list(col) for col in zip(*g)]  # transpose: same adjacency, same word set
                c = Candidate(grid_to_str(g), src["time_limit_s"], src["target_words"], 0.3, source="golden")
                note = f"copy of seed level {src['level_id']}"
            elif defect == "letter_spam":
                c = clean()
                g = parse_grid(c.grid)
                n = len(g)
                letter = rng.choice("ESTA")
                for idx in rng.sample(range(n * n), 7 if n == 4 else 9):
                    g[idx // n][idx % n] = letter
                c.grid = grid_to_str(g)
                note = f"spammed '{letter}'"
            elif defect == "rare_letter_overload":
                c = clean()
                g = parse_grid(c.grid)
                n = len(g)
                for idx, ch in zip(rng.sample(range(n * n), 4), "QZXJ"):
                    g[idx // n][idx % n] = ch
                c.grid = grid_to_str(g)
            else:  # malformed_grid
                c = clean(4)
                rows = c.grid.split("/")
                c.grid = "/".join(rows[:2] + [rows[2][:3]] + rows[3:]) if rng.random() < 0.5 else c.grid[:-1] + "7"
            items.append({"id": f"G{len(items) + 1:03d}", "grid": c.grid, "target_words": c.target_words,
                          "time_limit_s": c.time_limit_s, "goal": c.goal, "required_words": c.required_words,
                          "defect": defect, "label": "accept" if defect == "clean" else "reject",
                          "label_source": "construction", "note": note})
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["id"] = f"G{i:03d}"
    return items


def save_golden(items: list[dict], path: Path = GOLDEN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(it) + "\n" for it in items))
    return path


def load_golden(path: Path = GOLDEN_PATH, human_csv: Path | None = None) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    human_csv = human_csv or path.with_name("golden_levels_human.csv")
    if human_csv.exists():
        with human_csv.open() as fh:
            labels = {r["id"]: r["label"].strip().lower() for r in csv.DictReader(fh)}
        for it in items:
            if it["id"] in labels:
                it["label"], it["label_source"] = labels[it["id"]], "human"
    return items


def cohens_kappa(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("cohens_kappa needs two equal-length, non-empty label sequences")
    n = len(y_true)
    labels = sorted(set(y_true) | set(y_pred))
    po = sum(a == b for a, b in zip(y_true, y_pred)) / n
    pe = sum((sum(a == k for a in y_true) / n) * (sum(b == k for b in y_pred) / n) for k in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def evaluate_critic(critic: CriticAgent, items: list[dict]) -> dict:
    y_true, y_pred, rows = [], [], []
    for it in items:
        c = Candidate(it["grid"], it["time_limit_s"], it["target_words"], it["goal"], required_words=it.get("required_words", []))
        rep = critic.review(c)
        pred = "accept" if rep.approve else "reject"
        y_true.append(it["label"]); y_pred.append(pred)
        rows.append({**it, "critic": pred, "scores": rep.score_map,
                     "failed": [s.rubric + ": " + s.reason for s in rep.scores if s.score < 3]})
    tp = sum(t == "reject" and p == "reject" for t, p in zip(y_true, y_pred))
    tn = sum(t == "accept" and p == "accept" for t, p in zip(y_true, y_pred))
    fp = sum(t == "accept" and p == "reject" for t, p in zip(y_true, y_pred))
    fn = sum(t == "reject" and p == "accept" for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    per_defect: dict[str, dict] = {}
    for r in rows:
        d = per_defect.setdefault(r["defect"], {"n": 0, "agree": 0})
        d["n"] += 1
        d["agree"] += r["critic"] == r["label"]
    return {
        "n": len(items), "label_source": sorted({it["label_source"] for it in items}),
        "agreement_pct": round(100 * (tp + tn) / len(items), 1),
        "precision_reject": round(prec, 4), "recall_reject": round(rec, 4),
        "f1_reject": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "precision_accept": round(tn / (tn + fn), 4) if tn + fn else 0.0,
        "recall_accept": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "cohens_kappa": round(cohens_kappa(y_true, y_pred), 4),
        "confusion": {"tp_reject": tp, "tn_accept": tn, "fp_wrongly_rejected": fp, "fn_missed_defect": fn},
        "per_defect": per_defect, "disagreements": [r for r in rows if r["critic"] != r["label"]],
    }


# ============================================================================ harness-wired factory
class GoalArgs(StrictArgs):
    goal: float = Field(ge=0.02, le=0.8)
    size: int = Field(ge=4, le=6)
    time_limit_s: int = Field(ge=30, le=240)
    n: int = Field(ge=1, le=200)
    slot: str = Field(default="", max_length=40)


class LevelArgs(StrictArgs):
    grid: str = Field(max_length=64)
    target_words: int = Field(ge=1, le=60)
    time_limit_s: int = Field(ge=10, le=300)
    goal: float = Field(default=0.3, ge=0.0, le=1.0)
    required_words: list[str] = Field(default_factory=list, max_length=20)


class PublishArgs(StrictArgs):
    level_id: int = Field(ge=0, le=100_000, description="0 = allocate a new level id")
    grid: str = Field(max_length=64)
    target_words: int = Field(ge=1, le=60)
    time_limit_s: int = Field(ge=10, le=300)
    status: str = Field(pattern="^(live|candidate)$")
    notes: str = Field(default="", max_length=200)


class HealthArgs(StrictArgs):
    min_attempts: int = Field(default=40, ge=1)
    margin: float = Field(default=0.20, ge=0.0, le=1.0)


class RetireArgs(StrictArgs):
    level_id: int
    version: int
    reason: str = Field(max_length=200)


FACTORY_TOOLS = frozenset({"generate_candidates", "prove_level", "predict_difficulty", "critique_level",
                           "publish_level", "live_level_health", "retire_level"})


class LevelFactory:
    SYSTEM = ("You are the Autonomous Level Factory. Generate word-grid levels, prove them with the solver, "
              "predict difficulty, critique them, and publish only levels that pass every gate.")

    def __init__(self, db: TelemetryDB, *, lex: Lexicon | None = None, model: DifficultyModel | None = None,
                 llm: LLMClient | None = None, tracer: Tracer | None = None, defenses: DefenseConfig = DefenseConfig(),
                 seed: int = 0):
        self.db = db
        self.lex = lex or load_lexicon()
        self.solver = LevelSolver(self.lex)
        if model is None:
            with db.cursor() as con:
                model, self.calibration_points = calibrated_model(con, self.lex)
        else:
            self.calibration_points = []
        self.model = model
        # No explicit client -> provider selection by environment: ANTHROPIC_API_KEY / OPENAI_API_KEY give a real
        # model behind retries + a circuit breaker; no key (or LLM_PROVIDER=mock) gives the deterministic mock.
        self.llm = llm = llm if llm is not None else get_llm()
        self.tracer = tracer or default_tracer()
        self.generator = LevelGenerator(self.solver, self.model, llm, seed=seed, tracer=self.tracer)
        self.llm_budget = self._fresh_llm_budget()
        self.generator.budget = self.llm_budget
        self.critic_llm_stats = {"reviews": 0, "errors": 0, "budget_stops": 0, "last_error": ""}
        self.ledger: dict[str, dict] = {}  # grid -> gate results from THIS process
        self.approvals = ApprovalLedger(db, "level_factory", self._release_gate, "release_gate")
        self.runner = AgentRunner(
            AgentSpec("level_factory", self.SYSTEM, AgentPolicy(FACTORY_TOOLS, get_settings().data_dir)),
            self._registry(), llm, defenses=defenses, tracer=self.tracer, approver=self.approvals)
        self.funnel = {"generated": 0, "solver_verified": 0, "in_difficulty_band": 0, "critic_approved": 0, "shipped": 0}

    @staticmethod
    def _fresh_llm_budget() -> BudgetGuard:
        cfg = get_settings()  # HARNESS_MAX_RUN_USD / HARNESS_MAX_RUN_TOKENS cap the model spend of ONE pipeline run
        return BudgetGuard(max_usd=cfg.max_run_usd, max_tokens=cfg.max_run_tokens, max_steps=10_000)

    # ------------------------------------------------------------------ gates
    def _release_gate(self, req: ApprovalRequest) -> tuple[bool, str]:
        if req.tool != "publish_level":
            return False, "release gate only approves publish_level"
        rec = self.ledger.get(req.args.get("grid", "").upper())
        if not rec:
            return False, "grid never passed through prove/critique in this run"
        if not rec["proof_ok"]:
            return False, f"solver proof failed: {rec['proof_reason']}"
        if not rec["critic_approve"]:
            return False, "critic rejected the level"
        if rec["target_words"] != req.args.get("target_words") or rec["time_limit_s"] != req.args.get("time_limit_s"):
            return False, "published config differs from the critiqued config"
        return True, f"proof ok, critic {rec['scores']}, predicted fail {rec['predicted']}"

    def _existing_word_sets(self) -> list[set[str]]:
        with self.db.cursor() as con:
            rows = con.execute("SELECT valid_words FROM levels WHERE status IN ('live', 'candidate')").fetchall()
        return [set(json.loads(r[0])) for r in rows]

    # ------------------------------------------------------------------ tools
    def _t_generate(self, a: GoalArgs, ctx: ToolContext) -> list[dict]:
        if is_real_llm(self.llm):  # generation starts a pipeline run: generator + critic share one fresh spend cap
            self.llm_budget = self.generator.budget = self._fresh_llm_budget()
        cands = self.generator.propose(a.goal, a.size, a.time_limit_s, a.n, a.slot)
        self.funnel["generated"] += len(cands)
        return [asdict(c) for c in cands]

    def _t_prove(self, a: LevelArgs, ctx: ToolContext) -> dict:
        p = self.solver.prove(a.grid, a.required_words, a.target_words)
        return {**p.summary(), "valid_words": p.valid_words}

    def _t_predict(self, a: LevelArgs, ctx: ToolContext) -> dict:
        p = self.solver.prove(a.grid)
        if not p.ok:
            return {"predicted_fail_rate": None, "reason": p.reason}
        f = LevelFeatures.from_words(p.valid_words, self.lex, a.target_words, a.time_limit_s, p.branching_factor)
        return {"predicted_fail_rate": round(self.model.predict(f), 4), "features": f.describe()}

    def _t_critique(self, a: LevelArgs, ctx: ToolContext) -> dict:
        critic = CriticAgent(self.solver, self.model, self._existing_word_sets(), self.llm, budget=self.llm_budget,
                             tracer=self.tracer)
        c = Candidate(a.grid, a.time_limit_s, a.target_words, a.goal, required_words=a.required_words)
        proof = self.solver.prove(a.grid, a.required_words, a.target_words)
        rep = critic.review(c, proof)
        for k in ("reviews", "errors", "budget_stops"):
            self.critic_llm_stats[k] += critic.llm_stats[k]
        self.critic_llm_stats["last_error"] = critic.llm_stats["last_error"] or self.critic_llm_stats["last_error"]
        self.ledger[proof.grid] = {"proof_ok": proof.ok, "proof_reason": proof.reason, "critic_approve": rep.approve,
                                   "scores": rep.score_map, "predicted": rep.predicted_fail_rate,
                                   "target_words": a.target_words, "time_limit_s": a.time_limit_s}
        return rep.to_dict()

    def _t_publish(self, a: PublishArgs, ctx: ToolContext) -> dict:
        rec = self.ledger[a.grid.upper()]
        with self.db.cursor() as con:
            lid = a.level_id or next_level_id(con)
            row = publish_version(con, self.lex, self.solver.trie, lid, a.grid, a.target_words, a.time_limit_s,
                                  created_by="level_factory", notes=a.notes, status=a.status,
                                  predicted_fail_rate=rec["predicted"])
        audit(self.db, "level_factory", "publish_level", level_id=lid, version=row["version"], status=a.status,
              grid=a.grid, predicted=rec["predicted"])
        return {"level_id": lid, "version": row["version"], "status": a.status, "word_count": row["word_count"],
                "predicted_fail_rate": rec["predicted"]}

    def _t_health(self, a: HealthArgs, ctx: ToolContext) -> list[dict]:
        with self.db.cursor() as con:
            rows = con.execute("""
                SELECT l.level_id, l.version, l.predicted_fail_rate, l.notes, count(e.attempt_id) AS attempts,
                       avg((e.event_name = 'level_fail')::INT) AS observed,
                       (SELECT count(*) FROM levels p WHERE p.level_id = l.level_id AND p.status = 'live'
                                                     AND p.version < l.version) AS earlier_live
                FROM levels l LEFT JOIN events e
                  ON e.level_id = l.level_id AND e.level_version = l.version
                 AND e.event_name IN ('level_fail', 'level_complete')
                WHERE l.created_by = 'level_factory' AND l.status = 'live'
                GROUP BY ALL ORDER BY 1, 2""").fetchall()
        out = []
        for lid, ver, pred, notes, n, obs, earlier in rows:
            spike = n >= a.min_attempts and pred is not None and obs is not None and obs > pred + a.margin
            out.append({"level_id": lid, "version": ver, "predicted": pred, "observed": round(obs, 4) if obs is not None else None,
                        "attempts": n, "spike": spike, "action": ("rollback" if earlier else "retire") if spike else "none"})
        return out

    def _t_retire(self, a: RetireArgs, ctx: ToolContext) -> dict:
        with self.db.cursor() as con:
            earlier = con.execute("SELECT count(*) FROM levels WHERE level_id=? AND status='live' AND version<?",
                                  [a.level_id, a.version]).fetchone()[0]
            status = "rolled_back" if earlier else "retired"
            set_version_status(con, a.level_id, a.version, status, a.reason)
        audit(self.db, "level_factory", status, level_id=a.level_id, version=a.version, reason=a.reason)
        return {"level_id": a.level_id, "version": a.version, "status": status}

    def _registry(self) -> ToolRegistry:
        R, W = Privilege.READ, Privilege.WRITE
        slow = is_real_llm(self.llm)  # a real model round-trip (with retries) needs far more than the 5 s default
        return ToolRegistry([
            ToolSpec("generate_candidates", "Propose candidate grids for a difficulty goal", GoalArgs, self._t_generate, R,
                     timeout_s=150 if slow else 60),
            ToolSpec("prove_level", "Solver proof of solvability", LevelArgs, self._t_prove, R),
            ToolSpec("predict_difficulty", "Predict per-attempt fail rate", LevelArgs, self._t_predict, R),
            ToolSpec("critique_level", "4-rubric critic", LevelArgs, self._t_critique, R, timeout_s=90 if slow else 5),
            ToolSpec("publish_level", "Publish a level version (release-gated)", PublishArgs, self._t_publish, W,
                     idempotent=False, requires_approval=True),
            ToolSpec("live_level_health", "Observed vs predicted fail rate of shipped levels", HealthArgs, self._t_health, R),
            ToolSpec("retire_level", "Retire or roll back a shipped level version", RetireArgs, self._t_retire, W, idempotent=True),
        ])

    # ------------------------------------------------------------------ pipelines
    def build_level(self, wf, goal: float, size: int, time_limit_s: int, n: int, slot: str) -> dict | None:
        """generate -> prove -> difficulty band -> critique; best approved candidate or None."""
        best = None
        for c in wf.call("generate_candidates", goal=goal, size=size, time_limit_s=time_limit_s, n=n, slot=slot):
            proof = wf.call("prove_level", grid=c["grid"], target_words=c["target_words"], time_limit_s=time_limit_s)
            if not proof["ok"]:
                continue
            self.funnel["solver_verified"] += 1
            if c["predicted_fail_rate"] is None or abs(c["predicted_fail_rate"] - goal) > 0.12:
                continue
            self.funnel["in_difficulty_band"] += 1
            rep = wf.call("critique_level", grid=c["grid"], target_words=c["target_words"], time_limit_s=time_limit_s, goal=goal)
            if not rep["approve"]:
                continue
            self.funnel["critic_approved"] += 1
            key = (sum(rep["scores"].values()), -abs(rep["predicted_fail_rate"] - goal))
            if best is None or key > best["key"]:
                best = {"key": key, "candidate": c, "critic": rep, "proof": proof}
        return best

    def ship_daily(self, date: str, slots=DAILY_SLOTS, candidates_per_slot: int = 30) -> list[dict]:
        tag = f"daily:{date}"
        with self.db.cursor() as con:
            done = con.execute("SELECT level_id, version, grid, target_words, predicted_fail_rate, notes FROM levels "
                               "WHERE created_by='level_factory' AND notes LIKE ? ORDER BY level_id", [tag + "%"]).fetchall()
        if done:
            return [{"level_id": r[0], "version": r[1], "grid": r[2], "target_words": r[3], "predicted_fail_rate": r[4],
                     "notes": r[5], "already_shipped": True} for r in done]
        shipped = []
        with self.runner.workflow(f"ship_daily:{date}", mode=READ_WRITE) as wf:
            for slot, goal, size, tl in slots:
                best = self.build_level(wf, goal, size, tl, candidates_per_slot, slot)
                if best is None:
                    audit(self.db, "level_factory", "slot_unfilled", slot=slot, goal=goal)
                    continue
                c = best["candidate"]
                pub = wf.call("publish_level", level_id=0, grid=c["grid"], target_words=c["target_words"],
                              time_limit_s=tl, status="live", notes=f"{tag} slot={slot}")
                self.funnel["shipped"] += 1
                shipped.append({**pub, "slot": slot, "goal": goal, "grid": c["grid"], "target_words": c["target_words"],
                                "time_limit_s": tl, "scores": best["critic"]["scores"], "source": c["source"],
                                "word_count": best["proof"]["word_count"]})
        return shipped

    def propose_variant(self, level_id: int, goal: float, size: int = 4, time_limit_s: int = 90,
                        candidates: int = 40, note: str = "") -> dict:
        """Used by the LiveOps Copilot: build an easier/harder version and store it as a CANDIDATE."""
        with self.runner.workflow(f"propose_variant:{level_id}", mode=READ_WRITE) as wf:
            best = self.build_level(wf, goal, size, time_limit_s, candidates, f"variant-L{level_id}")
            if best is None:
                raise RuntimeError(f"no candidate passed every gate for goal {goal}")
            c = best["candidate"]
            pub = wf.call("publish_level", level_id=level_id, grid=c["grid"], target_words=c["target_words"],
                          time_limit_s=time_limit_s, status="candidate", notes=note or f"variant goal={goal}")
        return {**pub, "grid": c["grid"], "target_words": c["target_words"], "time_limit_s": time_limit_s,
                "critic": best["critic"]["scores"], "word_count": best["proof"]["word_count"]}

    def monitor_and_rollback(self, min_attempts: int = 40, margin: float = 0.20) -> list[dict]:
        actions = []
        with self.runner.workflow("live_monitor", mode=READ_WRITE) as wf:
            for h in wf.call("live_level_health", min_attempts=min_attempts, margin=margin):
                if h["spike"]:
                    res = wf.call("retire_level", level_id=h["level_id"], version=h["version"],
                                  reason=f"live fail {h['observed']:.2f} > predicted {h['predicted']:.2f} + {margin}")
                    actions.append({**h, **res})
                else:
                    actions.append(h)
        return actions
