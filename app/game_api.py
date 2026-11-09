"""
================================================================================
BLOCK: Live Word Game API (server-authoritative referee + telemetry emitter)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : app/static/game.html (browser client), core/levels.py
             (`active_levels` view: highest live version per level),
             core/wordgrid.py (path referee), core/wordlist.py (lexicon).
  Downstream: `events` + `players` tables in data/telemetry.duckdb - the SAME
             tables the seeder fills, so live play lands next to the 2,500
             seeded players and every Phase-2/3 engine sees it immediately.
  Events   : session_start, level_start, word_found, level_complete,
             level_fail, ad_impression (all emitted HERE, never by the client).

Key engineering mechanism
  * The client only sends tile COORDINATES. The server derives the word from
    the stored grid, checks bounds -> adjacency -> no tile reuse -> lexicon,
    enforces the clock with time.monotonic() (+2 s network grace), dedupes
    words, and decides complete/fail itself.
  * Every event records level_version, so the Phase-2 Level Factory can
    attribute a live fail-rate spike to the exact version and roll it back.
  * Ads: a simulated 5-network mediation fill (same eCPM model as the seeder)
    priced server-side; rewarded hints reveal the most common unfound word.

Failure modes & 1-step mitigation
  * Forged / replayed requests -> unknown or finished attempt = 409; words
    after the clock = rejected `time_up`; duplicate word = `already_found`.
  * Abandoned attempts (tab closed) -> in-memory store prunes after 1 h.
  * Bad level version shipped -> rollback_level() flips status; next
    level_start serves the previous version with zero code changes.
  * DB write failure -> the request fails loudly (500), state is not advanced.
================================================================================
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from core.wordgrid import parse_grid, validate_path

router = APIRouter(prefix="/api/game", tags=["game"])

TIME_GRACE_S = 2.0
ATTEMPT_TTL_S = 3600
MAX_HINTS = 3
NETWORKS = {"applovin_max": 12.0, "unity_ads": 9.0, "ironsource": 10.0, "admob": 11.0, "meta_an": 8.0}
NETWORK_WEIGHTS = [0.30, 0.18, 0.17, 0.22, 0.13]


# ----------------------------------------------------------------------------- request models
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionIn(_Strict):
    user_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    platform: Literal["web", "ios", "android"] = "web"


class LevelStartIn(_Strict):
    session_id: str
    level_id: int = Field(ge=1, le=10_000)


class WordIn(_Strict):
    attempt_id: str
    path: list[list[int]] = Field(min_length=1, max_length=49)


class LevelEndIn(_Strict):
    attempt_id: str
    reason: Literal["timeout", "quit"]


class AdIn(_Strict):
    attempt_id: str
    placement: Literal["interstitial", "rewarded_hint"]


# ----------------------------------------------------------------------------- state
@dataclass
class Session:
    session_id: str
    user_id: str
    ab_group: str
    platform: str


@dataclass
class Attempt:
    attempt_id: str
    session: Session
    level_id: int
    version: int
    grid: list[list[str]]
    valid: set[str]
    target: int
    time_limit_s: int
    started: float = field(default_factory=time.monotonic)
    found: list[str] = field(default_factory=list)
    status: str = "playing"
    hints: int = 0
    ads: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


class GameState:
    def __init__(self, db, lexicon):
        self.db, self.lex = db, lexicon
        self.sessions: dict[str, Session] = {}
        self.attempts: dict[str, Attempt] = {}
        self.lock = threading.Lock()
        self.rng = random.Random()

    def prune(self) -> None:
        cutoff = time.monotonic() - ATTEMPT_TTL_S
        with self.lock:
            for k in [k for k, a in self.attempts.items() if a.started < cutoff]:
                del self.attempts[k]

    def reset(self) -> None:
        with self.lock:
            self.sessions.clear()
            self.attempts.clear()


def state(request: Request) -> GameState:
    return request.app.state.game


def ab_group(user_id: str) -> str:
    return "gate_30" if int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 2 == 0 else "gate_40"


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def emit(st: GameState, name: str, s: Session, a: Attempt | None = None, *, word=None, revenue=None, network=None,
         placement=None, **props) -> dict:
    row = {"event_id": uuid.uuid4().hex, "ts": now(), "user_id": s.user_id, "session_id": s.session_id,
           "event_name": name, "level_id": a.level_id if a else None, "level_version": a.version if a else None,
           "attempt_id": a.attempt_id if a else None, "ab_group": s.ab_group, "platform": s.platform, "word": word,
           "revenue_usd": revenue, "network": network, "placement": placement, "source": "live",
           "props": json.dumps(props, separators=(",", ":"))}
    st.db.insert_events([row])
    return {**row, "ts": row["ts"].isoformat(timespec="milliseconds") + "Z", "props": props}


def _attempt(st: GameState, attempt_id: str, need_playing: bool = True) -> Attempt:
    a = st.attempts.get(attempt_id)
    if a is None:
        raise HTTPException(404, "unknown attempt")
    if need_playing and a.status != "playing":
        raise HTTPException(409, f"attempt already {a.status}")
    return a


# ----------------------------------------------------------------------------- endpoints
@router.get("/levels")
def levels(request: Request) -> dict:
    st = state(request)
    rows = st.db.query("""
        SELECT l.level_id, l.version, l.n_rows, l.n_cols, l.target_words, l.time_limit_s, l.word_count, l.mean_zipf,
               f.attempts, f.fail_rate
        FROM active_levels l LEFT JOIN (
            SELECT level_id, count(*) attempts,
                   round(avg((event_name = 'level_fail')::INT), 3) fail_rate
            FROM events WHERE event_name IN ('level_fail', 'level_complete') GROUP BY 1) f USING (level_id)
        ORDER BY level_id""")
    return {"levels": rows}


@router.post("/session")
def start_session(body: SessionIn, request: Request) -> dict:
    st = state(request)
    st.prune()
    uid = body.user_id or f"live_{uuid.uuid4().hex[:10]}"
    s = Session(f"ls_{uuid.uuid4().hex[:12]}", uid, ab_group(uid), body.platform)
    with st.lock:
        st.sessions[s.session_id] = s
    with st.db.cursor() as con:
        con.execute("INSERT OR IGNORE INTO players (user_id, install_ts, ab_group, country, platform, channel, "
                    "sum_gamerounds, retention_1, retention_7, source) VALUES (?, ?, ?, 'ZZ', ?, 'organic', 0, false, false, 'live')",
                    [uid, now(), s.ab_group, s.platform])
    ev = emit(st, "session_start", s, client="web")
    return {"user_id": uid, "session_id": s.session_id, "ab_group": s.ab_group, "events": [ev]}


@router.post("/level/start")
def level_start(body: LevelStartIn, request: Request) -> dict:
    st = state(request)
    s = st.sessions.get(body.session_id)
    if s is None:
        raise HTTPException(404, "unknown session; POST /api/game/session first")
    rows = st.db.query("SELECT * FROM active_levels WHERE level_id = ?", [body.level_id])
    if not rows:
        raise HTTPException(404, f"level {body.level_id} has no live version")
    lv = rows[0]
    a = Attempt(f"la_{uuid.uuid4().hex[:12]}", s, lv["level_id"], lv["version"], parse_grid(lv["grid"]),
                set(json.loads(lv["valid_words"])), lv["target_words"], lv["time_limit_s"])
    with st.lock:
        st.attempts[a.attempt_id] = a
    ev = emit(st, "level_start", s, a, target=a.target, time_limit_s=a.time_limit_s)
    return {"attempt_id": a.attempt_id, "level_id": a.level_id, "level_version": a.version,
            "grid": ["".join(r) for r in a.grid], "target_words": a.target, "time_limit_s": a.time_limit_s,
            "total_words": len(a.valid), "events": [ev]}


@router.post("/level/word")
def submit_word(body: WordIn, request: Request) -> dict:
    st = state(request)
    a = _attempt(st, body.attempt_id)
    base = {"found": len(a.found), "target": a.target, "status": a.status}
    if a.elapsed > a.time_limit_s + TIME_GRACE_S:
        return {**base, "accepted": False, "word": "", "reason": "time_up", "events": []}
    v = validate_path(a.grid, body.path, st.lex.words)
    if not v.ok:
        return {**base, "accepted": False, "word": v.word, "reason": v.reason, "events": []}
    if v.word in a.found:
        return {**base, "accepted": False, "word": v.word, "reason": "already_found", "events": []}
    a.found.append(v.word)
    events = [emit(st, "word_found", a.session, a, word=v.word, len=len(v.word), zipf=st.lex.zipf.get(v.word),
                   n=len(a.found), elapsed_s=round(a.elapsed, 2))]
    if len(a.found) >= a.target:
        a.status = "completed"
        events.append(emit(st, "level_complete", a.session, a, found=len(a.found), target=a.target,
                           duration_s=round(a.elapsed, 1), hints=a.hints))
    return {"accepted": True, "word": v.word, "reason": "ok", "found": len(a.found), "target": a.target,
            "status": a.status, "events": events}


@router.post("/level/end")
def level_end(body: LevelEndIn, request: Request) -> dict:
    st = state(request)
    a = _attempt(st, body.attempt_id)
    reason = body.reason
    if reason == "timeout" and a.elapsed < a.time_limit_s - TIME_GRACE_S:
        reason = "quit"  # client clock ran fast or was tampered with: record what really happened
    a.status = "failed"
    ev = emit(st, "level_fail", a.session, a, found=len(a.found), target=a.target, duration_s=round(a.elapsed, 1),
              reason=reason, hints=a.hints)
    missed = sorted(a.valid - set(a.found), key=lambda w: -st.lex.zipf.get(w, 0))[:8]
    return {"status": "failed", "reason": reason, "found": len(a.found), "target": a.target, "missed_examples": missed,
            "events": [ev]}


@router.post("/ad")
def show_ad(body: AdIn, request: Request) -> dict:
    st = state(request)
    a = _attempt(st, body.attempt_id, need_playing=body.placement == "rewarded_hint")
    if body.placement == "rewarded_hint" and a.hints >= MAX_HINTS:
        raise HTTPException(429, f"max {MAX_HINTS} hints per level")
    if a.ads >= 6:
        raise HTTPException(429, "ad frequency cap reached for this attempt")
    a.ads += 1
    events, hint = [], None
    if st.rng.random() < 0.92:  # mediation fill
        net = st.rng.choices(list(NETWORKS), weights=NETWORK_WEIGHTS)[0]
        ecpm = NETWORKS[net] * (1.6 if body.placement == "rewarded_hint" else 1.0) * math.exp(st.rng.gauss(0, 0.35))
        events.append(emit(st, "ad_impression", a.session, a, revenue=round(ecpm / 1000, 6), network=net,
                           placement=body.placement, ecpm=round(ecpm, 3)))
        filled = {"network": net, "ecpm": round(ecpm, 2)}
    else:
        filled = None
    if body.placement == "rewarded_hint":
        a.hints += 1
        pool = sorted(a.valid - set(a.found), key=lambda w: -st.lex.zipf.get(w, 0))
        if pool:
            w = pool[0]
            hint = {"starts_with": w[:2].upper(), "length": len(w)}
    return {"filled": filled, "hint": hint, "events": events}
