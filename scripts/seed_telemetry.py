"""
================================================================================
BLOCK: Telemetry Bootstrapper (Cookie Cats A/B + 2,500-player event cohort)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : data/cookie_cats.csv (Kaggle, optional) OR the calibrated
             marginals below; core/levels.py seed levels; core/wordlist.py.
  Downstream: tables cookie_cats, players, events, levels in
             data/telemetry.duckdb - the SAME tables live gameplay appends to,
             so Phase-2/3 statistical engines work on day 0.

Key engineering mechanism
  1. Cookie Cats (90,189 players, gate_30 vs gate_40). If the real CSV is
     present it is loaded verbatim. Otherwise a calibrated synthetic copy is
     generated whose marginals match the real dataset EXACTLY (integer counts,
     not just in expectation):
        split   gate_30 49.56259% (44,700) / gate_40 50.43741% (45,489)
        D1      gate_30 0.448188 (20,034)  / gate_40 0.442283 (20,119)
        D7      gate_30 0.190201 (8,502)   / gate_40 0.182000 (8,279)
        0 rounds gate_30 1,937             / gate_40 2,057
     A latent engagement score drives a heavy-tailed (log-normal, median ~16)
     sum_gamerounds AND retention, so rounds<->retention are correlated like
     the real data; exact counts come from rank-thresholding that score.
  2. Event cohort: 2,500 players stratified-sampled from cookie_cats by
     (group, D1, D7) cell (largest-remainder allocation) and expanded into
     event streams: install day 0; a session on day 1 iff retention_1; on day
     7 iff retention_7; other days by engagement. Level attempts follow the
     live level catalogue with per-attempt fail rates (level 7 = 0.62 spike).
     Ads (interstitial / rewarded_hint) across a 5-network mediation stack with
     tier x placement eCPMs.
  3. Bulk load via a temp CSV + DuckDB COPY (~0.5M rows in a few seconds).

Failure modes & 1-step mitigation
  * Server holds the DuckDB write lock -> clear DatabaseLockedError message
    + exit code 2 (never a hang, never a partial write).
  * Re-running on a seeded DB -> no-op unless --force (idempotent).
  * Malformed cookie_cats.csv -> column check with an explicit error; falls
    back to calibrated synthetic data with --synthetic.
  * Calibration drift -> verify() recomputes every target from the DB and the
    seeder exits non-zero if any synthetic marginal is off.
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/seed_telemetry.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings
from core.db import EVENT_COLUMNS, DatabaseLockedError, open_rw
from core.levels import build_seed_levels, insert_levels
from core.wordlist import Lexicon, load_lexicon

# ----------------------------------------------------------------------------- targets
COOKIE_CATS_N = 90_189
TARGETS = {
    "gate_30": {"share": 0.4956259, "d1": 0.448188, "d7": 0.190201, "zero_rounds": 1937},
    "gate_40": {"share": 0.5043741, "d1": 0.442283, "d7": 0.182000, "zero_rounds": 2057},
}
FAIL_RATE = {1: .05, 2: .08, 3: .12, 4: .15, 5: .18, 6: .22, 7: .62, 8: .25, 9: .27, 10: .30, 11: .32, 12: .35}
ATTEMPT_CAP = 45
NETWORKS = {"applovin_max": 12.0, "unity_ads": 9.0, "ironsource": 10.0, "admob": 11.0, "meta_an": 8.0}
NETWORK_WEIGHTS = [0.30, 0.18, 0.17, 0.22, 0.13]
COUNTRIES = [("US", .35, 1.0), ("GB", .08, .7), ("DE", .07, .7), ("JP", .05, .7), ("BR", .12, .25), ("IN", .18, .12), ("ZZ", .15, .3)]
CHANNELS = [("organic", .40), ("google_uac", .20), ("meta", .18), ("tiktok", .12), ("applovin", .10)]
COHORT_START = datetime(2026, 10, 1)


def group_counts() -> dict[str, dict[str, int]]:
    n30 = round(COOKIE_CATS_N * TARGETS["gate_30"]["share"])
    out = {}
    for g, n in (("gate_30", n30), ("gate_40", COOKIE_CATS_N - n30)):
        t = TARGETS[g]
        out[g] = {"n": n, "d1": round(n * t["d1"]), "d7": round(n * t["d7"]), "zero": t["zero_rounds"]}
    return out


# ----------------------------------------------------------------------------- cookie cats
def synth_cookie_cats(seed: int) -> list[tuple]:
    rng = random.Random(seed)
    counts = group_counts()
    ids = rng.sample(range(116, 10_000_000), COOKIE_CATS_N)
    groups = ["gate_30"] * counts["gate_30"]["n"] + ["gate_40"] * counts["gate_40"]["n"]
    rng.shuffle(groups)
    by_group: dict[str, list[int]] = {"gate_30": [], "gate_40": []}
    for uid, g in zip(ids, groups):
        by_group[g].append(uid)

    rows = []
    for g, uids in by_group.items():
        c = counts[g]
        eng = [rng.gauss(0, 1) for _ in uids]
        order = sorted(range(len(uids)), key=lambda i: eng[i])
        zero = set(order[: c["zero"]])
        rounds = [0 if i in zero else max(1, int(round(math.exp(2.77 + 1.35 * (0.8 * eng[i] + 0.6 * rng.gauss(0, 1))))))
                  for i in range(len(uids))]
        played = [i for i in range(len(uids)) if i not in zero]
        s1 = {i: 0.9 * eng[i] + 0.25 * math.log1p(rounds[i]) + rng.gauss(0, 0.6) for i in played}
        d1 = set(sorted(played, key=s1.get, reverse=True)[: c["d1"]])
        s7 = {i: eng[i] + 0.3 * math.log1p(rounds[i]) + (0.8 if i in d1 else 0) + rng.gauss(0, 0.6) for i in played}
        d7 = set(sorted(played, key=s7.get, reverse=True)[: c["d7"]])
        rows += [(uids[i], g, rounds[i], i in d1, i in d7, "synthetic_calibrated") for i in range(len(uids))]
    rows.sort()
    return rows


def load_cookie_cats(con, seed: int, csv_path: Path | None, force_synth: bool) -> str:
    if csv_path and csv_path.exists() and not force_synth:
        cols = {r[0].lower() for r in con.execute("DESCRIBE SELECT * FROM read_csv_auto(?)", [str(csv_path)]).fetchall()}
        need = {"userid", "version", "sum_gamerounds", "retention_1", "retention_7"}
        if not need <= cols:
            raise SystemExit(f"{csv_path} missing columns {need - cols}; re-run with --synthetic")
        con.execute("""INSERT INTO cookie_cats SELECT userid, version, sum_gamerounds,
                       CAST(retention_1 AS BOOLEAN), CAST(retention_7 AS BOOLEAN), 'kaggle_csv'
                       FROM read_csv_auto(?)""", [str(csv_path)])
        return "kaggle_csv"
    _bulk_copy(con, "cookie_cats", ["userid", "version", "sum_gamerounds", "retention_1", "retention_7", "provenance"],
               synth_cookie_cats(seed))
    return "synthetic_calibrated"


# ----------------------------------------------------------------------------- cohort
def stratified_sample(con, n: int, rng: random.Random) -> list[tuple]:
    cells = con.execute("""SELECT version, retention_1, retention_7, count(*) FROM cookie_cats
                           GROUP BY ALL ORDER BY ALL""").fetchall()
    total = sum(c[3] for c in cells)
    quotas = [(c, n * c[3] / total) for c in cells]
    alloc = {c[:3]: int(q) for c, q in quotas}
    for c, q in sorted(quotas, key=lambda t: t[1] - int(t[1]), reverse=True)[: n - sum(alloc.values())]:
        alloc[c[:3]] += 1
    out = []
    for (g, r1, r7), k in alloc.items():
        pool = con.execute("""SELECT userid, version, sum_gamerounds, retention_1, retention_7 FROM cookie_cats
                              WHERE version=? AND retention_1=? AND retention_7=? ORDER BY userid""", [g, r1, r7]).fetchall()
        out += rng.sample(pool, k)
    out.sort()
    return out


def _pick(rng: random.Random, table):
    return rng.choices([t[0] for t in table], weights=[t[1] for t in table])[0]


def generate_events(players: list[tuple], levels: list[dict], lex: Lexicon, rng: random.Random):
    lv = {l["level_id"]: l for l in levels}
    # words ranked common->rare: players find frequent words first
    words = {l["level_id"]: sorted(json.loads(l["valid_words"]), key=lambda w: -lex.zipf[w]) for l in levels}
    tier = {c[0]: c[2] for c in COUNTRIES}
    max_level = max(lv)
    events: list[list] = []
    player_rows = []
    seq = 0

    def emit(ts, uid, sid, name, ab, plat, level=None, attempt=None, word=None, rev=None, net=None, place=None, props=None):
        nonlocal seq
        seq += 1
        events.append([f"seed-{seq:08d}", ts.isoformat(" "), uid, sid, name, level,
                       1 if level else None, attempt, ab, plat, word, rev, net, place, "seed",
                       json.dumps(props or {}, separators=(",", ":"))])

    def ad(ts, uid, sid, ab, plat, country, level, attempt, placement):
        net = rng.choices(list(NETWORKS), weights=NETWORK_WEIGHTS)[0]
        ecpm = NETWORKS[net] * tier[country] * (1.6 if placement == "rewarded_hint" else 1.0) * math.exp(rng.gauss(0, 0.35))
        emit(ts, uid, sid, "ad_impression", ab, plat, level, attempt, rev=round(ecpm / 1000, 6), net=net,
             place=placement, props={"ecpm": round(ecpm, 3), "country": country})

    for uid_num, ab, rounds, r1, r7 in players:
        uid = f"cc_{uid_num}"
        country, plat, chan = _pick(rng, COUNTRIES), rng.choice(["ios", "android", "android"]), _pick(rng, CHANNELS)
        install = COHORT_START + timedelta(days=rng.randrange(0, 24), hours=rng.uniform(7, 19))
        player_rows.append([uid, install, ab, country, plat, chan, rounds, r1, r7, "seed"])
        attempts_left = min(rounds, ATTEMPT_CAP)
        eng = min(1.0, math.log1p(rounds) / math.log1p(200))
        days = [0] + ([1] if r1 else []) + ([7] if r7 else [])
        days += [d for d in range(2, 14) if d != 7 and rng.random() < 0.03 + 0.35 * eng]
        days.sort()
        sessions = []
        for d in days:
            for k in range(2 if rng.random() < 0.3 else 1):
                base = install if d == 0 else datetime.combine((install + timedelta(days=d)).date(), datetime.min.time())
                # day-0 sessions stay inside day 0 so they can never masquerade as D1 activity
                start = base + timedelta(hours=(k * 1.5 + 0.05) if d == 0 else (k * 4 + rng.uniform(8, 18)))
                sessions.append(start)
        if rounds == 0:
            sessions = sessions[:1]
        # split attempts across sessions (front-loaded, like real play)
        weights = [rng.random() * (1.0 / (1 + i * 0.3)) for i in range(len(sessions))]
        alloc = [0] * len(sessions)
        for _ in range(attempts_left):
            alloc[rng.choices(range(len(sessions)), weights=weights)[0]] += 1
        level, completed = 1, set()
        for s_idx, (start, n_att) in enumerate(zip(sessions, alloc)):
            sid = f"s_{uid_num}_{s_idx}"
            ts = start
            emit(ts, uid, sid, "session_start", ab, plat, props={"day": (start.date() - install.date()).days, "channel": chan})
            for a in range(n_att):
                replay = level > max_level
                lid = rng.randint(1, max_level) if replay else level
                spec = lv[lid]
                att = f"{sid}_a{a}"
                ts += timedelta(seconds=rng.uniform(2, 12))
                emit(ts, uid, sid, "level_start", ab, plat, lid, att, props={"replay": replay, "target": spec["target_words"]})
                fail = rng.random() < FAIL_RATE[lid]
                target, limit = spec["target_words"], spec["time_limit_s"]
                n_found = rng.randint(0, max(0, target - 1)) if fail else target
                dur = (limit if rng.random() > 0.15 else rng.uniform(10, limit)) if fail else rng.uniform(0.45, 0.95) * limit
                ranked = words[lid]
                pool = rng.sample(ranked[: min(len(ranked), n_found * 2 + 2)], n_found) if n_found else []
                for i, w in enumerate(pool):
                    wt = ts + timedelta(seconds=dur * (i + 1) / (n_found + 1))
                    emit(wt, uid, sid, "word_found", ab, plat, lid, att, word=w,
                         props={"len": len(w), "zipf": lex.zipf[w], "n": i + 1})
                if rng.random() < (0.25 if fail else 0.08):
                    ad(ts + timedelta(seconds=dur * 0.6), uid, sid, ab, plat, country, lid, att, "rewarded_hint")
                ts += timedelta(seconds=dur)
                emit(ts, uid, sid, "level_fail" if fail else "level_complete", ab, plat, lid, att,
                     props={"found": n_found, "target": target, "duration_s": round(dur, 1), "replay": replay})
                if rng.random() < 0.45:
                    ad(ts + timedelta(seconds=1), uid, sid, ab, plat, country, lid, att, "interstitial")
                if not fail and not replay:
                    completed.add(lid)
                    level += 1
    return events, player_rows


# ----------------------------------------------------------------------------- io
def _bulk_copy(con, table: str, cols: list[str], rows) -> None:
    tmp = get_settings().data_dir / f".seed_{table}_{os.getpid()}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(rows)
        con.execute(f"COPY {table} ({','.join(cols)}) FROM '{tmp.as_posix()}' (HEADER, NULL '')")
    finally:
        tmp.unlink(missing_ok=True)


def wipe(con) -> None:
    for t in ("events", "players", "cookie_cats", "levels"):
        con.execute(f"DELETE FROM {t}")


def verify(con) -> dict:
    cc = con.execute("""SELECT version, count(*) n, round(100*count(*)/sum(count(*)) OVER (),5) pct,
                               round(avg(retention_1::INT),6) d1, round(avg(retention_7::INT),6) d7,
                               sum((sum_gamerounds=0)::INT) zero, any_value(provenance)
                        FROM cookie_cats GROUP BY version ORDER BY version""").fetchall()
    cohort = con.execute("""
        WITH first AS (SELECT user_id, min(ts)::DATE d0 FROM events WHERE source='seed' AND event_name='session_start' GROUP BY 1),
             act AS (SELECT DISTINCT e.user_id, (e.ts::DATE - f.d0) AS day FROM events e JOIN first f USING (user_id)
                     WHERE e.event_name='session_start')
        SELECT p.ab_group, count(DISTINCT p.user_id) n,
               round(avg(CASE WHEN EXISTS (SELECT 1 FROM act a WHERE a.user_id=p.user_id AND a.day=1) THEN 1 ELSE 0 END),4) d1,
               round(avg(CASE WHEN EXISTS (SELECT 1 FROM act a WHERE a.user_id=p.user_id AND a.day=7) THEN 1 ELSE 0 END),4) d7
        FROM players p WHERE p.source='seed' GROUP BY 1 ORDER BY 1""").fetchall()
    funnel = con.execute("""SELECT level_id, sum((event_name='level_start')::INT) starts,
                                   round(sum((event_name='level_fail')::INT)/nullif(sum((event_name IN ('level_fail','level_complete'))::INT),0),3) fail_rate
                            FROM events WHERE level_id IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchall()
    n_events = con.execute("SELECT count(*) FROM events").fetchone()[0]
    return {"cookie_cats": cc, "cohort": cohort, "funnel": funnel, "events": n_events}


def check_calibration(report: dict) -> list[str]:
    errs = []
    counts = group_counts()
    for g, n, pct, d1, d7, zero, prov in report["cookie_cats"]:
        if prov != "synthetic_calibrated":
            continue
        t = TARGETS[g]
        if n != counts[g]["n"] or zero != t["zero_rounds"] or abs(d1 - t["d1"]) > 1e-5 or abs(d7 - t["d7"]) > 1e-5:
            errs.append(f"{g}: got n={n} d1={d1} d7={d7} zero={zero}, expected {counts[g]} / {t}")
    return errs


def seed_database(con, *, n_players: int = 2500, seed: int = 42, csv_path: Path | None = None,
                  force_synth: bool = False, lex: Lexicon | None = None) -> dict:
    """Seed an EMPTY database in-place. Used by the CLI and by server startup."""
    t0 = time.perf_counter()
    lex = lex or load_lexicon()
    rng = random.Random(seed)
    levels = build_seed_levels(lex)
    insert_levels(con, levels)
    provenance = load_cookie_cats(con, seed, csv_path, force_synth)
    players = stratified_sample(con, n_players, rng)
    events, player_rows = generate_events(players, levels, lex, rng)
    _bulk_copy(con, "players", ["user_id", "install_ts", "ab_group", "country", "platform", "channel",
                                "sum_gamerounds", "retention_1", "retention_7", "source"],
               [[r[0], r[1].strftime("%Y-%m-%d %H:%M:%S"), *r[2:]] for r in player_rows])
    _bulk_copy(con, "events", EVENT_COLUMNS, events)
    report = verify(con)
    report.update(provenance=provenance, seconds=round(time.perf_counter() - t0, 2))
    return report


def print_report(r: dict) -> None:
    print(f"\nCookie Cats ({r['provenance']}):")
    print(f"  {'group':8} {'players':>8} {'split%':>9} {'D1':>9} {'D7':>9} {'0 rounds':>9}")
    for g, n, pct, d1, d7, zero, _ in r["cookie_cats"]:
        print(f"  {g:8} {n:>8} {pct:>9} {d1:>9} {d7:>9} {zero:>9}")
    print("\nEvent cohort retention (recomputed from session_start events):")
    for g, n, d1, d7 in r["cohort"]:
        print(f"  {g:8} players={n:<5} D1={d1:<7} D7={d7}")
    print("\nLevel funnel (per-attempt fail rate):")
    for lid, starts, fr in r["funnel"]:
        flag = "   <-- spike" if fr and fr > 0.5 else ""
        print(f"  L{lid:<3} starts={starts:<6} fail={fr}{flag}")
    print(f"\n{r['events']:,} events seeded in {r['seconds']}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed data/telemetry.duckdb (Cookie Cats + 2,500-player cohort)")
    ap.add_argument("--force", action="store_true", help="wipe ALL tables (incl. live events) and reseed")
    ap.add_argument("--players", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", action="store_true", help="ignore data/cookie_cats.csv even if present")
    a = ap.parse_args()
    s = get_settings()
    try:
        con = open_rw(s.telemetry_db)
    except DatabaseLockedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        if con.execute("SELECT count(*) FROM levels").fetchone()[0] and not a.force:
            print(f"{s.telemetry_db} already seeded - nothing to do (use --force to wipe & reseed).")
            print_report({**verify(con), "provenance": "existing", "seconds": 0})
            return 0
        con.execute("BEGIN")
        wipe(con)
        report = seed_database(con, n_players=a.players, seed=a.seed, csv_path=s.cookie_cats_csv, force_synth=a.synthetic)
        errs = check_calibration(report)
        if errs:
            con.execute("ROLLBACK")
            print("CALIBRATION FAILED:\n  " + "\n  ".join(errs), file=sys.stderr)
            return 1
        con.execute("COMMIT")
        print_report(report)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
