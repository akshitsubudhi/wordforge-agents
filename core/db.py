"""
================================================================================
BLOCK: DuckDB Telemetry Store (schema + single-writer connection management)
--------------------------------------------------------------------------------
Role & contracts
  Owns data/telemetry.duckdb. Tables:
    events      - one row per telemetry event (seed + live share the table).
                  Hot properties are REAL columns (word, revenue_usd, network,
                  placement, level_id, level_version); the long tail goes into
                  `props` as a JSON *string* -> no JSON extension needed.
    players     - one row per player (event-level cohort + live players).
    cookie_cats - 90,189-row Cookie Cats A/B dataset (real CSV or calibrated).
    levels      - versioned level catalogue (level_id, version, status).
    feature_flags / approvals / experiment_exposures / agent_audit
                - Phase-2 experimentation state (flag cap enforced by a
                  CHECK constraint as a last line of defence).
  Upstream : app/game_api.py (live writes), scripts/seed_telemetry.py (bulk).
  Downstream: Phase-2 LiveOps Copilot / Level Factory SQL, dashboards.

Key engineering mechanism
  DuckDB is single-writer per file. `open_rw()` translates DuckDB's lock
  IOException into `DatabaseLockedError` carrying the holder PID and a one-line
  fix, so the seeder and server fail loudly and clearly instead of corrupting
  or hanging. A process-local RLock serialises writers inside the server.

Failure modes & 1-step mitigation
  * Seeder run while server is up -> DatabaseLockedError("... stop the server
    or use POST /api/admin/reseed") ; exit code 2.
  * Schema drift between phases -> CREATE ... IF NOT EXISTS + additive columns
    only; never destructive migrations.
================================================================================
"""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from pathlib import Path

import duckdb

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS events (
        event_id      VARCHAR,
        ts            TIMESTAMP,
        user_id       VARCHAR,
        session_id    VARCHAR,
        event_name    VARCHAR,
        level_id      INTEGER,
        level_version INTEGER,
        attempt_id    VARCHAR,
        ab_group      VARCHAR,
        platform      VARCHAR,
        word          VARCHAR,
        revenue_usd   DOUBLE,
        network       VARCHAR,
        placement     VARCHAR,
        source        VARCHAR,
        props         VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS players (
        user_id        VARCHAR PRIMARY KEY,
        install_ts     TIMESTAMP,
        ab_group       VARCHAR,
        country        VARCHAR,
        platform       VARCHAR,
        channel        VARCHAR,
        sum_gamerounds INTEGER,
        retention_1    BOOLEAN,
        retention_7    BOOLEAN,
        source         VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS cookie_cats (
        userid         BIGINT,
        version        VARCHAR,
        sum_gamerounds INTEGER,
        retention_1    BOOLEAN,
        retention_7    BOOLEAN,
        provenance     VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS levels (
        level_id         INTEGER,
        version          INTEGER,
        grid             VARCHAR,
        n_rows           INTEGER,
        n_cols           INTEGER,
        target_words     INTEGER,
        time_limit_s     INTEGER,
        valid_words      VARCHAR,   -- JSON array string, solver-verified
        word_count       INTEGER,
        mean_zipf        DOUBLE,
        branching_factor DOUBLE,
        status           VARCHAR,   -- live | rolled_back | retired | candidate
        created_by       VARCHAR,   -- seed | level_factory
        created_at       TIMESTAMP,
        notes            VARCHAR,
        PRIMARY KEY (level_id, version)
    )""",
    # additive migration (Phase 2): difficulty-model prediction stored with each version
    "ALTER TABLE levels ADD COLUMN IF NOT EXISTS predicted_fail_rate DOUBLE",
    """CREATE TABLE IF NOT EXISTS feature_flags (
        flag              VARCHAR PRIMARY KEY,
        level_id          INTEGER,
        control_version   INTEGER,
        treatment_version INTEGER,
        rollout_pct       DOUBLE CHECK (rollout_pct >= 0 AND rollout_pct <= 20),  -- DB-level exposure cap
        status            VARCHAR,   -- pending_approval | active | denied | rolled_back | shipped
        created_at        TIMESTAMP,
        updated_at        TIMESTAMP,
        notes             VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS approvals (
        approval_id VARCHAR PRIMARY KEY,
        ts          TIMESTAMP,
        agent       VARCHAR,
        tool        VARCHAR,
        args        VARCHAR,
        decision    VARCHAR,   -- approved | denied
        decided_by  VARCHAR,
        reason      VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS experiment_exposures (
        flag        VARCHAR,
        user_id     VARCHAR,
        variant     VARCHAR,   -- control | treatment
        day         INTEGER,
        exposed_at  TIMESTAMP,
        pre_metric  DOUBLE,    -- CUPED covariate: level completions in the 7 days before exposure
        outcome     DOUBLE,    -- primary: level completions in the 7 days after (NULL until observable)
        retained_d1 BOOLEAN,   -- guardrail (NULL until observable)
        source      VARCHAR    -- live | sim
    )""",
    """CREATE TABLE IF NOT EXISTS ad_mediation_config (
        version     INTEGER PRIMARY KEY,
        created_at  TIMESTAMP,
        mechanism   VARCHAR,
        reserve_cpm DOUBLE CHECK (reserve_cpm >= 0 AND reserve_cpm <= 50),
        disabled    VARCHAR,   -- JSON array of quarantined networks
        status      VARCHAR,   -- trial | good | rolled_back | superseded
        parent      INTEGER,
        reason      VARCHAR
    )""",
    """CREATE TABLE IF NOT EXISTS ad_hourly_metrics (
        run_id VARCHAR, hour INTEGER, network VARCHAR, config_version INTEGER,
        requests INTEGER, fills INTEGER, revenue_usd DOUBLE, timeouts INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS ua_allocations (
        run_id VARCHAR, day INTEGER, policy VARCHAR, channel VARCHAR, context VARCHAR,
        spend_usd DOUBLE, installs INTEGER, d1_ltv_usd DOUBLE
    )""",
    """CREATE TABLE IF NOT EXISTS agent_audit (
        ts     TIMESTAMP,
        agent  VARCHAR,
        action VARCHAR,
        detail VARCHAR
    )""",
    """CREATE OR REPLACE VIEW active_levels AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY level_id ORDER BY version DESC) AS rn
            FROM levels WHERE status = 'live')
        WHERE rn = 1""",
    "CREATE INDEX IF NOT EXISTS idx_events_name ON events(event_name)",
]

EVENT_COLUMNS = [
    "event_id", "ts", "user_id", "session_id", "event_name", "level_id", "level_version",
    "attempt_id", "ab_group", "platform", "word", "revenue_usd", "network", "placement",
    "source", "props",
]


class DatabaseLockedError(RuntimeError):
    def __init__(self, path: Path, raw: str):
        pid = re.search(r"PID (\d+)", raw)
        self.pid = int(pid.group(1)) if pid else None
        super().__init__(
            f"{path} is locked by another process"
            f"{f' (PID {self.pid})' if self.pid else ''} - most likely the game server "
            "(DuckDB allows one writer per file). Fix: stop the server (Ctrl-C / `make stop`) "
            "and re-run, or reseed through the running server: "
            "`curl -X POST localhost:8000/api/admin/reseed`."
        )


def open_rw(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        con = duckdb.connect(str(path))
    except duckdb.IOException as e:  # lock held by another process
        if "lock" in str(e).lower():
            raise DatabaseLockedError(path, str(e)) from e
        raise
    for stmt in SCHEMA:
        con.execute(stmt)
    return con


class TelemetryDB:
    """Long-lived writer used by the server. Thread-safe via an RLock."""

    def __init__(self, path: Path):
        self.path = path
        self.con = open_rw(path)
        self.lock = threading.RLock()

    @contextmanager
    def cursor(self):
        with self.lock:
            yield self.con

    def insert_events(self, rows: list[dict]) -> None:
        if not rows:
            return
        placeholders = ",".join("?" * len(EVENT_COLUMNS))
        vals = [[r.get(c) for c in EVENT_COLUMNS] for r in rows]
        with self.cursor() as con:
            con.executemany(f"INSERT INTO events ({','.join(EVENT_COLUMNS)}) VALUES ({placeholders})", vals)

    def query(self, sql: str, params: list | None = None) -> list[dict]:
        with self.cursor() as con:
            cur = con.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def is_empty(self) -> bool:
        with self.cursor() as con:
            return con.execute("SELECT count(*) FROM levels").fetchone()[0] == 0

    def close(self) -> None:
        with self.lock:
            self.con.close()
