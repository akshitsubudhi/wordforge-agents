"""
================================================================================
BLOCK: Tracing & SRE Metrics (OpenTelemetry-shaped spans -> DuckDB)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : runner.AgentRunner (agent.run / llm.call / tool.call spans) and
             any block that wants latency/cost accounting.
  Downstream: data/traces.duckdb `spans` table -> agent_metrics() p50/p95/p99
             latency, cost, success rate (Control Center dashboard);
             to_otlp_json() for shipping to any OTLP collector unchanged.

Key engineering mechanism
  * W3C-compatible ids (128-bit trace_id, 64-bit span_id); parent/child via
    contextvars, so nesting is correct across threads/async without plumbing.
  * Hot attributes (agent, tool, model, cost_usd, tokens) are promoted to real
    columns; the rest is a JSON string column (no JSON extension needed).
  * BatchSpanExporter: bounded deque buffer (maxlen) + background flusher
    (every flush_interval_s or batch_size spans). Each flush opens a
    SHORT-LIVED DuckDB connection and inserts the batch in one statement, so the
    dashboard / other processes can read between flushes (DuckDB is
    single-writer per file).
  * Metrics: quantile_cont() p50/p95/p99 per span name & tool in one SQL pass.

Failure modes & 1-step mitigation
  * DB locked / disk full / DB down -> flush fails, spans are put back in the
    buffer, exponential backoff before the next attempt; the AGENT NEVER
    BLOCKS OR CRASHES on telemetry.
  * Prolonged outage -> buffer is size-capped; oldest spans are dropped and
    counted (`dropped` stat) instead of growing memory unbounded.
  * Process exit -> close() retries the final flush with backoff, then
    SPILLS anything unwritten to <db>.spool.jsonl, which the next exporter
    ingests (fixed after a missing-trace incident; see
    docs/THREE_WAYS_MY_AGENT_BROKE.md).
  * Concurrent flushes (background thread + forced flush) -> serialised by
    a flush lock; they used to race on CREATE TABLE (write-write conflict).
================================================================================
"""
from __future__ import annotations

import atexit
import contextvars
import json
import secrets
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

import duckdb

_current_span: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("current_span", default=None)

PROMOTED = ("agent", "tool", "model", "cost_usd", "input_tokens", "output_tokens", "run_id")

SPANS_DDL = """CREATE TABLE IF NOT EXISTS spans (
    trace_id VARCHAR, span_id VARCHAR, parent_span_id VARCHAR, name VARCHAR, kind VARCHAR,
    service_name VARCHAR, start_ts TIMESTAMP, end_ts TIMESTAMP, duration_ms DOUBLE,
    status VARCHAR, status_message VARCHAR, run_id VARCHAR, agent VARCHAR, tool VARCHAR,
    model VARCHAR, cost_usd DOUBLE, input_tokens INTEGER, output_tokens INTEGER,
    attributes VARCHAR, events VARCHAR)"""
SPAN_COLS = ["trace_id", "span_id", "parent_span_id", "name", "kind", "service_name", "start_ts", "end_ts",
             "duration_ms", "status", "status_message", "run_id", "agent", "tool", "model", "cost_usd",
             "input_tokens", "output_tokens", "attributes", "events"]


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    kind: str = "INTERNAL"
    service_name: str = "agent-harness"
    start_ns: int = field(default_factory=time.time_ns)
    end_ns: int | None = None
    status: str = "UNSET"  # UNSET | OK | ERROR (OTel StatusCode)
    status_message: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def set(self, **attrs) -> "Span":
        self.attributes.update(attrs)
        return self

    def add_event(self, name: str, **attrs) -> None:
        self.events.append({"name": name, "time_ns": time.time_ns(), "attributes": attrs})

    def error(self, msg: str) -> None:
        self.status, self.status_message = "ERROR", msg[:500]

    @property
    def duration_ms(self) -> float:
        return ((self.end_ns or time.time_ns()) - self.start_ns) / 1e6

    def to_row(self) -> list:
        a = self.attributes
        ts = lambda ns: datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).replace(tzinfo=None)
        rest = {k: v for k, v in a.items() if k not in PROMOTED}
        return [self.trace_id, self.span_id, self.parent_span_id, self.name, self.kind, self.service_name,
                ts(self.start_ns), ts(self.end_ns or time.time_ns()), round(self.duration_ms, 3), self.status,
                self.status_message, a.get("run_id"), a.get("agent"), a.get("tool"), a.get("model"),
                a.get("cost_usd"), a.get("input_tokens"), a.get("output_tokens"),
                json.dumps(rest, default=str), json.dumps(self.events, default=str)]

    def to_otlp(self) -> dict:
        def val(v):
            if isinstance(v, bool):
                return {"boolValue": v}
            if isinstance(v, int):
                return {"intValue": str(v)}
            if isinstance(v, float):
                return {"doubleValue": v}
            return {"stringValue": v if isinstance(v, str) else json.dumps(v, default=str)}

        return {
            "traceId": self.trace_id, "spanId": self.span_id, "parentSpanId": self.parent_span_id or "",
            "name": self.name, "kind": f"SPAN_KIND_{self.kind}",
            "startTimeUnixNano": str(self.start_ns), "endTimeUnixNano": str(self.end_ns or time.time_ns()),
            "attributes": [{"key": k, "value": val(v)} for k, v in self.attributes.items()],
            "events": [{"name": e["name"], "timeUnixNano": str(e["time_ns"]),
                        "attributes": [{"key": k, "value": val(v)} for k, v in e["attributes"].items()]} for e in self.events],
            "status": {"code": {"UNSET": 0, "OK": 1, "ERROR": 2}[self.status], "message": self.status_message},
        }


def to_otlp_json(spans: list[Span], service_name: str = "agent-harness") -> dict:
    """OTLP/JSON ExportTraceServiceRequest - POST it to any collector's /v1/traces."""
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": service_name}}]},
        "scopeSpans": [{"scope": {"name": "liveops.harness", "version": "1.0"}, "spans": [s.to_otlp() for s in spans]}],
    }]}


class SpanExporter(Protocol):
    def export(self, span: Span) -> None: ...


class InMemoryExporter:
    def __init__(self):
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)


def _insert_spans(con, rows: list[list]) -> None:
    """One multi-row parameterised INSERT per chunk (executemany is one round trip PER ROW in duckdb)."""
    one = "(" + ",".join("?" * len(SPAN_COLS)) + ")"
    for i in range(0, len(rows), 200):
        part = rows[i:i + 200]
        con.execute(f"INSERT INTO spans ({','.join(SPAN_COLS)}) VALUES " + ",".join([one] * len(part)),
                    [v for r in part for v in r])


# One short-lived connection to a spans DB at a time PER PROCESS. DuckDB refuses a second attach of the same
# file inside one process ("Unique file handle conflict") when a flush and a dashboard read overlap, and
# another PROCESS holding the file lock raises IOException: both are retried here with backoff.
_TRACES_IO = threading.RLock()


class TracesBusyError(RuntimeError):
    """The spans DB stayed locked by another process for the whole retry window."""


@contextmanager
def traces_connection(db_path: Path | str, *, attempts: int = 6, base_s: float = 0.05, ensure_schema: bool = True):
    last: Exception | None = None
    with _TRACES_IO:
        con = None
        for k in range(max(1, attempts)):
            try:
                con = duckdb.connect(str(db_path))
                break
            except duckdb.Error as e:  # IOException (other process) / ResourceInUse / ConnectionException
                last = e
                if k < attempts - 1:
                    time.sleep(base_s * 2 ** k)
        if con is None:
            raise TracesBusyError(f"{db_path} is busy ({type(last).__name__}: {str(last)[:140]}); "
                                  "another process is writing spans - retry in a moment") from last
        try:
            if ensure_schema:
                con.execute(SPANS_DDL)
            yield con
        finally:
            con.close()


class DuckDBSpanExporter:
    def __init__(self, db_path: Path | str, *, max_buffer: int = 10_000, batch_size: int = 256,
                 flush_interval_s: float = 1.0, background: bool = True):
        self.db_path = Path(db_path)
        self.max_buffer, self.batch_size, self.flush_interval_s = max_buffer, batch_size, flush_interval_s
        self.buffer: deque[Span] = deque()
        self.stats = {"exported": 0, "dropped": 0, "flush_failures": 0, "last_error": "", "spilled": 0, "unspooled": 0}
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()   # ONE flusher at a time: concurrent DuckDB writers raced on CREATE TABLE
        self.spool_path = self.db_path.with_suffix(".spool.jsonl")
        self._backoff_until = 0.0
        self._consecutive_failures = 0
        self._stop = threading.Event()
        self._schema_ready = False
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="span-exporter")
            self._thread.start()
        atexit.register(self.close)

    def export(self, span: Span) -> None:
        with self._lock:
            if len(self.buffer) >= self.max_buffer:
                self.buffer.popleft()
                self.stats["dropped"] += 1
            self.buffer.append(span)

    def _loop(self) -> None:
        while not self._stop.wait(self.flush_interval_s):
            self.flush()

    def flush(self, force: bool = False, retries: int = 0) -> int:
        """Never raises. Returns spans written. force=True waits for an in-flight flush instead of racing it."""
        if not force and time.monotonic() < self._backoff_until:
            return 0
        if not self._flush_lock.acquire(blocking=force):
            return 0  # background tick while another flush runs: skip, never race
        try:
            written = self._flush_locked()
            for k in range(retries):
                if not self.buffer:
                    break
                time.sleep(0.05 * 2 ** k)
                written += self._flush_locked()
            return written
        finally:
            self._flush_lock.release()

    def _ingest_spool(self, con) -> None:
        """Spans spilled by a previous process that could not reach the DB at shutdown."""
        if not self.spool_path.exists():
            return
        rows = [json.loads(line) for line in self.spool_path.read_text().splitlines() if line.strip()]
        if rows:
            _insert_spans(con, rows)
        self.spool_path.unlink()
        self.stats["unspooled"] += len(rows)

    def _flush_locked(self) -> int:
        written = 0
        while True:
            with self._lock:
                batch = [self.buffer.popleft() for _ in range(min(self.batch_size, len(self.buffer)))]
            if not batch:
                return written
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                with traces_connection(self.db_path, attempts=2, ensure_schema=not self._schema_ready) as con:
                    if not self._schema_ready:
                        self._ingest_spool(con)
                        self._schema_ready = True
                    _insert_spans(con, [s.to_row() for s in batch])
                written += len(batch)
                self.stats["exported"] += len(batch)
                self._consecutive_failures = 0
            except Exception as e:  # DB down / locked: keep spans, back off
                with self._lock:
                    room = self.max_buffer - len(self.buffer)
                    keep = batch[-room:] if room > 0 else []
                    self.stats["dropped"] += len(batch) - len(keep)
                    self.buffer.extendleft(reversed(keep))
                self.stats["flush_failures"] += 1
                self.stats["last_error"] = f"{type(e).__name__}: {str(e)[:160]}"
                self._consecutive_failures += 1
                self._backoff_until = time.monotonic() + min(30.0, 0.5 * 2 ** self._consecutive_failures)
                return written

    def close(self) -> None:
        """Shutdown: retry the final flush, and SPILL anything still unwritten to a JSONL spool (ingested by the
        next exporter) instead of silently losing it when the process exits."""
        self._stop.set()
        self.flush(force=True, retries=5)
        with self._lock:
            left = list(self.buffer)
            self.buffer.clear()
        if left:
            try:
                with self.spool_path.open("a", encoding="utf-8") as fh:
                    for s in left:
                        row = s.to_row()
                        row[6], row[7] = str(row[6]), str(row[7])
                        fh.write(json.dumps(row, default=str) + "\n")
                self.stats["spilled"] += len(left)
            except OSError:
                self.stats["dropped"] += len(left)


class Tracer:
    def __init__(self, exporter: SpanExporter | None = None, service_name: str = "agent-harness"):
        self.exporter = exporter or InMemoryExporter()
        self.service_name = service_name

    @contextmanager
    def span(self, name: str, kind: str = "INTERNAL", new_trace: bool = False, **attrs) -> Iterator[Span]:
        parent = None if new_trace else _current_span.get()
        sp = Span(name=name, trace_id=parent.trace_id if parent else secrets.token_hex(16), span_id=secrets.token_hex(8),
                  parent_span_id=parent.span_id if parent else None, kind=kind, service_name=self.service_name)
        if parent:  # inherit correlation attributes
            for k in ("run_id", "agent"):
                if k in parent.attributes:
                    sp.attributes[k] = parent.attributes[k]
        sp.attributes.update(attrs)
        token = _current_span.set(sp)
        try:
            yield sp
            if sp.status == "UNSET":
                sp.status = "OK"
        except BaseException as e:
            sp.error(f"{type(e).__name__}: {e}")
            raise
        finally:
            sp.end_ns = time.time_ns()
            _current_span.reset(token)
            try:
                self.exporter.export(sp)
            except Exception:  # telemetry must never break the agent
                pass


_default: Tracer | None = None


def default_tracer() -> Tracer:
    global _default
    if _default is None:
        from core.config import get_settings

        _default = Tracer(DuckDBSpanExporter(get_settings().traces_db))
    return _default


def agent_metrics(db_path: Path | str, since_minutes: float | None = None) -> list[dict]:
    """p50/p95/p99 latency, cost, tokens and success rate per span name/tool."""
    where = f"WHERE start_ts >= now()::TIMESTAMP - INTERVAL {float(since_minutes)} MINUTE" if since_minutes else ""
    sql = f"""SELECT name, coalesce(tool, '') AS tool, count(*) AS n,
                     round(avg((status <> 'ERROR')::INT), 4) AS success_rate,
                     round(quantile_cont(duration_ms, 0.50), 2) AS p50_ms,
                     round(quantile_cont(duration_ms, 0.95), 2) AS p95_ms,
                     round(quantile_cont(duration_ms, 0.99), 2) AS p99_ms,
                     round(coalesce(sum(cost_usd), 0), 6) AS cost_usd,
                     coalesce(sum(input_tokens + output_tokens), 0) AS tokens
              FROM spans {where} GROUP BY ALL ORDER BY name, tool"""
    with traces_connection(db_path) as con:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
