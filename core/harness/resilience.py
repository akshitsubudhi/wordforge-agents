"""
================================================================================
BLOCK: Resilience Primitives (retry, circuit breaker, cost guard, timeouts,
       subprocess sandbox)
--------------------------------------------------------------------------------
Role & contracts
  Stateless helpers + small state machines consumed by runner.AgentRunner and
  by any agent that calls flaky infrastructure. No agent logic lives here.

Key engineering mechanism
  * retry_call: exponential backoff with FULL jitter
        sleep_k ~ U(0, min(cap, base * 2^k))
    (AWS Architecture Blog, "Exponential Backoff and Jitter") - decorrelates
    retry storms. The runner calls it ONLY for tools flagged idempotent, so
    a send_email is never duplicated by a retry.
  * CircuitBreaker: CLOSED -> OPEN after N consecutive failures; OPEN rejects
    instantly for recovery_s; HALF_OPEN admits limited probes; a success
    closes it. Injectable clock -> deterministic tests.
  * BudgetGuard: PRE-call worst-case check
        spent + price_in*est_in + price_out*max_out <= max_usd
    so the hard USD / token cap can never be exceeded, not merely detected
    after the fact. Also caps steps (runaway-loop / denial-of-wallet guard).
  * call_with_timeout: daemon-thread watchdog for in-process tools.
  * run_sandboxed_python: `python -I -S` child with rlimits (CPU, AS, FSIZE,
    NOFILE), empty env, temp cwd, wall-clock kill of the process group and a
    PEP-578 audit hook (see _sandbox_boot.py) denying network + process spawn.

Failure modes & 1-step mitigation
  * Retrying a non-idempotent write -> impossible by construction (flag).
  * Thread timeouts cannot kill Python threads -> the caller is unblocked and
    the result discarded; truly untrusted/CPU-bound code goes to the
    subprocess sandbox, which IS killed (killpg).
  * Breaker stuck open -> half-open probe after recovery_s, automatic.
================================================================================
"""
from __future__ import annotations

import contextvars
import os
import random
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

from core.harness.llm import Usage, price_for

T = TypeVar("T")


# ============================================================================ retry
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_s: float = 0.2
    cap_s: float = 5.0


def full_jitter_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    return rng.uniform(0, min(policy.cap_s, policy.base_s * (2 ** attempt)))


def retry_call(fn: Callable[[], T], policy: RetryPolicy = RetryPolicy(), *, retry_on: tuple[type[BaseException], ...] = (Exception,),
               rng: random.Random | None = None, sleep: Callable[[float], None] = time.sleep,
               on_retry: Callable[[int, BaseException, float], None] | None = None) -> tuple[T, int]:
    """Returns (result, attempts_used). Re-raises the last error when exhausted."""
    rng = rng or random.Random()
    for attempt in range(policy.max_attempts):
        try:
            return fn(), attempt + 1
        except retry_on as e:
            if attempt == policy.max_attempts - 1:
                raise
            delay = full_jitter_delay(attempt, policy, rng)
            if on_retry:
                on_retry(attempt + 1, e, delay)
            sleep(delay)
    raise AssertionError("unreachable")


# ============================================================================ breaker
class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, name: str, failure_threshold: int = 5, recovery_s: float = 30.0, half_open_max: int = 1,
                 clock: Callable[[], float] = time.monotonic):
        self.name, self.failure_threshold, self.recovery_s, self.half_open_max = name, failure_threshold, recovery_s, half_open_max
        self.clock = clock
        self._state, self._failures, self._opened_at, self._probes = self.CLOSED, 0, 0.0, 0
        self._lock = threading.Lock()
        self.transitions: list[tuple[str, str]] = []

    def _set(self, new: str) -> None:
        if new != self._state:
            self.transitions.append((self._state, new))
            self._state = new

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN and self.clock() - self._opened_at >= self.recovery_s:
                self._set(self.HALF_OPEN)
                self._probes = 0
            return self._state

    def allow(self) -> None:
        st = self.state
        with self._lock:
            if st == self.OPEN:
                raise CircuitOpenError(f"circuit '{self.name}' open; retry in {self.recovery_s - (self.clock() - self._opened_at):.1f}s")
            if st == self.HALF_OPEN:
                if self._probes >= self.half_open_max:
                    raise CircuitOpenError(f"circuit '{self.name}' half-open; probe in flight")
                self._probes += 1

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._set(self.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                self._set(self.OPEN)
                self._opened_at = self.clock()

    @contextmanager
    def guard(self):
        self.allow()
        try:
            yield
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()


class BreakerRegistry:
    def __init__(self, **defaults):
        self.defaults = defaults
        self._b: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._b:
                self._b[name] = CircuitBreaker(name, **self.defaults)
            return self._b[name]

    def snapshot(self) -> dict[str, str]:
        return {k: b.state for k, b in self._b.items()}


# ============================================================================ budget
class BudgetExceeded(RuntimeError):
    def __init__(self, kind: str, detail: str):
        self.kind = kind
        super().__init__(f"{kind}: {detail}")


BudgetExceededError = BudgetExceeded  # public alias: the name the docs and tests use


@dataclass
class BudgetGuard:
    max_usd: float = 0.25
    max_tokens: int = 200_000
    max_steps: int = 12
    spent_usd: float = 0.0
    tokens: int = 0
    steps: int = 0
    llm_calls: int = 0

    def precheck_llm(self, model: str, est_input_tokens: int, max_output_tokens: int) -> float:
        pin, pout = price_for(model)
        worst = (est_input_tokens * pin + max_output_tokens * pout) / 1e6
        if self.spent_usd + worst > self.max_usd:
            raise BudgetExceeded("usd", f"next call worst-case ${worst:.5f} would exceed ${self.max_usd:.4f} (spent ${self.spent_usd:.5f})")
        if self.tokens + est_input_tokens + max_output_tokens > self.max_tokens:
            raise BudgetExceeded("tokens", f"next call could reach {self.tokens + est_input_tokens + max_output_tokens} > {self.max_tokens}")
        return worst

    def charge_llm(self, model: str, usage: Usage) -> float:
        pin, pout = price_for(model)
        cost = (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6
        self.spent_usd += cost
        self.tokens += usage.input_tokens + usage.output_tokens
        self.llm_calls += 1
        return cost

    def charge_step(self) -> None:
        if self.steps >= self.max_steps:
            raise BudgetExceeded("steps", f"step limit {self.max_steps} reached")
        self.steps += 1

    def snapshot(self) -> dict:
        return {"spent_usd": round(self.spent_usd, 6), "tokens": self.tokens, "steps": self.steps,
                "llm_calls": self.llm_calls, "max_usd": self.max_usd, "max_tokens": self.max_tokens}


# ============================================================================ timeouts
class ToolTimeout(RuntimeError):
    pass


def call_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as e:  # propagate to caller
            box["error"] = e

    # run in a COPY of the caller's context: spans opened inside the tool (e.g. its llm.call) must stay children of
    # the tool.call span instead of starting orphan traces on the watchdog thread
    ctx = contextvars.copy_context()
    t = threading.Thread(target=lambda: ctx.run(target), daemon=True, name="tool-call")
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise ToolTimeout(f"exceeded {timeout_s}s")
    if "error" in box:
        raise box["error"]
    return box["result"]


# ============================================================================ sandbox
BOOT = Path(__file__).with_name("_sandbox_boot.py")


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    duration_ms: float
    blocked_events: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.blocked_events


def run_sandboxed_python(code: str, *, timeout_s: float = 3.0, cpu_s: int = 2, mem_mb: int = 256,
                         max_output: int = 64_000) -> SandboxResult:
    def limits():  # runs in the child between fork and exec
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sbx_") as cwd:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", str(BOOT)], cwd=cwd, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=limits, start_new_session=True,
        )
        timed_out = False
        try:
            out, err = proc.communicate(code.encode(), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
    so, se = out.decode(errors="replace")[:max_output], err.decode(errors="replace")[:max_output]
    blocked = [ln.split(":", 1)[1] for ln in se.splitlines() if ln.startswith("SANDBOX_BLOCKED:")]
    if proc.returncode is not None and proc.returncode < 0 and -proc.returncode == signal.SIGXCPU:
        timed_out = True
    return SandboxResult(so, se, proc.returncode, timed_out, round((time.perf_counter() - t0) * 1000, 1), blocked)
