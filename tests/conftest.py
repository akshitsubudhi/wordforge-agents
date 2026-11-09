"""Suite-wide isolation: the tests are OFFLINE and deterministic.

A developer (or CI runner) may have ANTHROPIC_API_KEY / OPENAI_API_KEY exported. Without this fixture the agents
would pick the real provider up through core.harness.llm.get_llm() and the suite would spend money, need network
and stop being reproducible. Tests that exercise the real-provider path opt back in explicitly with monkeypatch.
"""
import os
import pickle
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    from core.harness.llm import HEALTH

    HEALTH.reset()   # provider health is process-wide: one test's simulated outage must not leak into the next


# ---------------------------------------------------------------------------- shared seeded databases
# Seeding (Cookie Cats table + event cohort) costs ~5 s and used to run once per module - and once per TEST in
# test_phase3. Each distinct (n_players, seed) is now built ONCE per pytest invocation and every consumer gets its
# own private COPY of the file (a ~10 ms copy), so tests can mutate freely without seeing each other's writes.
def _template_dir(tmp_path_factory) -> Path:
    base = tmp_path_factory.getbasetemp()
    # under pytest-xdist every worker has its own basetemp below a shared parent: share the templates there
    root = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base
    d = root / "seed_templates"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture(scope="session")
def seeded_copy(tmp_path_factory):
    """seeded_copy(dest_dir, n_players=..., seed=...) -> seed report. Writes dest_dir/telemetry.duckdb."""
    from core.db import open_rw
    from scripts.seed_telemetry import seed_database

    tdir = _template_dir(tmp_path_factory)

    def copy(dest_dir: Path, *, n_players: int, seed: int) -> dict:
        tpl, rep = tdir / f"seed_{n_players}_{seed}.duckdb", tdir / f"seed_{n_players}_{seed}.pkl"
        if not rep.exists():   # build under a private name, then publish atomically (safe across xdist workers)
            tmp = tdir / f".build_{os.getpid()}_{n_players}_{seed}.duckdb"
            tmp.unlink(missing_ok=True)
            con = open_rw(tmp)
            report = seed_database(con, n_players=n_players, seed=seed)
            con.execute("CHECKPOINT")
            con.close()
            os.replace(tmp, tpl)
            rep_tmp = tmp.with_suffix(".pkl")
            rep_tmp.write_bytes(pickle.dumps(report))
            os.replace(rep_tmp, rep)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tpl, dest_dir / "telemetry.duckdb")
        return pickle.loads(rep.read_bytes())

    return copy
