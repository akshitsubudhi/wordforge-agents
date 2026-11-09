# LiveOps Agent Platform

A playable, server-refereed word game whose telemetry feeds a set of production-style AI agents.
Everything runs on one Python process with FastAPI, DuckDB and vanilla JS. There is no Docker,
nothing to provision, and no API key is needed: without one, a deterministic mock LLM drives
every agent, and all tests and evals run offline.

```bash
make setup      # uv if installed, otherwise venv + pip
make run        # http://localhost:8000/game  (seeds an empty DB on first start, ~10 s)
make check      # pytest + guardrail conformance suite (CI gate)
```

## Phase 1

| Component | Where | Proof |
|---|---|---|
| LetterLoop word game: server referee, levels 1–12, ads, hints | `app/` | `test_game_end_to_end` |
| DuckDB telemetry store; Cookie Cats + 2,500-player cohort | `core/db.py`, `scripts/seed_telemetry.py` | D1 / D7 / split / zero-round counts match exactly |
| Versioned levels with solver-gated publish and one-step rollback | `core/levels.py` | `test_publish_and_rollback` |
| SRE agent harness | `core/harness/` | retry, breaker, budget, timeout, sandbox and telemetry tests |
| Three-layer guardrails with a 5-way ablation | `core/harness/redteam_guard.py`, `run_redteam_eval.py` | all_on blocks **25/25**, **0/10** false alarms |

## Phase 2

| Component | Where | Proof |
|---|---|---|
| LiveOps Experimentation Copilot: anomaly scan, power, ≤20% flag with human approval, CUPED + mSPRT, automatic rollback | `agents/liveops_copilot/` | `make phase2` scorecard; `tests/test_phase2.py` |
| Autonomous Level Factory: generator, solver proof, difficulty model, 4-rubric critic, golden eval, daily ship, live retire | `agents/level_factory/` | critic κ 0.92 on the 100-level golden set; 3 daily levels shipped into the game |

`make phase2` resets only Phase 2 state, runs both pipelines against `data/telemetry.duckdb`, and
writes [docs/phase2_scorecard.md](docs/phase2_scorecard.md). Stop the server first, because DuckDB
allows one writer per file.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the diagram, the block-by-block
failure-mitigation matrix and the measured results.

## Configuration

| Variable | Effect |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Use a real model, with a circuit-breaker fallback to the mock |
| `LLM_MODEL` | Override the model id |
| `LIVEOPS_DATA_DIR` | Put the databases somewhere else (the tests use temp dirs) |
| `HARNESS_MAX_RUN_USD` | Hard per-run spend cap |
| `HARNESS_MAX_RUN_TOKENS` | Hard per-run token cap |
| `data/cookie_cats.csv` | When present, the seeder loads the real Kaggle file instead of the calibrated synthetic copy |

DuckDB allows one writer per file. While the server is running, reseed through it with
`curl -X POST localhost:8000/api/admin/reseed`. Running `make seed` at the same time fails
with exit code 2 and a message naming the process that holds the lock.
