"""
================================================================================
BLOCK: Central Configuration
--------------------------------------------------------------------------------
Role & contracts
  Single source of truth for filesystem paths, DB locations, LLM provider
  selection and guard-rail limits. Every other block imports from here; nothing
  else reads os.environ directly (keeps evaluator runs reproducible).

Key engineering mechanism
  Frozen dataclass populated once from env vars with safe offline defaults
  (no API key -> deterministic mock LLM; all state under ./data).

Failure modes & 1-step mitigation
  * Missing/garbage env var -> typed parse with fallback default, never crash.
  * Tests/evals clobbering real data -> every path is overridable via env
    (LIVEOPS_DATA_DIR), tests point it at tmp dirs.
================================================================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("LIVEOPS_DATA_DIR", REPO_ROOT / "data")).resolve())
    wordlist_path: Path = REPO_ROOT / "data" / "wordlist.tsv"
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY") or None)
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY") or None)
    llm_model: str | None = field(default_factory=lambda: os.environ.get("LLM_MODEL") or None)
    max_run_usd: float = field(default_factory=lambda: _float("HARNESS_MAX_RUN_USD", 0.25))
    max_run_tokens: int = field(default_factory=lambda: int(_float("HARNESS_MAX_RUN_TOKENS", 200_000)))

    @property
    def telemetry_db(self) -> Path:
        return self.data_dir / "telemetry.duckdb"

    @property
    def traces_db(self) -> Path:
        return self.data_dir / "traces.duckdb"

    @property
    def cookie_cats_csv(self) -> Path:
        return self.data_dir / "cookie_cats.csv"


def get_settings() -> Settings:
    """Re-read env each call (cheap) so tests can monkeypatch env vars."""
    return Settings()
