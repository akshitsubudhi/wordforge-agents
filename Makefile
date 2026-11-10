# LiveOps Agent Platform
# `make setup run` takes a fresh clone to a playable game. No Docker, no external services.
PY      ?= .venv/bin/python
PORT    ?= 8000
UV      := $(shell command -v uv 2>/dev/null)

.PHONY: help fresh-traces setup seed reseed run stop conformance redteam metrics test check clean wordlist phase2 copilot-scan phase3 ad-economy intake-eval scorecard faults

help:
	@echo "make setup        create .venv and install deps (uv if present, else venv+pip)"
	@echo "make run          start the server on :$(PORT) (self-seeds an empty DB) -> http://localhost:$(PORT)/game"
	@echo "make seed         seed data/telemetry.duckdb (server must be stopped)"
	@echo "make reseed       wipe + reseed (server stopped) ; with server up: curl -X POST :$(PORT)/api/admin/reseed"
	@echo "make conformance  guardrail boundary & canary conformance suite + 5-way ablation"
	@echo "make metrics      p50/p95/p99 latency, cost and success rate from data/traces.duckdb"
	@echo "make test         pytest"
	@echo "make check        test + conformance (the CI gate)"
	@echo "make phase2       run Copilot + Level Factory end to end, print the scorecard (server must be stopped)"
	@echo "make copilot-scan funnel anomaly scan only (read-only workflow)"
	@echo "make phase3       ad economy scorecard + insurance intake 50-submission eval (server must be stopped)"
	@echo "make scorecard    re-run every eval and rebuild docs/scorecard.json + SCORECARD.md (~2 min, server stopped)"
	@echo "make faults       run the 4 production fault drills from the CLI (server stopped)"
	@echo "open http://localhost:8000 after make run -> Evaluator Control Center"

# Published numbers are generated with the deterministic mock so they reproduce bit-for-bit even on a machine
# that has an API key exported. Run e.g. `LLM_PROVIDER=auto make phase2` to exercise the real provider instead.
scorecard phase2 phase3 ad-economy intake-eval faults conformance test check: export LLM_PROVIDER ?= mock

setup:
ifdef UV
	uv venv -q .venv
	uv pip install -q --python $(PY) -r requirements.txt
else
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt
endif
	@echo "ready: make run"

seed:
	$(PY) scripts/seed_telemetry.py

reseed:
	$(PY) scripts/seed_telemetry.py --force

run:
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

stop:
	-pkill -f "uvicorn app.main:app" && echo "server stopped"

conformance:
	$(PY) -m core.harness.run_redteam_eval --json data/reports/conformance.json

redteam: conformance

metrics:
	$(PY) -c "from core.harness.telemetry import agent_metrics; from core.config import get_settings; \
	[print(r) for r in agent_metrics(get_settings().traces_db)]"

test:
	$(PY) -m pytest

phase2:
	$(PY) scripts/verify_phase2.py --out docs/phase2_scorecard.md

copilot-scan:
	$(PY) -m agents.liveops_copilot.agent scan

ad-economy:
	$(PY) scripts/run_ad_economy_sim.py --out docs/phase3_ad_economy_scorecard.md

intake-eval:
	$(PY) scripts/run_intake_eval.py --out docs/phase3_intake_eval_scorecard.md

phase3: ad-economy intake-eval

# traces.duckdb is wiped first: the latency / cost-per-task rows describe THIS run, not an accumulation of old ones
fresh-traces:
	rm -f data/traces.duckdb data/traces.duckdb.wal data/traces.spool.jsonl

scorecard: fresh-traces phase2 phase3
	$(PY) -m core.harness.run_redteam_eval --json data/reports/conformance.json
	$(PY) scripts/build_scorecard.py

faults:
	$(PY) scripts/run_fault_drills.py

check: test conformance

clean:
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# Dev-only: regenerate the committed word list (needs wordfreq + a Hunspell en_US.dic)
wordlist:
	uv pip install -q --python $(PY) wordfreq
	curl -sSfL -o .en_US.dic https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_US.dic
	$(PY) scripts/build_wordlist.py --dic .en_US.dic --out data/wordlist.tsv
