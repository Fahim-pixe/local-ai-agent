PYTHON ?= python3.12
VENV ?= .venv

.PHONY: setup bootstrap test test-docker test-ollama test-retrieval lint format run health sandbox-image compose-up compose-down

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"
	cp -n .env.example .env || true

bootstrap:
	$(VENV)/bin/python scripts/bootstrap_workspace.py

test:
	$(VENV)/bin/pytest

test-docker:
	RUN_DOCKER_INTEGRATION=1 $(VENV)/bin/pytest -m docker

test-ollama:
	RUN_OLLAMA_EVALUATION=1 PYTHONPATH=src $(VENV)/bin/pytest -m ollama

test-retrieval:
	PYTHONPATH=src $(VENV)/bin/python scripts/run_retrieval_benchmark.py

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

format:
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/ruff format src tests

run:
	$(VENV)/bin/uvicorn local_ai_agent.main:app --host 127.0.0.1 --port 8000 --reload

health:
	curl --fail --silent http://127.0.0.1:8000/health

sandbox-image:
	docker build --tag local-ai-agent-sandbox:latest docker/sandbox

compose-up:
	docker compose up --build

compose-down:
	docker compose down
