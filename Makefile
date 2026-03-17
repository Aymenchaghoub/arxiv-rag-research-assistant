.PHONY: setup dev docker-up docker-down test lint ingest benchmark

setup:
\tuv sync

dev:
\tuv run uvicorn app.api.main:app --reload

docker-up:
\tdocker-compose up --build

docker-down:
\tdocker-compose down -v

test:
\tuv run pytest tests/ -v

lint:
\tuv run ruff check . && uv run ruff format --check .

ingest:
\tcurl -X POST localhost:8001/api/v1/ingest

benchmark:
\tpython -m app.evaluation.run_benchmark

