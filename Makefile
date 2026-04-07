.PHONY: setup dev docker-up docker-down test lint ingest benchmark

setup:
	uv sync

dev:
	uv run uvicorn app.api.main:app --reload

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check . && uv run ruff format --check .

ingest:
	curl -X POST localhost:8001/api/v1/ingest

benchmark:
	uv run python -m app.evaluation.run_benchmark

