# arxiv-rag-research-assistant

Retrieval-Augmented Generation (RAG) assistant for ArXiv papers, with ingestion, chunking, retrieval, chat, and evaluation workflows.

## What This Project Includes

- FastAPI backend for ingestion and question answering
- Streamlit frontend for chat and evaluation
- Multiple chunking strategies for retrieval benchmarking
- Dense, sparse, and hybrid retrieval components
- RAGAS-based evaluation utilities
- Docker Compose stack (`chroma`, `api`, `frontend`)

## Repository Layout

- `app/`: backend modules (API, ingestion, retrieval, generation, evaluation)
- `frontend/`: Streamlit UI
- `tests/`: unit and API tests
- `notebooks/`: exploration and analysis notebooks
- `data/`: raw, processed, and evaluation data
- `results/`: benchmark outputs

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run The API Locally

```bash
uv run uvicorn app.api.main:app --reload --port 8001
```

### 3. Run The Streamlit UI

```bash
uv run streamlit run frontend/streamlit_app.py
```

### 4. Run With Docker Compose

```bash
docker compose up --build
```

## Main API Endpoints

- `POST /api/v1/ingest`: fetch an ArXiv paper, parse, chunk, and index it
- `POST /api/v1/chat`: answer a question with retrieved source chunks
- `GET /api/v1/collections`: list available Chroma collections
- `DELETE /api/v1/collection/{name}`: delete a collection

## Benchmarking And Analysis

- Notebooks:
	- `notebooks/01_data_exploration.ipynb`
	- `notebooks/02_chunking_benchmark.ipynb`
	- `notebooks/03_ragas_results_viz.ipynb`
- Benchmark results file:
	- `results/benchmark_results.csv`

## RAGAS Metrics (One Line Each)

- `faithfulness`: measures whether the generated answer is supported by the retrieved context.
- `answer_relevancy`: measures how directly the answer addresses the user question.
- `context_recall`: measures whether the retrieved context contains the information needed for the ground truth answer.
- `context_precision`: measures how much of the retrieved context is actually relevant to answering the question.

## Known Limitations

- End-to-end quality depends on external services and model availability (Hugging Face inference and embedding downloads).
- Chunk quality and retrieval quality are sensitive to strategy and hyperparameters (`chunk_size`, overlap, `k`).
- Sentence chunking depends on local NLTK resources and may require first-run tokenizer downloads.
- RAGAS evaluation can be slow and expensive for larger benchmark grids.
- Benchmark quality depends on the quality and size of `data/eval/test_questions.json`.

## Development Commands

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
make benchmark
```

