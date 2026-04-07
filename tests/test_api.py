from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_api_collections(monkeypatch):
    class MockCollection:
        @property
        def name(self):
            return "test_collection"

    class MockClient:
        def list_collections(self):
            return [MockCollection()]

        def delete_collection(self, name):
            return None

    monkeypatch.setattr("chromadb.HttpClient", lambda *args, **kwargs: MockClient())

    resp = client.get("/api/v1/collections")
    assert resp.status_code == 200
    assert "test_collection" in resp.json()["collections"]

    resp_del = client.delete("/api/v1/collection/test_collection")
    assert resp_del.status_code == 200


def test_api_ingest(monkeypatch):
    class MockArxivLoader:
        def __init__(self, *args, **kwargs):
            return None

        def fetch_papers(self, query, max_results):
            return [
                {"arxiv_id": "1234.5678", "pdf_path": "fake.pdf", "title": "T", "authors": ["A"]}
            ]

    class MockPDFParser:
        def parse(self, path):
            from app.ingestion.pdf_parser import PDFParseResult

            return PDFParseResult(
                full_text="test", pages=["test"], sections={}, page_count=1, token_count=1
            )

    class MockDenseRetriever:
        def __init__(self, *args, **kwargs):
            return None

        def index_documents(self, docs):
            return len(docs)

    monkeypatch.setattr("app.api.routes.ingest.ArxivLoader", MockArxivLoader)
    monkeypatch.setattr("app.api.routes.ingest.PDFParser", MockPDFParser)
    from app.embeddings.embedding_factory import EmbeddingFactory

    class FakeEmb:
        def embed_documents(self, texts):
            return [[0.1] * 384 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 384

    monkeypatch.setattr(EmbeddingFactory, "get", lambda name: FakeEmb())
    monkeypatch.setattr("app.api.routes.ingest.DenseRetriever", MockDenseRetriever)

    resp = client.post(
        "/api/v1/ingest",
        json={"arxiv_id": "1234.5678", "chunking_strategy": "sentence", "chunk_size": 200},
    )
    assert resp.status_code == 200
    assert resp.json()["document_count"] > 0


def test_api_chat(monkeypatch):
    class MockDocs:
        def __init__(self):
            self.page_content = "context"
            self.metadata = {"arxiv_id": "1234.5678"}

    class MockDenseRetriever:
        def __init__(self, *args, **kwargs):
            return None

        def retrieve(self, q, k):
            return [MockDocs()]

    class MockChain:
        def invoke(self, q):
            return "Answer"

    monkeypatch.setattr("app.api.routes.query.DenseRetriever", MockDenseRetriever)
    monkeypatch.setattr("app.api.routes.query.get_rag_chain", lambda r: MockChain())
    from app.embeddings.embedding_factory import EmbeddingFactory

    class FakeEmb:
        def __init__(self):
            return None

    monkeypatch.setattr(EmbeddingFactory, "get", lambda name: FakeEmb())

    resp = client.post(
        "/api/v1/chat",
        json={"question": "hello", "collection_name": "test", "retriever_type": "dense", "k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Answer"
    assert len(data["sources"]) == 1
