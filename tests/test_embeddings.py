from __future__ import annotations

from app.embeddings.embedding_factory import EmbeddingFactory


def test_embedding_factory_caching_and_dim(monkeypatch):
    class MockHuggingFaceEmbeddings:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 384

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 384 for _ in texts]

    monkeypatch.setattr(
        "app.embeddings.embedding_factory.HuggingFaceEmbeddings", MockHuggingFaceEmbeddings
    )

    emb1 = EmbeddingFactory.get("minilm")
    assert emb1 is not None

    emb2 = EmbeddingFactory.get("minilm")
    assert emb1 is emb2

    # Test dimension directly by inferring correctly
    vectors = emb1.embed_documents(["test"])
    assert len(vectors[0]) == 384
