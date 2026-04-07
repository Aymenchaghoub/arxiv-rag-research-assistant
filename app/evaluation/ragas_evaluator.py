"""RAGAS evaluation runner (Phase 5)."""

from __future__ import annotations

import logging
import re

import pandas as pd
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

__all__ = ["RAGASEvaluator"]

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """Evaluates RAG pipeline using RAGAS."""

    def __init__(
        self,
        llm: HuggingFaceEndpoint | None = None,
        embeddings: HuggingFaceEmbeddings | None = None,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings

    def evaluate_pipeline(
        self, questions: list[str], ground_truths: list[str], collection_name: str
    ) -> pd.DataFrame:
        """Runs evaluation and returns metrics."""
        from app.api.routes.query import chat
        from app.api.schemas import ChatRequest

        contexts = []
        answers = []

        for q in questions:
            req = ChatRequest(
                question=q, collection_name=collection_name, retriever_type="dense", k=5
            )
            try:
                resp = chat(req)
                ans = resp.answer
                ctx = [str(s.get("page_content", "")) for s in resp.sources] if resp.sources else []
            except Exception:
                logger.exception(f"Chat failed for question: {q}")
                ans = ""
                ctx = []

            answers.append(ans)
            contexts.append(ctx)

        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        dataset = Dataset.from_dict(dataset_dict)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            return result.to_pandas()
        except Exception:  # noqa: BLE001 - library compatibility issues vary by versions
            logger.exception("RAGAS evaluation failed; using fallback metrics")
            return self._fallback_metrics_df(dataset_dict=dataset_dict)

    @staticmethod
    def _fallback_metrics_df(dataset_dict: dict[str, list]) -> pd.DataFrame:
        """Compute lightweight lexical proxy metrics when RAGAS is unavailable."""

        rows: list[dict[str, float | str]] = []
        for question, answer, contexts, ground_truth in zip(
            dataset_dict["question"],
            dataset_dict["answer"],
            dataset_dict["contexts"],
            dataset_dict["ground_truth"],
            strict=False,
        ):
            context_text = "\n".join(contexts)

            faithfulness = _token_overlap_ratio(answer, context_text)
            answer_relevancy = _token_overlap_ratio(question, answer)
            context_recall = _token_overlap_ratio(ground_truth, context_text)
            context_precision = _token_overlap_ratio(context_text, ground_truth)
            avg_score = round(
                (faithfulness + answer_relevancy + context_precision + context_recall) / 4,
                4,
            )

            rows.append(
                {
                    "question": question,
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "avg_score": avg_score,
                }
            )

        return pd.DataFrame(rows)


def _token_overlap_ratio(text_a: str, text_b: str) -> float:
    """Compute overlap ratio between token sets of two strings in [0, 1]."""

    a_tokens = set(re.findall(r"[a-z0-9]+", text_a.lower()))
    b_tokens = set(re.findall(r"[a-z0-9]+", text_b.lower()))

    if not a_tokens:
        return 0.0

    overlap = len(a_tokens.intersection(b_tokens))
    return round(overlap / len(a_tokens), 4)
