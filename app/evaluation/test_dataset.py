"""Test dataset generation/loading (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["load_test_questions"]


def load_test_questions(path: str) -> list[dict[str, str]]:
    """Load evaluation QA pairs from a JSON file.

    Expected schema:
    - list[dict] where each item contains at least:
      - "question": str
      - "ground_truth": str
    """

    file_path = Path(path)
    if not file_path.exists():
        return []

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        normalized = _normalize_item(item)
        if normalized is not None:
            out.append(normalized)

    return out


def _normalize_item(item: dict[str, Any]) -> dict[str, str] | None:
    question = str(item.get("question", "")).strip()
    ground_truth = str(item.get("ground_truth", item.get("answer", ""))).strip()

    if not question or not ground_truth:
        return None

    return {"question": question, "ground_truth": ground_truth}
