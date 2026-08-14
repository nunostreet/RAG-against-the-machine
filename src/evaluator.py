"""Local retrieval evaluator.

The official project evaluator is external, but this module gives quick local
feedback while developing. It compares retrieved `MinimalSource` objects with
the ground-truth sources in answered datasets.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def _load_json(path: str) -> Any:
    """Load a JSON file without applying a schema yet."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_student_results(path: str) -> StudentSearchResults:
    """Load and validate a student search result file."""
    return StudentSearchResults.model_validate(_load_json(path))


def load_dataset(path: str) -> RagDataset:
    """Load and validate a RAG dataset file."""
    return RagDataset.model_validate(_load_json(path))


def _effective_source(
    source: MinimalSource, max_context_length: int | None
) -> MinimalSource:
    """Apply the evaluator context cap to a retrieved source.

    The moulinette accepts `max_context_length`; this helper approximates that
    behavior by limiting how much of a retrieved chunk can count as context.
    """
    if max_context_length is None:
        return source
    last_index = min(
        source.last_character_index,
        source.first_character_index + max_context_length,
    )
    return MinimalSource(
        file_path=source.file_path,
        first_character_index=source.first_character_index,
        last_character_index=last_index,
    )


def source_iou(
    retrieved: MinimalSource,
    expected: MinimalSource,
    max_context_length: int | None = None,
) -> float:
    """Compute interval IoU for two sources in the same file.

    IoU is intersection length divided by union length. Sources in different
    files never match, even if their numeric offsets overlap.
    """
    retrieved = _effective_source(retrieved, max_context_length)
    if retrieved.file_path != expected.file_path:
        return 0.0

    overlap_start = max(
        retrieved.first_character_index,
        expected.first_character_index,
    )
    overlap_end = min(
        retrieved.last_character_index,
        expected.last_character_index,
    )
    intersection = max(0, overlap_end - overlap_start)
    if intersection == 0:
        return 0.0

    union_start = min(
        retrieved.first_character_index,
        expected.first_character_index,
    )
    union_end = max(
        retrieved.last_character_index,
        expected.last_character_index,
    )
    return intersection / (union_end - union_start)


def source_is_found(
    retrieved_sources: Iterable[MinimalSource],
    expected_source: MinimalSource,
    threshold: float,
    max_context_length: int | None,
) -> bool:
    """Return true if any retrieved source overlaps enough with expected."""
    return any(
        source_iou(retrieved, expected_source, max_context_length) >= threshold
        for retrieved in retrieved_sources
    )


def evaluate_results(
    student_results: StudentSearchResults,
    dataset: RagDataset,
    k: int,
    max_context_length: int | None = 2000,
    threshold: float = 0.05,
) -> dict[str, float | int]:
    """Evaluate retrieval recall against answered questions in the dataset.

    Recall is computed at source level: each expected source counts as found if
    at least one of the top-k retrieved sources has IoU above `threshold`.
    Unanswered questions are ignored because they do not contain ground-truth
    sources.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be in the interval (0, 1]")

    results_by_id = {
        result.question_id: result
        for result in student_results.search_results
    }
    answered_questions = [
        question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion)
    ]

    total_sources = 0
    found_sources = 0
    answered_with_result = 0

    for question in answered_questions:
        result = results_by_id.get(question.question_id)
        if result is None:
            continue

        answered_with_result += 1
        retrieved_sources = result.retrieved_sources[:k]
        for expected_source in question.sources:
            total_sources += 1
            if source_is_found(
                retrieved_sources,
                expected_source,
                threshold,
                max_context_length,
            ):
                found_sources += 1

    recall = found_sources / total_sources if total_sources else 0.0
    return {
        "k": k,
        "threshold": threshold,
        "max_context_length": max_context_length or 0,
        "answered_questions": len(answered_questions),
        "answered_questions_with_result": answered_with_result,
        "expected_sources": total_sources,
        "found_sources": found_sources,
        "recall": recall,
    }


def evaluate_files(
    student_results_path: str,
    dataset_path: str,
    k: int,
    max_context_length: int | None = 2000,
    threshold: float = 0.05,
) -> dict[str, float | int]:
    """Load files and evaluate retrieval recall.

    This wrapper is used by the CLI so invalid JSON or invalid Pydantic schemas
    become a single user-facing `ValueError`.
    """
    try:
        student_results = load_student_results(student_results_path)
        dataset = load_dataset(dataset_path)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid evaluation input: {exc}") from exc

    return evaluate_results(
        student_results,
        dataset,
        k=k,
        max_context_length=max_context_length,
        threshold=threshold,
    )
