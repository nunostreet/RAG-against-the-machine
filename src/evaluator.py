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


def _validate_source_length(
    source: MinimalSource, max_context_length: int | None
) -> None:
    """Reject retrieved sources exceeding the configured character limit."""
    if max_context_length is None:
        return
    if max_context_length <= 0:
        raise ValueError("max_context_length must be positive or None")
    length = source.last_character_index - source.first_character_index
    if length > max_context_length:
        raise ValueError(
            f"retrieved source '{source.file_path}' "
            f"({source.first_character_index}-{source.last_character_index}) "
            f"has {length} characters; maximum is {max_context_length}"
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
    _validate_source_length(retrieved, max_context_length)
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

    Recall is averaged over answered questions, giving each equal weight.
    A source is found when a top-k result has IoU >= `threshold` in the same
    file. Missing results and answered questions with no expected sources
    contribute zero recall. Unanswered questions are ignored. Overlong sources
    invalidate the output, including sources outside the top-k being scored.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not 0 < threshold <= 1):
        raise ValueError("threshold must be a number in the interval (0, 1]")
    if max_context_length is not None and (
            type(max_context_length) is not int or max_context_length <= 0):
        raise ValueError("max_context_length must be a positive integer")

    for search_result in student_results.search_results:
        for source in search_result.retrieved_sources:
            _validate_source_length(source, max_context_length)

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
    recall_sum = 0.0

    for question in answered_questions:
        result = results_by_id.get(question.question_id)
        total_sources += len(question.sources)
        retrieved_sources = (
            [] if result is None else result.retrieved_sources[:k]
        )
        if result is not None:
            answered_with_result += 1
        question_found = 0
        for expected_source in question.sources:
            if source_is_found(
                retrieved_sources,
                expected_source,
                threshold,
                max_context_length,
            ):
                question_found += 1
        found_sources += question_found
        if question.sources:
            recall_sum += question_found / len(question.sources)

    recall = (
        recall_sum / len(answered_questions) if answered_questions else 0.0
    )
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
    student_search_results_path: str,
    dataset_path: str,
    k: int,
    max_context_length: int | None = 2000,
    threshold: float = 0.05,
) -> dict[str, float | int]:
    """Load files and evaluate retrieval recall.

    This wrapper turns file, encoding, JSON and schema errors into a single
    user-facing `ValueError` handled by the CLI.
    """
    try:
        student_results = load_student_results(student_search_results_path)
        dataset = load_dataset(dataset_path)
    except (
        OSError, UnicodeError, json.JSONDecodeError, ValidationError
    ) as exc:
        raise ValueError(f"invalid evaluation input: {exc}") from exc

    return evaluate_results(
        student_results,
        dataset,
        k=k,
        max_context_length=max_context_length,
        threshold=threshold,
    )
