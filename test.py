import pytest

from src.evaluator import evaluate_results, source_iou
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def test_minimal_source_rejects_invalid_offsets() -> None:
    with pytest.raises(ValueError):
        MinimalSource(
            file_path="file.py",
            first_character_index=10,
            last_character_index=10,
        )


def test_source_iou_requires_same_file() -> None:
    source_a = MinimalSource(
        file_path="a.py",
        first_character_index=0,
        last_character_index=100,
    )
    source_b = MinimalSource(
        file_path="b.py",
        first_character_index=0,
        last_character_index=100,
    )

    assert source_iou(source_a, source_b) == 0.0


def test_evaluate_results_counts_found_sources() -> None:
    expected_source = MinimalSource(
        file_path="src/example.py",
        first_character_index=0,
        last_character_index=100,
    )
    dataset = RagDataset(rag_questions=[
        AnsweredQuestion(
            question_id="q1",
            question="Where is the scheduler?",
            sources=[expected_source],
            answer="In src/example.py.",
        )
    ])
    student_results = StudentSearchResults(
        k=1,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question_str="Where is the scheduler?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="src/example.py",
                        first_character_index=10,
                        last_character_index=90,
                    )
                ],
            )
        ],
    )

    metrics = evaluate_results(student_results, dataset, k=1)

    assert metrics["expected_sources"] == 1
    assert metrics["found_sources"] == 1
    assert metrics["recall"] == 1.0
