from __future__ import annotations

import uuid
from typing import List, Union

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A pointer to a slice of a file, identified by character offsets."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question from the dataset that has not yet been answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """An answered question with its supporting sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """The input dataset — a mix of answered and unanswered questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """BM25 retrieval results for a single question."""

    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results extended with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Full retrieval output for all questions, including the k used."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Full output including generated answers for all questions."""

    search_results: List[MinimalAnswer]  # type: ignore[assignment]
