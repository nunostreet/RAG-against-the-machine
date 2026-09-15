from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, model_validator


class MinimalSource(BaseModel):
    """A pointer to a slice of a file, identified by character offsets."""

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "MinimalSource":
        """Ensure the source slice is not empty or inverted."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                "last_character_index must be greater than "
                "first_character_index"
            )
        return self


class UnansweredQuestion(BaseModel):
    """A question from the dataset that has not yet been answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """An answered question with its supporting sources."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """The input dataset — a mix of answered and unanswered questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """BM25 retrieval results for a single question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results extended with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Full retrieval output for all questions, including the k used."""

    search_results: list[MinimalSearchResults]
    k: int = Field(gt=0)


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Full output including generated answers for all questions."""

    search_results: list[MinimalAnswer]  # type: ignore[assignment]
    k: int
