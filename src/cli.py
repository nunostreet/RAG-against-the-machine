"""Command-line interface for the RAG deliverable.

This module keeps all user-facing commands in one class because `python-fire`
turns public methods into CLI subcommands. Helper functions stay private so the
CLI surface remains close to the subject requirements: index, search,
search_dataset, answer, answer_dataset, and evaluate.
"""

import json
import os
import sys
import uuid
from typing import NoReturn

import bm25s
import src.retriever as retriever
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src.chunker import build_chunks
from src.evaluator import evaluate_files
from src.generator import generate, load_model
from src.indexer import build_corpus, build_index, load_chunks, save_chunks
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

INDEX_PATH = "data/processed/bm25_index"
CHUNKS_PATH = "data/processed/chunks.json"
RAW_DIR = "data/raw"
MAX_ALLOWED_CHUNK_SIZE = 2000
DEFAULT_SEARCH_OUTPUT_DIR = "data/output/search_results"
DEFAULT_ANSWER_OUTPUT_DIR = "data/output/answers"


def _die(message: str) -> NoReturn:
    """Print a user-facing error and stop without a Python traceback."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def _load_index_and_chunks() -> tuple[list[MinimalSource], bm25s.BM25]:
    """Load the persisted retrieval artifacts.

    The BM25 index and the chunk metadata are stored separately: bm25s stores
    only internal corpus positions, while `chunks.json` maps those positions
    back to file paths and character offsets required by the output schema.
    """
    if not os.path.exists(CHUNKS_PATH):
        _die(
            "chunks file not found. Run 'uv run python -m src index' first."
        )
    chunks = load_chunks(CHUNKS_PATH)
    try:
        index = retriever.load_index(INDEX_PATH)
    except FileNotFoundError as exc:
        _die(str(exc))
    return chunks, index


def _load_dataset(dataset_path: str) -> RagDataset:
    """Load and validate a dataset JSON file against the Pydantic schema."""
    if not os.path.exists(dataset_path):
        _die(f"dataset file not found: {dataset_path}")
    try:
        with open(dataset_path, encoding="utf-8") as f:
            return RagDataset.model_validate(json.load(f))
    except (json.JSONDecodeError, ValidationError) as exc:
        _die(f"invalid dataset file: {exc}")


def _load_student_search_results(path: str) -> StudentSearchResults:
    """Load and validate retrieval output before answer generation."""
    if not os.path.exists(path):
        _die(f"student search results file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return StudentSearchResults.model_validate(json.load(f))
    except (json.JSONDecodeError, ValidationError) as exc:
        _die(f"invalid student search results file: {exc}")


def _validate_chunk_size(max_chunk_size: int) -> None:
    """Validate that max_chunk_size is within allowed bounds."""
    if max_chunk_size <= 0 or max_chunk_size > MAX_ALLOWED_CHUNK_SIZE:
        _die(
            f"max_chunk_size must be between 1 and "
            f"{MAX_ALLOWED_CHUNK_SIZE}, got {max_chunk_size}."
        )


def _validate_k(k: int) -> None:
    """Validate that k is a positive integer."""
    if k <= 0:
        _die(f"k must be a positive integer, got {k}.")


def _resolve_output_path(
    output_path: str | None,
    save_directory: str,
    input_path: str,
) -> str:
    """Return an exact output path.

    The subject asks for `save_directory`, but earlier local code used
    `output_path`. Supporting both keeps the CLI compatible while making the
    documented workflow match the subject.
    """
    if output_path:
        return output_path
    return os.path.join(save_directory, os.path.basename(input_path))


def _write_json_model(output: BaseModel, output_path: str) -> None:
    """Serialize a Pydantic model to pretty JSON, creating directories."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))


class RAGSystem():
    """Public CLI commands exposed by `python -m src`.

    Fire maps each method below to a subcommand. For example:
    `uv run python -m src index --max_chunk_size 1000`.
    """

    def index(
            self,
            max_chunk_size: int = 1000,
    ) -> None:
        """Chunk the repository and build the BM25 index, saving both to disk.

        The command reads indexable files from `data/raw`, creates
        character-offset chunks, writes those chunks to `chunks.json`, and
        saves the BM25 index under `data/processed/bm25_index`.

        Args:
            max_chunk_size: Maximum characters per chunk (max 2000).
        """
        _validate_chunk_size(max_chunk_size)

        if not os.path.exists(RAW_DIR):
            _die(f"raw data directory not found: {RAW_DIR}")

        print("Chunking repository...")
        chunks = build_chunks(RAW_DIR, max_chunk_size=max_chunk_size)
        if not chunks:
            _die(f"no indexable files found in {RAW_DIR}.")
        save_chunks(chunks, CHUNKS_PATH)
        print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")

        print("Building BM25 index...")
        corpus = build_corpus(chunks)
        build_index(corpus, INDEX_PATH)
        print(f"Index saved to {INDEX_PATH}")

    def search_dataset(
            self,
            dataset_path: str,
            save_directory: str = DEFAULT_SEARCH_OUTPUT_DIR,
            output_path: str | None = None,
            k: int = 10,
    ) -> None:
        """Run BM25 retrieval for every question in a dataset.

        The output follows the `StudentSearchResults` schema expected by the
        subject: one result per question and exactly `k` as metadata.

        Args:
            dataset_path: Path to the input JSON dataset file.
            save_directory: Directory where the output JSON will be written.
            output_path: Optional exact output path, kept for compatibility.
            k: Number of chunks to retrieve per question.
        """
        _validate_k(k)
        chunks, index = _load_index_and_chunks()
        data = _load_dataset(dataset_path)
        output_path = _resolve_output_path(
            output_path, save_directory, dataset_path
        )

        search_result: list[MinimalSearchResults] = []
        for question in tqdm(data.rag_questions, desc="Searching"):
            results = retriever.search(question.question, index, chunks, k=k)
            search_result.append(MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=results,
            ))

        output = StudentSearchResults(search_results=search_result, k=k)
        _write_json_model(output, output_path)
        print(f"Results saved to {output_path}")

    def answer_dataset(
            self,
            student_search_results_path: str | None = None,
            save_directory: str = DEFAULT_ANSWER_OUTPUT_DIR,
            output_path: str | None = None,
            dataset_path: str | None = None,
            k: int = 10,
    ) -> None:
        """Generate answers from a StudentSearchResults file.

        Preferred mode is two-step: run `search_dataset` first, then pass its
        JSON here through `student_search_results_path`. A legacy one-step mode
        is also supported with `dataset_path`, which retrieves and answers in
        one command.

        Args:
            student_search_results_path: JSON created by search_dataset.
            save_directory: Directory where the output JSON will be written.
            output_path: Optional exact output path, kept for compatibility.
            dataset_path: Optional dataset path for legacy retrieve+answer
                mode.
            k: Number of chunks to use per answer.
        """
        _validate_k(k)
        tokenizer, model = load_model()

        if student_search_results_path is not None:
            search_results = _load_student_search_results(
                student_search_results_path
            )
            input_path = student_search_results_path
        elif dataset_path is not None:
            chunks, index = _load_index_and_chunks()
            data = _load_dataset(dataset_path)
            generated_results: list[MinimalSearchResults] = []
            for question in tqdm(data.rag_questions, desc="Searching"):
                generated_results.append(MinimalSearchResults(
                    question_id=question.question_id,
                    question_str=question.question,
                    retrieved_sources=retriever.search(
                        question.question, index, chunks, k=k
                    ),
                ))
            search_results = StudentSearchResults(
                search_results=generated_results, k=k
            )
            input_path = dataset_path
        else:
            _die(
                "provide student_search_results_path, or dataset_path for "
                "legacy retrieve+answer mode."
            )

        output_path = _resolve_output_path(
            output_path, save_directory, input_path
        )
        answers: list[MinimalAnswer] = []
        for question in tqdm(search_results.search_results, desc="Answering"):
            retrieved_sources = question.retrieved_sources[:k]
            answers.append(MinimalAnswer(
                question_id=question.question_id,
                question_str=question.question_str,
                retrieved_sources=retrieved_sources,
                answer=generate(
                    question.question_str,
                    retrieved_sources,
                    tokenizer,
                    model,
                ),
            ))

        output = StudentSearchResultsAndAnswer(search_results=answers, k=k)
        _write_json_model(output, output_path)
        print(f"Results saved to {output_path}")

    def search(
            self,
            query: str,
            k: int = 10,
    ) -> None:
        """Search the index with a single query and print the top-k results.

        This command is mainly for debugging retrieval quality by inspecting
        which file slices BM25 selects before generation.

        Args:
            query: The search query string.
            k: Number of chunks to retrieve.
        """
        if not query or not query.strip():
            _die("query must not be empty.")
        _validate_k(k)
        chunks, index = _load_index_and_chunks()
        results = retriever.search(query, index, chunks, k=k)
        for i, chunk in enumerate(results):
            print(f"[{i + 1}] {chunk.file_path} "
                  f"({chunk.first_character_index}-"
                  f"{chunk.last_character_index})")

    def answer(
            self,
            query: str,
            k: int = 10,
    ) -> None:
        """Answer a single query using retrieved context from the index.

        The command returns a `MinimalAnswer` JSON object so the response keeps
        the answer and the exact retrieved sources together.

        Args:
            query: The question to answer.
            k: Number of chunks to retrieve.
        """
        if not query or not query.strip():
            _die("query must not be empty.")
        _validate_k(k)
        chunks, index = _load_index_and_chunks()
        tokenizer, model = load_model()
        results = retriever.search(query, index, chunks, k=k)
        response = generate(query, results, tokenizer, model)
        output = MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question_str=query,
            retrieved_sources=results,
            answer=response,
        )
        print(output.model_dump_json(indent=2))

    def evaluate(
            self,
            student_results_path: str,
            dataset_path: str,
            k: int = 10,
            max_context_length: int = 2000,
            threshold: float = 0.05,
    ) -> None:
        """Evaluate retrieval output against an answered dataset.

        This is a local development metric. It compares retrieved sources with
        expected sources using interval overlap in the same file. The official
        moulinette should still be used for final grading.
        """
        _validate_k(k)
        try:
            metrics = evaluate_files(
                student_results_path=student_results_path,
                dataset_path=dataset_path,
                k=k,
                max_context_length=max_context_length,
                threshold=threshold,
            )
        except ValueError as exc:
            _die(str(exc))

        print(json.dumps(metrics, indent=2))
