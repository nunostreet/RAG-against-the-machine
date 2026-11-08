"""Command-line interface

This module keeps all user-facing commands in one class because `python-fire`
turns public methods into CLI subcommands. Helper functions stay private.
Subject requirements for CLI: index, search, search_dataset, answer,
answer_dataset, and evaluate.
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
DEFAULT_ANSWER_OUTPUT_DIR = "data/output/search_results_and_answer"


def _die(message: str) -> NoReturn:
    """Print a user-facing error and stop without a Python traceback."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def _load_index_and_chunks(
    index_path: str = INDEX_PATH,
    chunks_path: str = CHUNKS_PATH,
) -> tuple[list[MinimalSource], bm25s.BM25]:
    """Load the persisted retrieval artifacts.

    The BM25 index and the chunk metadata are stored separately: bm25s stores
    only internal corpus positions, while `chunks.json` maps those positions
    back to file paths and character offsets required by the output schema.
    """
    if not os.path.exists(chunks_path):
        _die(
            f"chunks file not found: {chunks_path}. "
            "Run 'uv run python -m src index' first."
        )
    chunks = load_chunks(chunks_path)
    try:
        index = retriever.load_index(index_path)
    except FileNotFoundError as exc:
        _die(str(exc))
    if not chunks or index.scores["num_docs"] != len(chunks):
        _die("index and chunks do not match. Rebuild the index.")
    for chunk in chunks:
        if (chunk.last_character_index - chunk.first_character_index
                > MAX_ALLOWED_CHUNK_SIZE):
            _die("index contains sources longer than 2000 characters. "
                 "Rebuild the index.")
    return chunks, index


def _load_dataset(dataset_path: str) -> RagDataset:
    """Load and validate a dataset JSON file against the Pydantic schema."""
    if not os.path.exists(dataset_path):
        _die(f"dataset file not found: {dataset_path}")
    try:
        with open(dataset_path, encoding="utf-8") as f:
            dataset = RagDataset.model_validate(json.load(f))
    except (json.JSONDecodeError, ValidationError) as exc:
        _die(f"invalid dataset file: {exc}")
    for question in dataset.rag_questions:
        _validate_query(question.question)
    return dataset


def _load_student_search_results(path: str) -> StudentSearchResults:
    """Load and validate retrieval output before answer generation."""
    if not os.path.exists(path):
        _die(f"student search results file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            results = StudentSearchResults.model_validate(json.load(f))
    except (json.JSONDecodeError, ValidationError) as exc:
        _die(f"invalid student search results file: {exc}")
    for question in results.search_results:
        _validate_query(question.question)
    return results


def _validate_query(query: str) -> None:
    """Reject empty and non-text questions before loading artifacts."""
    if not isinstance(query, str) or not query.strip():
        _die("query must be a non-empty string.")


def _validate_chunk_size(max_chunk_size: int) -> None:
    """Validate that max_chunk_size is within allowed bounds."""
    if (type(max_chunk_size) is not int
            or not 1 <= max_chunk_size <= MAX_ALLOWED_CHUNK_SIZE):
        _die(
            f"max_chunk_size must be between 1 and "
            f"{MAX_ALLOWED_CHUNK_SIZE}, got {max_chunk_size}."
        )


def _validate_k(k: int) -> None:
    """Validate that k is a positive integer."""
    if type(k) is not int or k <= 0:
        _die(f"k must be a positive integer, got {k}.")


def _validate_answer_sources(results: StudentSearchResults, k: int) -> None:
    """Check selected source files and offsets before loading the model."""
    file_lengths: dict[str, int] = {}
    for question in results.search_results:
        for source in question.retrieved_sources[:k]:
            if source.file_path not in file_lengths:
                with open(source.file_path, encoding="utf-8",
                          errors="ignore") as f:
                    file_lengths[source.file_path] = len(f.read())
            if source.last_character_index > file_lengths[source.file_path]:
                _die(f"source offsets exceed file length: {source.file_path}")
            if (source.last_character_index - source.first_character_index
                    > MAX_ALLOWED_CHUNK_SIZE):
                _die(f"source exceeds 2000 characters: {source.file_path}")


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
            max_chunk_size: int = 2000,
            raw_dir: str = RAW_DIR,
            index_path: str = INDEX_PATH,
            chunks_path: str = CHUNKS_PATH,
    ) -> None:
        """Chunk the repository and build the BM25 index, saving both to disk.

        The command reads indexable files from `data/raw`, creates
        character-offset chunks, writes those chunks to `chunks.json`, and
        saves the BM25 index under `data/processed/bm25_index`.

        Args:
            max_chunk_size: Maximum characters per chunk (max 2000).
            raw_dir: Directory containing the corpus.
            index_path: Directory for the persisted BM25 index.
            chunks_path: File for chunk metadata.
        """
        _validate_chunk_size(max_chunk_size)

        if not os.path.isdir(raw_dir):
            _die(f"raw data directory not found: {raw_dir}")

        print("Chunking repository...")
        chunks = build_chunks(raw_dir, max_chunk_size=max_chunk_size)
        if not chunks:
            _die(f"no indexable files found in {raw_dir}.")

        print("Building BM25 index...")
        corpus = build_corpus(chunks)
        build_index(corpus, index_path)
        save_chunks(chunks, chunks_path)
        print(f"Saved {len(chunks)} chunks to {chunks_path}")
        print(f"Index saved to {index_path}")

    def search_dataset(
            self,
            dataset_path: str,
            save_directory: str = DEFAULT_SEARCH_OUTPUT_DIR,
            output_path: str | None = None,
            k: int = 10,
            index_path: str = INDEX_PATH,
            chunks_path: str = CHUNKS_PATH,
    ) -> None:
        """Run BM25 retrieval for every question in a dataset.

        The output follows the `StudentSearchResults` schema expected by the
        subject: one result per question and exactly `k` as metadata.

        Args:
            dataset_path: Path to the input JSON dataset file.
            save_directory: Directory where the output JSON will be written.
            output_path: Optional exact output path, kept for compatibility.
            k: Number of chunks to retrieve per question.
            index_path: Directory containing the BM25 index.
            chunks_path: File containing chunk metadata.
        """
        _validate_k(k)
        data = _load_dataset(dataset_path)
        chunks, index = _load_index_and_chunks(index_path, chunks_path)
        output_path = _resolve_output_path(
            output_path, save_directory, dataset_path
        )

        search_result: list[MinimalSearchResults] = []
        for question in tqdm(data.rag_questions, desc="Searching"):
            results = retriever.search(question.question, index, chunks, k=k)
            search_result.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
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
            k: int | None = None,
            index_path: str = INDEX_PATH,
            chunks_path: str = CHUNKS_PATH,
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
            k: Chunks per answer; defaults to the input file's k, or 10
                when retrieving directly from a dataset.
            index_path: Directory containing the BM25 index (dataset mode).
            chunks_path: File containing chunk metadata (dataset mode).
        """
        if k is not None:
            _validate_k(k)
        if (student_search_results_path is not None
                and dataset_path is not None):
            _die("provide either student_search_results_path or dataset_path")

        if student_search_results_path is not None:
            search_results = _load_student_search_results(
                student_search_results_path
            )
            input_path = student_search_results_path
            if k is None:
                k = search_results.k
        elif dataset_path is not None:
            if k is None:
                k = 10
            data = _load_dataset(dataset_path)
            chunks, index = _load_index_and_chunks(index_path, chunks_path)
            generated_results: list[MinimalSearchResults] = []
            for question in tqdm(data.rag_questions, desc="Searching"):
                generated_results.append(MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
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
        _validate_answer_sources(search_results, k)
        answers: list[MinimalAnswer] = []
        if search_results.search_results:
            tokenizer, model = load_model()
            for question in tqdm(search_results.search_results,
                                 desc="Answering"):
                retrieved_sources = question.retrieved_sources[:k]
                answers.append(MinimalAnswer(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=retrieved_sources,
                    answer=generate(
                        question.question,
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
            save_directory: str = DEFAULT_SEARCH_OUTPUT_DIR,
            output_path: str | None = None,
            index_path: str = INDEX_PATH,
            chunks_path: str = CHUNKS_PATH,
    ) -> None:
        """Search one query and write and print StudentSearchResults JSON.

        Args:
            query: The search query string.
            k: Number of chunks to retrieve.
            save_directory: Directory for query.json unless output_path is set.
            output_path: Optional exact output file path.
            index_path: Directory containing the BM25 index.
            chunks_path: File containing chunk metadata.
        """
        _validate_query(query)
        _validate_k(k)
        chunks, index = _load_index_and_chunks(index_path, chunks_path)
        results = retriever.search(query, index, chunks, k=k)
        output = StudentSearchResults(search_results=[MinimalSearchResults(
            question_id=str(uuid.uuid4()), question=query,
            retrieved_sources=results,
        )], k=k)
        _write_json_model(output, _resolve_output_path(
            output_path, save_directory, "query.json"
        ))
        print(output.model_dump_json(indent=2))

    def answer(
            self,
            query: str,
            k: int = 10,
            save_directory: str = DEFAULT_ANSWER_OUTPUT_DIR,
            output_path: str | None = None,
            index_path: str = INDEX_PATH,
            chunks_path: str = CHUNKS_PATH,
    ) -> None:
        """Answer a single query using retrieved context from the index.

        The command writes and prints `StudentSearchResultsAndAnswer` JSON.

        Args:
            query: The question to answer.
            k: Number of chunks to retrieve.
            save_directory: Directory for query.json unless output_path is set.
            output_path: Optional exact output file path.
            index_path: Directory containing the BM25 index.
            chunks_path: File containing chunk metadata.
        """
        _validate_query(query)
        _validate_k(k)
        chunks, index = _load_index_and_chunks(index_path, chunks_path)
        results = retriever.search(query, index, chunks, k=k)
        search_results = StudentSearchResults(search_results=[
            MinimalSearchResults(question_id=str(uuid.uuid4()),
                                 question=query, retrieved_sources=results),
        ], k=k)
        _validate_answer_sources(search_results, k)
        tokenizer, model = load_model()
        response = generate(query, results, tokenizer, model)
        answer = MinimalAnswer(
            question_id=search_results.search_results[0].question_id,
            question=query,
            retrieved_sources=results,
            answer=response,
        )
        output = StudentSearchResultsAndAnswer(search_results=[answer], k=k)
        _write_json_model(output, _resolve_output_path(
            output_path, save_directory, "query.json"
        ))
        print(output.model_dump_json(indent=2))

    def evaluate(
            self,
            student_search_results_path: str | None = None,
            dataset_path: str | None = None,
            k: int = 10,
            max_context_length: int = 2000,
            threshold: float = 0.05,
            student_results_path: str | None = None,
            output_path: str | None = None,
    ) -> None:
        """Evaluate retrieval output against an answered dataset.

        This is a local development metric. It compares retrieved sources with
        expected sources using interval overlap in the same file. The official
        moulinette should still be used for final grading.
        """
        _validate_k(k)
        if (student_search_results_path is not None
                and student_results_path is not None):
            _die("provide only one search results path")
        results_path = student_search_results_path or student_results_path
        if results_path is None or dataset_path is None:
            _die("student_search_results_path and dataset_path are required")
        try:
            metrics = evaluate_files(
                student_results_path=results_path,
                dataset_path=dataset_path,
                k=k,
                max_context_length=max_context_length,
                threshold=threshold,
            )
        except ValueError as exc:
            _die(str(exc))

        text = json.dumps(metrics, indent=2)
        if output_path is not None:
            directory = os.path.dirname(output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        print(text)
