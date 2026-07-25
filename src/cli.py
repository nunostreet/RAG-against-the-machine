import json
import os
import sys
import src.retriever as retriever
from tqdm import tqdm
from pydantic import ValidationError
from src.chunker import build_chunks
from src.indexer import build_corpus, build_index, save_chunks, load_chunks
from src.generator import load_model, generate
from src.models import (
    RagDataset, StudentSearchResults, MinimalSearchResults,
    MinimalAnswer, StudentSearchResultsAndAnswer,
)

INDEX_PATH = "data/processed/bm25_index"
CHUNKS_PATH = "data/processed/chunks.json"
RAW_DIR = "data/raw"
MAX_ALLOWED_CHUNK_SIZE = 2000


def _load_index_and_chunks() -> tuple:
    """Load chunks and BM25 index from disk, with clear error messages."""
    if not os.path.exists(CHUNKS_PATH):
        print(
            "Error: index not found. Run 'uv run python -m src index' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    chunks = load_chunks(CHUNKS_PATH)
    index = retriever.load_index(INDEX_PATH)
    return chunks, index


def _load_dataset(dataset_path: str) -> RagDataset:
    """Load and validate a dataset JSON file."""
    if not os.path.exists(dataset_path):
        print(
            f"Error: dataset file not found: {dataset_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(dataset_path) as f:
            return RagDataset.model_validate(json.load(f))
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error: invalid dataset file: {e}", file=sys.stderr)
        sys.exit(1)


def _validate_chunk_size(max_chunk_size: int) -> None:
    """Validate that max_chunk_size is within allowed bounds."""
    if max_chunk_size <= 0 or max_chunk_size > MAX_ALLOWED_CHUNK_SIZE:
        print(
            f"Error: max_chunk_size must be between 1 and "
            f"{MAX_ALLOWED_CHUNK_SIZE}, got {max_chunk_size}.",
            file=sys.stderr,
        )
        sys.exit(1)


def _validate_k(k: int) -> None:
    """Validate that k is a positive integer."""
    if k <= 0:
        print(
            f"Error: k must be a positive integer, got {k}.",
            file=sys.stderr,
        )
        sys.exit(1)


class RAGSystem():
    """CLI entry point for the RAG system, exposed via python-fire."""

    def index(
            self,
            max_chunk_size: int = 1000,
    ) -> None:
        """Chunk the repository and build the BM25 index, saving both to disk.

        Args:
            max_chunk_size: Maximum characters per chunk (max 2000).
        """
        _validate_chunk_size(max_chunk_size)

        if not os.path.exists(RAW_DIR):
            print(
                f"Error: raw data directory not found: {RAW_DIR}",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Chunking repository...")
        chunks = build_chunks(RAW_DIR, max_chunk_size=max_chunk_size)
        if not chunks:
            print(
                f"Error: no indexable files found in {RAW_DIR}.",
                file=sys.stderr,
            )
            sys.exit(1)
        save_chunks(chunks, CHUNKS_PATH)
        print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")

        print("Building BM25 index...")
        corpus = build_corpus(chunks)
        build_index(corpus)
        print(f"Index saved to {INDEX_PATH}")

    def search_dataset(
            self,
            dataset_path: str,
            output_path: str,
            k: int = 10,
    ) -> None:
        """Run BM25 retrieval for every question in a dataset.

        Args:
            dataset_path: Path to the input JSON dataset file.
            output_path:  Path where the output JSON will be written.
            k:            Number of chunks to retrieve per question.
        """
        _validate_k(k)
        chunks, index = _load_index_and_chunks()
        data = _load_dataset(dataset_path)

        search_result: list = []
        for question in tqdm(data.rag_questions, desc="Searching"):
            results = retriever.search(
                question.question, index, chunks, k=k
            )
            search_result.append(MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=results,
            ))

        output = StudentSearchResults(search_results=search_result, k=k)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Results saved to {output_path}")

    def answer_dataset(
            self,
            dataset_path: str,
            output_path: str,
            k: int = 10,
    ) -> None:
        """Retrieve chunks and generate an answer for every question.

        Args:
            dataset_path: Path to the input JSON dataset file.
            output_path:  Path where the output JSON will be written.
            k:            Number of chunks to retrieve per question.
        """
        _validate_k(k)
        chunks, index = _load_index_and_chunks()
        data = _load_dataset(dataset_path)
        tokenizer, model = load_model()

        answers: list = []
        for question in tqdm(data.rag_questions, desc="Answering"):
            results = retriever.search(
                question.question, index, chunks, k=k
            )
            answers.append(MinimalAnswer(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=results,
                answer=generate(
                    question.question, results, tokenizer, model
                ),
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=k
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))
        print(f"Results saved to {output_path}")

    def search(
            self,
            query: str,
            k: int = 10,
    ) -> None:
        """Search the index with a single query and print the top-k results.

        Args:
            query: The search query string.
            k:     Number of chunks to retrieve.
        """
        if not query or not query.strip():
            print("Error: query must not be empty.", file=sys.stderr)
            sys.exit(1)
        _validate_k(k)
        chunks, index = _load_index_and_chunks()
        results = retriever.search(query, index, chunks, k=k)
        for i, chunk in enumerate(results):
            print(f"[{i + 1}] {chunk.file_path} "
                  f"({chunk.first_character_index}"
                  f"–{chunk.last_character_index})")

    def answer(
            self,
            query: str,
            k: int = 10,
    ) -> None:
        """Answer a single query using retrieved context from the index.

        Args:
            query: The question to answer.
            k:     Number of chunks to retrieve.
        """
        if not query or not query.strip():
            print("Error: query must not be empty.", file=sys.stderr)
            sys.exit(1)
        _validate_k(k)
        chunks, index = _load_index_and_chunks()
        tokenizer, model = load_model()
        results = retriever.search(query, index, chunks, k=k)
        response = generate(query, results, tokenizer, model)
        print(response)
