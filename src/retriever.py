"""BM25 retrieval helpers.

This module is intentionally small: loading the index and translating BM25
corpus positions back into `MinimalSource` objects.
"""

import os

from src.models import MinimalSource
import bm25s


def load_index(file_path: str) -> bm25s.BM25:
    """Load a previously saved BM25 index from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"BM25 index not found at '{file_path}'. "
            "Run 'uv run python -m src index' first."
        )
    return bm25s.BM25.load(file_path)


def search(
        query: str,
        retriever: bm25s.BM25,
        chunks: list[MinimalSource],
        k: int = 10) -> list[MinimalSource]:
    """Return the top-k chunks most relevant to the query via BM25.

    bm25s raises an error if `k` is larger than the corpus size, so `k` is
    clamped to the number of available chunks. This keeps tiny test corpora and
    real corpora using the same code path.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    if not chunks:
        return []

    effective_k = min(k, len(chunks))

    # retrieve returns (results, scores); only positions are needed here.
    results, _ = retriever.retrieve(
        bm25s.tokenize([query]), k=effective_k
    )

    # results[0] contains corpus positions for the single query we passed in.
    return [chunks[i] for i in results[0]]
