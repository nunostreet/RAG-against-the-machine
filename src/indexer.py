"""BM25 index construction and chunk persistence.

The index stores tokenized text for ranking, while `chunks.json` stores the
metadata needed to reconstruct `MinimalSource` outputs after retrieval.
"""

import json
import os

from src.models import MinimalSource
import bm25s

DEFAULT_INDEX_PATH = "data/processed/bm25_index"
DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"


def get_chunk_text(chunk: MinimalSource) -> str:
    """Read and return the raw text for a chunk using character offsets.

    The subject defines offsets as character positions, so the file is read as
    text and sliced as a Python string.
    """
    with open(chunk.file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[chunk.first_character_index:chunk.last_character_index]


def build_corpus(chunks: list[MinimalSource]) -> list[str]:
    """Convert chunk metadata into the raw strings indexed by BM25."""
    corpus = [get_chunk_text(chunk) for chunk in chunks]
    return corpus


def build_index(
    corpus: list[str], index_path: str = DEFAULT_INDEX_PATH
) -> bm25s.BM25:
    """Build and save a BM25 index from a corpus of text strings.

    bm25s persists the ranking structure, but not the original `MinimalSource`
    objects. The caller must save chunks separately with `save_chunks`.
    """
    if not corpus:
        raise ValueError("cannot build a BM25 index from an empty corpus")

    # Creating an empty BM25 object
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    # Saving the index in the index_path
    retriever.save(index_path)
    return retriever


def save_chunks(
    chunks: list[MinimalSource], path: str = DEFAULT_CHUNKS_PATH
) -> None:
    """Serialize chunk metadata to JSON on disk."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([chunk.model_dump() for chunk in chunks], f)


def load_chunks(
    path: str = DEFAULT_CHUNKS_PATH
) -> list[MinimalSource]:
    """Load chunk metadata and validate it through Pydantic."""
    with open(path, encoding="utf-8") as f:
        return [MinimalSource(**item) for item in json.load(f)]
