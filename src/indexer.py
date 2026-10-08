"""BM25 index construction and chunk persistence.

The index stores tokenized text for ranking, while `chunks.json` stores the
metadata needed to reconstruct `MinimalSource` outputs after retrieval.
"""

import json
import os
import re

from src.models import MinimalSource
import bm25s

DEFAULT_INDEX_PATH = "data/processed/bm25_index"
DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"


def _path_words(file_path: str) -> str:
    """Turn a corpus path into searchable words.

    The `data/raw/<repo>/` prefix is dropped because every chunk shares it.
    The rest is split on `/`, `.`, `_` and `-`, so
    `fused_moe/fused_batched_moe.py` gives `fused moe fused batched moe py`.

    Args:
        file_path: Path of the chunk's file, as stored in MinimalSource.

    Returns:
        The path words separated by spaces.
    """
    relative = re.sub(r"^data/raw/[^/]+/", "", file_path)
    return " ".join(part for part in re.split(r"[/._\-]+", relative) if part)


def get_chunk_text(chunk: MinimalSource) -> str:
    """Read and return the raw text for a chunk using character offsets.

    The subject defines offsets as character positions, so the file is read as
    text and sliced as a Python string.
    """
    with open(chunk.file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text[chunk.first_character_index:chunk.last_character_index]


def build_corpus(chunks: list[MinimalSource]) -> list[str]:
    """Build the text BM25 indexes for each chunk.

    Each entry is the chunk's content plus the words of its file path, so a
    question that names a module (e.g. "fused batched MoE") also matches the
    chunks of that file. The path words only affect ranking: the chunks
    themselves, and their offsets, are unchanged.

    Args:
        chunks: Chunks in index order.

    Returns:
        One text per chunk, in the same order as `chunks`.
    """
    return [
        f"{get_chunk_text(chunk)}\n{_path_words(chunk.file_path)}"
        for chunk in chunks
    ]


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
