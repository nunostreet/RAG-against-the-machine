import json
import os
from src.models import MinimalSource
import bm25s


def get_chunk_text(chunk: MinimalSource) -> str:
    """Read and return the raw text for a chunk using its character offsets."""
    with open(chunk.file_path, encoding="utf-8", errors="ignore") as f:
        f.seek(chunk.first_character_index)
        text = f.read(chunk.last_character_index - chunk.first_character_index)
    return text


def build_corpus(chunks: list[MinimalSource]) -> list[str]:
    """Convert a list of chunks into raw text strings for indexing."""
    corpus = [get_chunk_text(chunk) for chunk in chunks]
    return corpus


def build_index(corpus: list[str]) -> bm25s.BM25:
    """Build and save a BM25 index from a corpus of text strings."""
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    retriever.save("data/processed/bm25_index")
    return retriever


def save_chunks(
    chunks: list[MinimalSource], path: str = "data/processed/chunks.json"
) -> None:
    """Serialize a list of chunks to JSON on disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([chunk.model_dump() for chunk in chunks], f)


def load_chunks(
    path: str = "data/processed/chunks.json"
) -> list[MinimalSource]:
    """Load a list of chunks from a JSON file on disk."""
    with open(path) as f:
        return [MinimalSource(**item) for item in json.load(f)]
