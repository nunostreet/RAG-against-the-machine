from __future__ import annotations

import os
from src.models import MinimalSource

CHUNK_SIZE = 1000
OVERLAP = 200
INDEXED_EXTENSIONS = {".py", ".md", ".txt"}


def chunk_file(file_path: str) -> list[MinimalSource]:
    """Split a single file into overlapping chunks of CHUNK_SIZE characters.

    Each chunk is a MinimalSource pointing to a slice of the file via
    character offsets. Overlap ensures no information is cut off at boundaries.
    """
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Relative path from the repo root so chunks are portable across machines
    relative_path = os.path.relpath(file_path, start=".")
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(MinimalSource(
            file_path=relative_path,
            first_character_index=start,
            last_character_index=end,
        ))
        if end == len(text):
            break
        # Advance by (CHUNK_SIZE - OVERLAP) so consecutive chunks share
        # OVERLAP characters — this prevents information loss at chunk edges
        start += CHUNK_SIZE - OVERLAP

    return chunks


def build_chunks(raw_dir: str) -> list[MinimalSource]:
    """Walk raw_dir recursively and chunk every file with an indexed extension.
    """
    all_chunks: list[MinimalSource] = []

    for dirpath, _, filenames in os.walk(raw_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1] in INDEXED_EXTENSIONS:
                all_chunks.extend(chunk_file(os.path.join(dirpath, fname)))

    return all_chunks
