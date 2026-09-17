"""File discovery and chunk creation.

The retrieval output must point back to exact file slices, so chunks are stored
as `MinimalSource` objects rather than only raw text. Each source contains a
relative path and character offsets into that file.
"""

from __future__ import annotations

import ast
import os

from src.models import MinimalSource

CHUNK_SIZE = 2000
OVERLAP = 200
INDEXED_EXTENSIONS = {".py", ".md", ".txt"}


def chunk_python(
    file_path: str,
    text: str,
    chunk_size: int,
) -> list[MinimalSource]:

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_text(file_path, text, chunk_size)

    lines = text.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    interesting_types = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    boundaries = [0]

    for node in tree.body:
        if isinstance(node, interesting_types):
            start = line_starts[node.lineno - 1]
            boundaries.append(start)

    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))

    relative_path = os.path.relpath(file_path, start=".")

    start = 0
    chunks: list[MinimalSource] = []

    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))

        possible_ends = [
            boundary
            for boundary in boundaries
            if start < boundary <= maximum_end
        ]

        end = max(possible_ends, default=maximum_end)

        chunks.append(
            MinimalSource(
                file_path=relative_path,
                first_character_index=start,
                last_character_index=end,
            )
        )

        start = end

    return chunks


def chunk_text(
    file_path: str,
    text: str,
    chunk_size: int,
) -> list[MinimalSource]:

    relative_path = os.path.relpath(file_path, start=".")

    # To avoid negative / very small overlaps
    overlap = min(OVERLAP, max(0, chunk_size // 5))
    step = max(1, chunk_size - overlap)

    chunks: list[MinimalSource] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(MinimalSource(
            file_path=relative_path,
            first_character_index=start,
            last_character_index=end,
        ))
        if end == len(text):
            break
        # Consecutive chunks overlap to avoid losing context at boundaries
        start += step

    return chunks


def chunk_file(
    file_path: str, chunk_size: int = CHUNK_SIZE
) -> list[MinimalSource]:
    """Split a single file into overlapping chunks of chunk_size characters.

    Each chunk is a MinimalSource pointing to a slice of the file via
    character offsets. Overlap ensures no information is cut off at boundaries.
    The overlap is capped for very small chunk sizes so the loop always makes
    progress.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".py":
        return chunk_python(file_path, text, chunk_size)

    return chunk_text(file_path, text, chunk_size)


def build_chunks(
    raw_dir: str, max_chunk_size: int = CHUNK_SIZE
) -> list[MinimalSource]:
    """Walk raw_dir recursively and chunk every file with an indexed extension.

    Directory and file names are sorted to make the corpus order deterministic.
    This matters because bm25s returns corpus positions, which are mapped back
    to chunks by list index.
    """
    all_chunks: list[MinimalSource] = []

    for dirpath, dirnames, filenames in os.walk(raw_dir):
        dirnames.sort()
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in INDEXED_EXTENSIONS:
                all_chunks.extend(
                    chunk_file(os.path.join(dirpath, fname), max_chunk_size)
                )

    return all_chunks
