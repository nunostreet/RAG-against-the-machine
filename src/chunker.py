"""File discovery and chunk creation.

The retrieval output must point back to exact file slices, so chunks are stored
as `MinimalSource` objects rather than only raw text. Each source contains a
relative path and character offsets into that file.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

from src.models import MinimalSource

CHUNK_SIZE = 2000
OVERLAP = 200
INDEXED_EXTENSIONS = {".py", ".md", ".txt"}


def _relative_posix_path(file_path: str) -> str:
    """Return the path relative to the project root with forward slashes.

    file_path must match the corpus path exactly, so "as_posix" is used.

    Args:
        file_path: Path to a file inside the corpus.

    Returns:
        The relative path using `/` as separator.
    """
    return Path(os.path.relpath(file_path, start=".")).as_posix()


def chunk_python(
    file_path: str,
    text: str,
    chunk_size: int,
) -> list[MinimalSource]:
    """Split Python source into chunks that keep definitions whole.

    The file is cut only at the start of top-level `def`, `async def` and
    `class` statements, and each chunk packs as many whole definitions as
    fit in `chunk_size`. A definition longer than `chunk_size` is cut at the
    limit. Chunks do not overlap. Files that do not parse as Python fall back
    to `chunk_text`.

    Args:
        file_path: Path of the source file, used for the chunk metadata.
        text: Full content of the file.
        chunk_size: Maximum number of characters per chunk.

    Returns:
        The chunks as character ranges of `text`, in file order.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return chunk_text(file_path, text, chunk_size)

    lines = text.splitlines(keepends=True)  # to keep line breaks included
    line_starts = [0]
    # MinimalSource needs character positions
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

    relative_path = _relative_posix_path(file_path)

    start = 0
    chunks: list[MinimalSource] = []

    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))

        possible_ends = [
            b for b in boundaries
            if start < b <= maximum_end
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
    """Split text into fixed-size windows that overlap.

    Consecutive chunks share `OVERLAP` characters, so a sentence cut at one
    boundary is still whole in the next chunk. The overlap is capped at a
    fifth of `chunk_size` so the window always moves forward.

    Args:
        file_path: Path of the source file, used for the chunk metadata.
        text: Full content of the file.
        chunk_size: Maximum number of characters per chunk.

    Returns:
        The chunks as character ranges of `text`, in file order.
    """
    relative_path = _relative_posix_path(file_path)

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
    """Read a file and chunk it with the strategy for its extension.

    Args:
        file_path: Path of the file to chunk.
        chunk_size: Maximum number of characters per chunk.

    Returns:
        The chunks of the file, in file order.

    Raises:
        ValueError: If `chunk_size` is not positive.
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

    `os.walk` returns entries in file system order, which differs between
    machines. Directory and file names are sorted so every run produces the
    same chunk list, and BM25 ties between chunks break the same way on
    every machine. Files that cannot be read are skipped with a warning on
    stderr.

    Args:
        raw_dir: Root directory of the corpus.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The chunks of every indexed file, in walk order.
    """
    all_chunks: list[MinimalSource] = []

    for dirpath, dirnames, filenames in os.walk(raw_dir):
        # In place: os.walk reads this list to choose the next directories
        dirnames.sort()
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in INDEXED_EXTENSIONS:
                path = os.path.join(dirpath, fname)
                try:
                    all_chunks.extend(chunk_file(path, max_chunk_size))
                except OSError as exc:
                    print(f"Warning: skipped {path}: {exc}", file=sys.stderr)

    return all_chunks
