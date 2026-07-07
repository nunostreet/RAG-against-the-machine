from __future__ import annotations

import os
from src.models import MinimalSource

CHUNK_SIZE = 1000
OVERLAP = 200
INDEXED_EXTENSIONS = {".py", ".md", ".txt"}


def chunk_file(file_path: str) -> list[MinimalSource]:
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # os.path.relpath converte um caminho absoluto num caminho relativo ao
    # diretório atual
    relative_path = os.path.relpath(file_path, start=".")
    chunks = []
    start = 0

    # avança CHUNK_SIZE - OVERLAP a cada iteração,
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(MinimalSource(
            file_path=relative_path,
            first_character_index=start,
            last_character_index=end,
        ))
        if end == len(text):
            break
        start += CHUNK_SIZE - OVERLAP

    return chunks


def build_chunks(raw_dir: str) -> list[MinimalSource]:
    all_chunks: list[MinimalSource] = []

    # os.walk percorre uma pasta recursivamente
    for dirpath, _, filenames in os.walk(raw_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1] in INDEXED_EXTENSIONS:
                all_chunks.extend(chunk_file(os.path.join(dirpath, fname)))

    return all_chunks
