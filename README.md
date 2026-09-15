*This project has been created as part of the 42 curriculum by nstreet-*

# RAG System for vLLM

This project implements a small Retrieval-Augmented Generation system for the
vLLM repository. It indexes files from `data/raw`, retrieves relevant source
chunks with BM25, and generates grounded answers with `Qwen/Qwen3-0.6B`.

## Requirements

- Python 3.10+
- `uv`
- Dependencies from `pyproject.toml`
- The vLLM source tree under `data/raw`
- The Qwen model available in the HuggingFace cache for answer generation

Install:

```bash
uv sync --extra dev
```

## CLI

Build the index:

```bash
uv run python -m src index --max_chunk_size 1000
```

Search one question:

```bash
uv run python -m src search "How is KV cache managed?" --k 10
```

Search a dataset:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --save_directory data/output/search_results \
  --k 10
```

Generate answers from search results:

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_code_public.json \
  --save_directory data/output/answers \
  --k 10
```

Evaluate retrieval against an answered dataset:

```bash
uv run python -m src evaluate \
  --student_results_path data/output/search_results/dataset_code_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 10 \
  --max_context_length 2000
```

The local evaluator is intended for development feedback. The official
moulinette remains the source of truth for final grading.

## Architecture

The system is intentionally simple:

- `src/models.py`: Pydantic models matching the expected input/output schemas.
- `src/chunker.py`: deterministic recursive file discovery and overlapping
  character-based chunks.
- `src/indexer.py`: corpus construction and BM25 index persistence.
- `src/retriever.py`: top-k BM25 retrieval with bounds checking.
- `src/generator.py`: bounded context construction and Qwen answer generation.
- `src/evaluator.py`: local recall evaluation against answered datasets.
- `src/cli.py`: Fire-based command-line interface.

## Chunking And Retrieval

Chunks are character ranges stored as `MinimalSource` objects. The file path is
relative to the repository root and offsets point to the exact slice used as
context. Files are walked in sorted order so the index is reproducible across
runs.

Retrieval uses BM25 through `bm25s`. If `k` is larger than the number of indexed
chunks, the retriever automatically clamps it to the available corpus size.

## Generation

Answers are generated with `Qwen/Qwen3-0.6B` using only retrieved context. The
context is capped before generation to avoid sending very large prompts during
dataset answering.

## Development

Run linting and typing:

```bash
make lint
```

Run tests:

```bash
make test
```

## AI Use

AI assistance was used to review the subject, identify missing requirements,
document the implementation, and improve the CLI, validation, retrieval, local
evaluation, and tests.
