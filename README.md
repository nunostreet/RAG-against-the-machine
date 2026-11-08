*This project has been created as part of the 42 curriculum by nstreet-*

# RAG against the machine

## Description

This project answers questions about the supplied vLLM source code and
documentation. Instead of expecting a language model to know the repository,
it searches for useful passages and gives those passages to the model. That
is the idea behind Retrieval-Augmented Generation, or RAG.

Search uses BM25. Answer generation uses `Qwen/Qwen3-0.6B`, running locally on
CPU. Results include file paths and character offsets, so you can open the
evidence and check the answer yourself.

You can run search without loading Qwen. Start there when checking whether the
system is finding the right information; a fluent answer alone does not tell
you whether retrieval worked.

## Instructions

### 1. Set up the environment

You need Python 3.10 or later, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and enough free space for PyTorch, the model weights, and the corpus. A GPU is
not required. Python 3.10 is the version used for the local checks below.

Run commands from the repository root:

```bash
uv sync --python 3.10
```

For tests and code checks, also install the development dependencies:

```bash
uv sync --extra dev
```

There is no separate compilation step. `uv` creates `.venv` and installs the
dependencies using `pyproject.toml` and `uv.lock`.

### 2. Put the supplied data in place

Extract the **provided vLLM 0.10.1 archive**, keeping its directory name:

```text
data/
├── raw/
│   └── vllm-0.10.1/
│       ├── docs/
│       ├── examples/
│       └── vllm/
└── datasets/
    ├── UnansweredQuestions/
    │   ├── dataset_docs_public.json
    │   └── dataset_code_public.json
    └── AnsweredQuestions/
        ├── dataset_docs_public.json
        └── dataset_code_public.json
```

The public datasets are included in this repository. Each of the four files
above contains 100 questions. Duplicate copies have been removed.

Use the supplied corpus rather than a checkout of the latest vLLM version.
Evaluation compares paths exactly, and its reference offsets refer to that
specific version. For example, `data/raw/vllm-0.10.1/docs/features/lora.md` must
keep that prefix.

### 3. Build the index

```bash
uv run python -m src index --max_chunk_size 2000
```

This creates the BM25 index in `data/processed/bm25_index/` and source locations
in `data/processed/chunks.json`. Build the index again whenever you change the
corpus or the chunking settings. Keep these two artifacts together: BM25 result
positions refer to the order of entries in `chunks.json`.

### 4. Search before generating

Try one question:

```bash
uv run python -m src search "How can I enable LoRA in vLLM?" --k 5
```

The command prints JSON and saves it to
`data/output/search_results/query.json`. Each source is a half-open character
range: `text[first_character_index:last_character_index]` in the original file.

To search both public question sets:

```bash
for dataset in docs code; do
    uv run python -m src search_dataset \
        --dataset_path "data/datasets/UnansweredQuestions/dataset_${dataset}_public.json" \
        --save_directory data/output/search_results/UnansweredQuestions \
        --k 10
done
```

The output keeps the input file name. For example, the docs results are saved
as `data/output/search_results/UnansweredQuestions/dataset_docs_public.json`.
Use a separate `AnsweredQuestions` output directory when searching that scope;
the two scopes share dataset file names.

### 5. Check retrieval quality

Use the answered dataset as ground truth for the corresponding search output:

```bash
for dataset in docs code; do
    uv run python -m src evaluate \
        --student_search_results_path "data/output/search_results/UnansweredQuestions/dataset_${dataset}_public.json" \
        --dataset_path "data/datasets/AnsweredQuestions/dataset_${dataset}_public.json" \
        --k 5 \
        --max_context_length 2000
done
```

For each question, recall@k is the fraction of its expected sources found in
the first k results. The evaluator averages those fractions across answered
questions. A source matches when it has the exact same file path and an
intersection-over-union (IoU) of at least 0.05 between the character ranges.

Missing results count as zero. Unanswered questions are ignored; answered
questions with no reference sources contribute zero. A retrieved source longer
than the configured character limit makes the output invalid, even if that
source is outside the top-k being scored.

This evaluator is for development. The supplied **moulinette on Linux** is the
final check. Run its `evaluate_student_search_results` command with the search
results path first, the answered dataset second, and `--k 10
--max_context_length 2000`. Check that the output is valid and recall@5 meets
both dataset thresholds.

### 6. Download Qwen, then generate answers

The generator only loads models already present in the Hugging Face cache.
Download the model once, while you have an internet connection:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B')"
```

Then try one answer:

```bash
uv run python -m src answer "How can I enable LoRA in vLLM?" --k 5
```

It prints JSON and saves it to
`data/output/search_results_and_answer/query.json`. To answer a previously
searched dataset:

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

`answer_dataset` inherits `k` from the search file. Use `--k 5`, for example,
to use only its first five sources. It does not retrieve additional sources
when answering an existing search file. Dataset generation is much slower than
retrieval on CPU; start with a few questions before attempting a full run.

Empty questions in a dataset are kept with no retrieved sources and an empty
answer. Single-query commands reject empty input. Arguments such as `123` that
Fire interprets as Python literals are converted back to query text.

### Choosing other paths

| Option | Commands | Purpose |
| --- | --- | --- |
| `--raw_dir` | `index` | Corpus directory; defaults to `data/raw` |
| `--index_path` | `index`, `search`, `search_dataset`, `answer`, direct dataset answering | BM25 index directory |
| `--chunks_path` | Same commands as `--index_path` | Chunk metadata file |
| `--save_directory` | Search and answer commands | Output directory |
| `--output_path` | Search, answer, and evaluation commands | Exact output file; overrides the output directory |

Single-query commands use `query.json` by default, so a later run replaces that
file. Set `--output_path` when you want to keep several results:

```bash
uv run python -m src search "How is KV cache managed?" \
    --k 5 --output_path data/output/search_results/kv_cache.json
```

`answer_dataset --dataset_path <path>` also supports retrieving and answering
in one command, with `k=10` by default. Keeping search and generation separate
is usually easier to inspect. `evaluate` still accepts the older
`--student_results_path` name as an alias.

## System architecture

The pipeline is:

```text
vLLM files → chunks + source locations → BM25 index
question → top-k source locations → context → Qwen → answer + sources
```

| Module | Responsibility |
| --- | --- |
| `src/chunker.py` | Discover files and split Python and text into bounded spans |
| `src/indexer.py` | Read chunk text and save the BM25 index and metadata |
| `src/retriever.py` | Rank chunks and map index positions back to sources |
| `src/generator.py` | Build the prompt and run Qwen |
| `src/evaluator.py` | Measure local retrieval recall |
| `src/models.py` | Validate exchanged data with Pydantic |
| `src/cli.py`, `src/__main__.py` | Expose the pipeline through Python Fire |

Search JSON follows `StudentSearchResults`. Answer JSON follows
`StudentSearchResultsAndAnswer`. Both contain a `search_results` list and `k`;
each answer adds an `answer` field to its question and retrieved sources.

## Chunking strategy

The indexer reads `.py`, `.md`, and `.txt` files in sorted order. The default
maximum chunk size is **2000 characters**, and the CLI accepts values from 1 to
2000. The limit applies to each source, not the combined model context.

For **Python**, the AST identifies the start of top-level functions, async
functions, and classes. The chunker chooses a definition boundary within the
size limit when possible. A large definition is split at the character limit;
methods inside a class are not treated as separate AST boundaries. These
chunks cover the file without overlap. Invalid Python falls back to text
chunking. Files that cannot be read are skipped with a warning so the remaining
corpus can still be indexed.

For **Markdown and text**, fixed character windows overlap by 200 characters,
or one fifth of the chunk size when that is smaller. This helps preserve
information near a boundary. This strategy does not currently split on
Markdown headings or paragraphs.

## Retrieval method and design decisions

BM25 ranks chunks using query term matches, term rarity, and document-length
normalization. Corpus and questions use `bm25s.tokenize()` with its defaults.
Each indexed entry contains the chunk text followed by words from its file
path. The common `data/raw/<repo>/` prefix is removed and path separators,
underscores, hyphens, and dots become word boundaries. This helps questions
that name modules, while source offsets still refer only to the original text.
There is no embedding model, reranker, or semantic query expansion.

BM25 gives a small, fast baseline that is easy to inspect on CPU. Its weakness
is vocabulary mismatch: a question can describe a concept without using the
words that appear in the code. Identifiers are useful when the question quotes
them, but the current tokenizer does not add custom camelCase or snake_case
expansions. If `k` exceeds the corpus size, retrieval uses the available chunks.

Source locations are stored separately from ranking data to keep outputs tied
to the original files. This also means you should rebuild after editing the
corpus; the saved offsets do not track file changes automatically.

For generation, sources are read in rank order and prefixed with their
locations. Context is capped at 12,000 characters, including headers and
separators. The complete chat prompt is then checked against the model and
tokenizer context limits, reserving up to 512 tokens for the answer. If needed,
only the context is shortened; the question is preserved. There is no extra
fixed input-token ceiling.

Qwen runs with thinking disabled, greedy decoding, and inference mode. The
prompt asks it to use the supplied context and acknowledge insufficient
evidence. That instruction helps, but it does not guarantee a correct answer.
The JSON lists the selected sources, including sources whose text may have
been shortened or omitted by the context limit; it is not a list of verified
claim-by-claim citations.

## Performance analysis

Local measurements on **8 November 2026**, on macOS ARM64 with Python 3.10,
used the complete public datasets: 100 docs questions and 100 code questions.
The saved index was rebuilt with the current chunker at 2000 characters.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Docs | 61% | 78% | **85%** | 90% | 80% |
| Code | 43% | 59% | **67%** | 72% | 50% |

Indexing produced **12,819 chunks from 1,856 files** in **6.88 seconds**. Search
over the 200 questions took **5.24 seconds** across two CLI invocations. These
wall times include process startup and, for search, loading the saved index.
The requirements are at most 5 minutes for indexing and 90 seconds for 200
searches. Generation time is not included in these retrieval measurements.

A local comparison with the same indexing method and all 100 questions per
dataset gave recall@5 of 81% for docs and 64% for code with 1000-character
chunks. The 2000-character default gave 85% and 67%, so it is the baseline used
here. This comparison does not establish the best possible chunk size.

These are **local evaluator results**, not official Linux moulinette results.
Hardware, dependency versions, and filesystem caching affect timings. Retrieval
recall also does not measure answer quality. When present, the generated report
is at `data/output/evaluation/benchmark.json`; it is excluded from Git.

## Challenges faced

The main issues during development were at the boundaries between stages:

- **Exact source locations:** paths must match the reference corpus, and offsets
  must remain character positions. Chunks store those locations and are read
  using the same text slicing convention.
- **Model inputs:** the tokenizer returns named inputs. Passing its whole result
  as a positional tensor broke generation. The generator now passes `input_ids`
  and `attention_mask` by name and removes the prompt from the decoded output.
- **Misleading evaluation:** missing questions were previously skipped, and a
  global source average gave more weight to questions with more sources. The
  evaluator now includes missing results and averages per question.
- **Reproducible runs:** outdated index files and a missing public question made
  comparisons unreliable. The original datasets were restored and the index
  rebuilt before recording the results above.
- **CLI edge cases:** malformed files and invalid arguments need understandable
  errors. Tests cover these paths, and answer inputs are checked before loading
  the model.

## Development

```bash
make lint
make lint-strict
make test
```

There are currently **51 tests**, covering chunking, evaluation, prompt budgets,
and the CLI, including a small persisted search pipeline. Most generation tests
use test doubles, so passing them does not replace trying the real model.

The Makefile also provides `install`, `run`, `debug`, and `clean`. `make run`
opens the CLI command listing; use the examples above to run a specific stage.
`make clean` removes Python/type-checker caches, not the saved index or results.

VS Code defaults to `.venv/bin/python` for this workspace. If it already has
another interpreter selected, use **Python: Select Interpreter** to choose this
environment. Pyright checks `src/` and `test.py` in standard mode. Linting and
mypy exclude the downloaded corpus, supplied tools, and local AST experiment.

Before submission, repeat installation and evaluation on a fresh Linux machine
and review a sample of generated answers against their source passages.

## Resources

- [Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
  introduces the original RAG approach. This project uses a simpler lexical
  retriever rather than reproducing that paper's trained system.
- [BM25S documentation and source](https://github.com/xhluca/bm25s) explains the
  library used for lexical retrieval.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) documents the
  required model and its chat-template usage.
- [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download)
  explains model downloads and the local cache.
- [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
  covers installing the project manager.

AI assistance was used to compare the project with the subject, identify and
fix CLI and evaluation issues, correct model-input handling and typing in the
generator, add regression tests, restore the supplied datasets, run local
benchmarks, and draft documentation. This involved `src/cli.py`,
`src/__main__.py`, `src/evaluator.py`, `src/generator.py`, supporting chunk/index
validation and configuration, `test.py`, and the project documentation. The
student remains responsible for understanding the code and checking its results.
