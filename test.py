import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import BatchEncoding

from src.chunker import build_chunks, chunk_file, chunk_text
from src.__main__ import main
from src.cli import RAGSystem
from src.evaluator import evaluate_files, evaluate_results, source_iou
from src.generator import generate
from src.indexer import build_corpus, build_index
from src.retriever import search
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_python_chunks_max_size(tmp_path: Path) -> None:
    text = (
        "def first():\n"
        "    return 1\n"
        "\n"
        "def large_function():\n"
        f"    value = \"{'x' * 100}\"\n"
        "    return value\n"
    )

    file_path = tmp_path / "example.py"

    file_path.write_text(text, encoding="utf-8")

    chunk_size = 40

    chunks = chunk_file(
        str(file_path),
        chunk_size=chunk_size,
    )

    assert chunks

    for chunk in chunks:
        actual_size = (
            chunk.last_character_index
            - chunk.first_character_index
        )

        assert actual_size <= chunk_size


def test_python_chunks_cover_entire_file(tmp_path: Path) -> None:

    text = (
        "VALUE = 42\n"
        "\n"
        "def first():\n"
        "    return 1\n"
        "\n"
        "def second():\n"
        "    return 2\n"
    )

    file_path = tmp_path / "example.py"
    file_path.write_text(text, encoding="utf-8")

    chunks = chunk_file(
        str(file_path),
        chunk_size=30,
    )

    assert chunks
    assert chunks[0].first_character_index == 0
    assert chunks[-1].last_character_index == len(text)

    for current, following in zip(chunks, chunks[1:]):
        assert (
            current.last_character_index
            == following.first_character_index
        )


def test_invalid_python(tmp_path: Path) -> None:
    text = (
        "def broken_function(\n"
        "    return 42\n"
    )

    file_path = tmp_path / "broken.py"
    file_path.write_text(text, encoding="utf-8")

    chunk_size = 10

    actual_chunks = chunk_file(
        str(file_path),
        chunk_size=chunk_size,
    )

    expected_chunks = chunk_text(
        str(file_path),
        text,
        chunk_size,
    )

    assert actual_chunks == expected_chunks


def test_minimal_source_rejects_invalid_offsets() -> None:
    with pytest.raises(ValueError):
        MinimalSource(
            file_path="file.py",
            first_character_index=10,
            last_character_index=10,
        )


def test_source_iou_requires_same_file() -> None:
    source_a = MinimalSource(
        file_path="a.py",
        first_character_index=0,
        last_character_index=100,
    )
    source_b = MinimalSource(
        file_path="b.py",
        first_character_index=0,
        last_character_index=100,
    )

    assert source_iou(source_a, source_b) == 0.0


def test_evaluate_results_counts_found_sources() -> None:
    expected_source = MinimalSource(
        file_path="src/example.py",
        first_character_index=0,
        last_character_index=100,
    )
    dataset = RagDataset(rag_questions=[
        AnsweredQuestion(
            question_id="q1",
            question="Where is the scheduler?",
            sources=[expected_source],
            answer="In src/example.py.",
        )
    ])
    student_results = StudentSearchResults(
        k=1,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="Where is the scheduler?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="src/example.py",
                        first_character_index=10,
                        last_character_index=90,
                    )
                ],
            )
        ],
    )

    metrics = evaluate_results(student_results, dataset, k=1)

    assert metrics["expected_sources"] == 1
    assert metrics["found_sources"] == 1
    assert metrics["recall"] == 1.0


def _recall_fixture() -> tuple[StudentSearchResults, RagDataset]:
    """Provide two questions with one and three expected sources."""
    sources = [MinimalSource(
        file_path=f"{name}.py", first_character_index=0,
        last_character_index=100,
    ) for name in ("a", "b", "c")]
    dataset = RagDataset(rag_questions=[
        AnsweredQuestion(
            question_id="q1", question="first", answer="first",
            sources=sources[:1],
        ),
        AnsweredQuestion(
            question_id="q2", question="second", answer="second",
            sources=sources,
        ),
    ])
    results = StudentSearchResults(k=2, search_results=[
        MinimalSearchResults(
            question_id=qid, question=qid, retrieved_sources=sources[:1],
        ) for qid in ("q1", "q2")
    ])
    return results, dataset


def test_recall_averages_questions_with_different_source_counts() -> None:
    results, dataset = _recall_fixture()
    metrics = evaluate_results(results, dataset, k=2)
    assert metrics["recall"] == pytest.approx((1 + 1 / 3) / 2)
    assert metrics["expected_sources"] == 4
    assert metrics["found_sources"] == 2


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_results_count_as_zero_recall(missing: bool) -> None:
    results, dataset = _recall_fixture()
    if missing:
        results.search_results.pop()
    else:
        results.search_results[1].retrieved_sources = []
    metrics = evaluate_results(results, dataset, k=2)
    assert metrics["recall"] == 0.5
    assert metrics["expected_sources"] == 4
    assert metrics["found_sources"] == 1
    assert metrics["answered_questions_with_result"] == (1 if missing else 2)


def test_recall_uses_only_top_k_and_includes_iou_threshold() -> None:
    results, dataset = _recall_fixture()
    dataset.rag_questions = dataset.rag_questions[:1]
    source = MinimalSource(
        file_path="a.py", first_character_index=0, last_character_index=2000,
    )
    assert source_iou(source, results.search_results[0].retrieved_sources[0]) \
        == pytest.approx(0.05)
    results.search_results[0].retrieved_sources = [
        MinimalSource(
            file_path="wrong.py", first_character_index=0,
            last_character_index=100,
        ), source,
    ]
    assert evaluate_results(results, dataset, k=1)["recall"] == 0.0
    assert evaluate_results(results, dataset, k=2)["recall"] == 1.0


def test_overlong_source_invalidates_output_even_outside_top_k() -> None:
    results, dataset = _recall_fixture()
    results.search_results[0].retrieved_sources.append(MinimalSource(
        file_path="too_long.py", first_character_index=0,
        last_character_index=2001,
    ))
    with pytest.raises(ValueError, match="2001 characters; maximum is 2000"):
        evaluate_results(results, dataset, k=1)
    assert evaluate_results(results, dataset, k=1,
                            max_context_length=None)["recall"] > 0


@pytest.mark.parametrize("limit", [0, -1])
def test_evaluator_rejects_invalid_length_limit_on_empty_input(
    limit: int,
) -> None:
    with pytest.raises(ValueError, match="max_context_length"):
        evaluate_results(StudentSearchResults(k=1, search_results=[]),
                         RagDataset(rag_questions=[]), k=1,
                         max_context_length=limit)


def test_empty_ground_truth_and_unanswered_questions() -> None:
    results = StudentSearchResults(k=1, search_results=[])
    dataset = RagDataset(rag_questions=[
        UnansweredQuestion(question="No ground truth"),
    ])
    assert evaluate_results(results, dataset, k=1)["answered_questions"] == 0
    assert evaluate_results(results, dataset, k=1)["recall"] == 0.0
    dataset.rag_questions.append(AnsweredQuestion(
        question="No sources", answer="Unknown", sources=[],
    ))
    metrics = evaluate_results(results, dataset, k=1)
    assert metrics["answered_questions"] == 1
    assert metrics["recall"] == 0.0


def test_evaluation_file_errors_become_value_errors(tmp_path: Path) -> None:
    missing = str(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="invalid evaluation input"):
        evaluate_files(missing, missing, k=1)


def _generation_doubles(
    model_window: int, tokenizer_window: int
) -> tuple[MagicMock, MagicMock]:
    """Use character tokens to check prompt budgeting without model weights."""
    tokenizer = MagicMock(model_max_length=tokenizer_window)

    def encode(messages: list[dict[str, str]], **kwargs: object
               ) -> BatchEncoding:
        assert kwargs["return_dict"] is True
        text = "\n".join(message["content"] for message in messages)
        ids = torch.tensor([[ord(char) for char in text]])
        return BatchEncoding({
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        })

    tokenizer.apply_chat_template.side_effect = encode
    tokenizer.decode.side_effect = lambda ids, **kw: "".join(
        chr(int(token)) for token in ids
    )
    model = MagicMock(
        config=SimpleNamespace(max_position_embeddings=model_window),
        device=torch.device("cpu"),
    )

    def complete(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int, do_sample: bool) -> torch.Tensor:
        assert input_ids.shape == attention_mask.shape
        assert input_ids.shape[-1] + max_new_tokens <= model_window
        assert not do_sample
        assert torch.is_inference_mode_enabled()
        return torch.cat((input_ids, torch.tensor([[ord("A")]])), dim=1)

    model.generate.side_effect = complete
    return tokenizer, model


@pytest.mark.parametrize("model_window,tokenizer_window,budget", [
    (512, 1024, 496),
    (300, 1024, 284),
    (1024, 300, 284),
])
def test_generation_budgets_prompt_and_preserves_question(
    tmp_path: Path, model_window: int, tokenizer_window: int,
    budget: int,
) -> None:
    text = "evidence 漢字 " * 200
    path = tmp_path / "context.txt"
    path.write_text(text, encoding="utf-8")
    source = MinimalSource(
        file_path=str(path), first_character_index=0,
        last_character_index=len(text),
    )
    tokenizer, model = _generation_doubles(model_window, tokenizer_window)
    question = "What does the evidence say?"
    assert generate(
        question, [source], tokenizer, model,
        max_new_tokens=16,
    ) == "A"
    inputs = model.generate.call_args.kwargs["input_ids"]
    assert inputs.shape[-1] <= budget
    prompt = "".join(chr(int(token)) for token in inputs[0])
    assert prompt.endswith(f"Question: {question}")
    assert "evidence" in prompt


def test_generation_rejects_question_that_cannot_fit() -> None:
    tokenizer, model = _generation_doubles(512, 512)
    with pytest.raises(ValueError, match="question and chat template"):
        generate("question " * 100, [], tokenizer, model,
                 max_new_tokens=16)
    model.generate.assert_not_called()


@pytest.fixture
def cli_paths(tmp_path: Path) -> dict[str, str]:
    """Build a small persisted corpus using configurable CLI paths."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "example.md").write_text(
        "LoRA adapters enable efficient model customization.\n" * 8,
        encoding="utf-8",
    )
    paths = {
        "index_path": str(tmp_path / "processed" / "index"),
        "chunks_path": str(tmp_path / "processed" / "chunks.json"),
    }
    RAGSystem().index(max_chunk_size=100, raw_dir=str(raw), **paths)
    return paths


def _invoke_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    """Exercise the Fire entry point without launching a new interpreter."""
    monkeypatch.setattr(sys, "argv", ["rag", *args])
    main()


def _path_flags(paths: dict[str, str]) -> list[str]:
    """Convert artifact paths to command-line flags."""
    return [part for key, value in paths.items()
            for part in (f"--{key}", value)]


def test_cli_search_dataset_evaluate_pipeline(
    cli_paths: dict[str, str], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    capsys.readouterr()
    output_dir = tmp_path / "single"
    _invoke_cli(monkeypatch, "search", "LoRA adapters", "--k", "2",
                "--save_directory", str(output_dir), *_path_flags(cli_paths))
    single = StudentSearchResults.model_validate_json(capsys.readouterr().out)
    assert json.loads((output_dir / "query.json").read_text()) \
        == single.model_dump()
    assert single.k == 2
    source = single.search_results[0].retrieved_sources[0]
    assert Path(source.file_path).is_file()

    dataset = tmp_path / "questions.json"
    dataset.write_text(RagDataset(rag_questions=[AnsweredQuestion(
        question_id="q1", question="LoRA adapters", answer="Adapters",
        sources=[source],
    )]).model_dump_json())
    batch_dir = tmp_path / "batch"
    _invoke_cli(monkeypatch, "search_dataset", "--dataset_path", str(dataset),
                "--save_directory", str(batch_dir), "--k", "2",
                *_path_flags(cli_paths))
    capsys.readouterr()
    batch_path = batch_dir / dataset.name
    batch = StudentSearchResults.model_validate_json(batch_path.read_text())
    assert batch.search_results[0].question_id == "q1"
    metrics_path = tmp_path / "metrics" / "recall.json"
    for flag in ("--student_search_results_path", "--student_results_path"):
        _invoke_cli(monkeypatch, "evaluate", flag, str(batch_path),
                    "--dataset_path", str(dataset), "--k", "2",
                    "--output_path", str(metrics_path))
        metrics = json.loads(capsys.readouterr().out)
        assert metrics["recall"] == 1.0
        assert json.loads(metrics_path.read_text()) == metrics


@pytest.mark.parametrize("override,expected_k", [(None, 2), (1, 1)])
def test_cli_answer_dataset_inherits_k_or_accepts_override(
    cli_paths: dict[str, str], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    override: int | None, expected_k: int,
) -> None:
    search_path = tmp_path / "retrieval.json"
    RAGSystem().search("LoRA", k=2, output_path=str(search_path), **cli_paths)
    capsys.readouterr()
    loader = MagicMock(return_value=(object(), object()))
    generator = MagicMock(return_value="LoRA enables model customization.")
    monkeypatch.setattr("src.cli.load_model", loader)
    monkeypatch.setattr("src.cli.generate", generator)
    output_dir = tmp_path / "answers"
    flags = [] if override is None else ["--k", str(override)]
    _invoke_cli(monkeypatch, "answer_dataset",
                "--student_search_results_path", str(search_path),
                "--save_directory", str(output_dir), *flags)
    output = StudentSearchResultsAndAnswer.model_validate_json(
        (output_dir / search_path.name).read_text()
    )
    assert output.k == expected_k
    assert len(output.search_results[0].retrieved_sources) == expected_k
    assert len(generator.call_args.args[1]) == expected_k
    loader.assert_called_once()


def test_cli_single_answer_writes_structured_output(
    cli_paths: dict[str, str], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("src.cli.load_model", lambda: (object(), object()))
    monkeypatch.setattr("src.cli.generate", lambda *args: "LoRA adapters.")
    capsys.readouterr()
    output_path = tmp_path / "answers" / "custom.json"
    _invoke_cli(monkeypatch, "answer", "LoRA", "--k", "2",
                "--output_path", str(output_path), *_path_flags(cli_paths))
    output = StudentSearchResultsAndAnswer.model_validate_json(
        capsys.readouterr().out
    )
    assert json.loads(output_path.read_text()) == output.model_dump()
    assert output.k == 2
    assert output.search_results[0].answer == "LoRA adapters."


@pytest.mark.parametrize("args", [
    ["search", ""],
    ["search", "   "],
    ["search", "LoRA", "--k", "0"],
    ["search", "LoRA", "--k", "oops"],
    ["search", "LoRA", "--k", "True"],
    ["index", "--max_chunk_size", "2001"],
    ["index", "--max_chunk_size", "oops"],
    ["answer_dataset"],
])
def test_cli_rejects_invalid_arguments_without_traceback(
    args: list[str], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader = MagicMock()
    monkeypatch.setattr("src.cli.load_model", loader)
    with pytest.raises(SystemExit) as exc:
        _invoke_cli(monkeypatch, *args)
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "Error:" in error
    assert "Traceback" not in error
    loader.assert_not_called()


@pytest.mark.parametrize("content", ["{", "{}", "null", "[]"])
def test_cli_rejects_invalid_answer_input_before_loading_model(
    content: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(content)
    loader = MagicMock()
    monkeypatch.setattr("src.cli.load_model", loader)
    with pytest.raises(SystemExit) as exc:
        _invoke_cli(monkeypatch, "answer_dataset",
                    "--student_search_results_path", str(path))
    assert exc.value.code == 1
    assert "Traceback" not in capsys.readouterr().err
    loader.assert_not_called()


@pytest.mark.parametrize("content", ["{", "{}", "null", "[1]", "[]"])
def test_cli_rejects_corrupt_chunk_metadata(
    content: str, cli_paths: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    Path(cli_paths["chunks_path"]).write_text(content)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        _invoke_cli(monkeypatch, "search", "LoRA", *_path_flags(cli_paths))
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "Error:" in error
    assert "Traceback" not in error


@pytest.mark.parametrize("case", ["missing", "offsets", "overlong"])
def test_cli_rejects_invalid_sources_before_loading_model(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "source.txt"
    if case != "missing":
        source_path.write_text("text" if case == "offsets" else "x" * 2001)
    search_path = tmp_path / "search.json"
    search_path.write_text(StudentSearchResults(k=1, search_results=[
        MinimalSearchResults(question_id="q1", question="question",
                             retrieved_sources=[MinimalSource(
                                 file_path=str(source_path),
                                 first_character_index=0,
                                 last_character_index=2001,
                             )]),
    ]).model_dump_json())
    loader = MagicMock()
    monkeypatch.setattr("src.cli.load_model", loader)
    with pytest.raises(SystemExit) as exc:
        _invoke_cli(monkeypatch, "answer_dataset",
                    "--student_search_results_path", str(search_path))
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "Error:" in error
    assert "Traceback" not in error
    loader.assert_not_called()


def test_cli_empty_answer_dataset_does_not_load_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "empty.json"
    path.write_text(StudentSearchResults(k=5, search_results=[])
                    .model_dump_json())
    output_path = tmp_path / "answers.json"
    loader = MagicMock()
    monkeypatch.setattr("src.cli.load_model", loader)
    _invoke_cli(monkeypatch, "answer_dataset",
                "--student_search_results_path", str(path),
                "--output_path", str(output_path))
    output = StudentSearchResultsAndAnswer.model_validate_json(
        output_path.read_text()
    )
    assert output.k == 5
    assert output.search_results == []
    loader.assert_not_called()


def test_path_words_help_retrieval_without_changing_source_offsets(
    tmp_path: Path,
) -> None:
    sources = []
    for name in ("unrelated.py", "kv_cache.py"):
        path = tmp_path / name
        path.write_text("shared implementation text", encoding="utf-8")
        sources.append(MinimalSource(
            file_path=str(path), first_character_index=0,
            last_character_index=len("shared implementation text"),
        ))
    corpus = build_corpus(sources)
    index = build_index(corpus, str(tmp_path / "index"))
    assert "kv cache py" in corpus[1]
    assert search("kv cache", index, sources, k=1) == [sources[1]]
    assert sources[1].last_character_index == len("shared implementation text")


def test_chunking_skips_unreadable_files_and_keeps_readable_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "bad.py"
    good = tmp_path / "good.md"
    bad.write_text("pass")
    good.write_text("readable text")

    def read(path: str, size: int) -> list[MinimalSource]:
        if Path(path) == bad:
            raise PermissionError("unreadable")
        return chunk_file(path, size)

    monkeypatch.setattr("src.chunker.chunk_file", read)
    chunks = build_chunks(str(tmp_path))
    assert len(chunks) == 1
    assert Path(chunks[0].file_path).resolve() == good.resolve()
    assert "skipped" in capsys.readouterr().err


def test_chunking_falls_back_when_ast_rejects_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "example.py"
    text = "x = 1\n" * 20
    path.write_text(text)
    monkeypatch.setattr("src.chunker.ast.parse", MagicMock(
        side_effect=ValueError("invalid source"),
    ))
    assert chunk_file(str(path), 30) == chunk_text(str(path), text, 30)


@pytest.mark.parametrize("query", ["123", "True"])
def test_cli_normalizes_fire_literals_to_query_text(
    query: str, cli_paths: dict[str, str], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    capsys.readouterr()
    _invoke_cli(monkeypatch, "search", query, "--output_path",
                str(tmp_path / "literal.json"), *_path_flags(cli_paths))
    output = StudentSearchResults.model_validate_json(capsys.readouterr().out)
    assert output.search_results[0].question == query


def test_cli_preserves_empty_questions_in_dataset_pipeline(
    cli_paths: dict[str, str], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "questions.json"
    path.write_text(RagDataset(rag_questions=[
        UnansweredQuestion(question_id="empty", question="  "),
        UnansweredQuestion(question_id="valid", question="LoRA"),
    ]).model_dump_json())
    search_path = tmp_path / "search.json"
    _invoke_cli(monkeypatch, "search_dataset", "--dataset_path", str(path),
                "--output_path", str(search_path), *_path_flags(cli_paths))
    results = StudentSearchResults.model_validate_json(search_path.read_text())
    assert results.search_results[0].question_id == "empty"
    assert results.search_results[0].retrieved_sources == []
    assert results.search_results[1].retrieved_sources
    loader = MagicMock(return_value=(object(), object()))
    generator = MagicMock(return_value="LoRA answer")
    monkeypatch.setattr("src.cli.load_model", loader)
    monkeypatch.setattr("src.cli.generate", generator)
    answer_path = tmp_path / "answers.json"
    _invoke_cli(monkeypatch, "answer_dataset",
                "--student_search_results_path", str(search_path),
                "--output_path", str(answer_path))
    answers = StudentSearchResultsAndAnswer.model_validate_json(
        answer_path.read_text()
    )
    assert answers.search_results[0].answer == ""
    assert answers.search_results[1].answer == "LoRA answer"
    generator.assert_called_once()
    loader.assert_called_once()


def test_cli_rejects_bad_output_directory_before_loading_model(
    cli_paths: dict[str, str], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    loader = MagicMock()
    monkeypatch.setattr("src.cli.load_model", loader)
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        _invoke_cli(monkeypatch, "answer", "LoRA", "--output_path",
                    str(tmp_path), *_path_flags(cli_paths))
    assert exc.value.code == 1
    assert "output path is a directory" in capsys.readouterr().err
    loader.assert_not_called()
