"""Answer generation with Qwen.

Generation is separated from retrieval so dataset search can be evaluated
without loading the language model. This is important because retrieval should
be fast and deterministic, while generation is slower and model-dependent.
"""

from typing import Protocol, cast

import torch

from src.models import MinimalSource
from src.indexer import get_chunk_text
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    BatchEncoding, PretrainedConfig, PreTrainedTokenizerBase,
)

MAX_CONTEXT_CHARS = 12000
MAX_NEW_TOKENS = 512


class GenerativeModel(Protocol):
    """The model interface used by the generator, including tensor outputs."""

    config: PretrainedConfig

    @property
    def device(self) -> torch.device:
        """Return the device on which model inputs must be placed."""
        ...

    def eval(self) -> object:
        """Switch the model to evaluation mode."""
        ...

    def generate(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        max_new_tokens: int, do_sample: bool,
    ) -> torch.Tensor:
        """Generate sequences containing the prompt and continuation."""
        ...


def load_model(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> tuple[PreTrainedTokenizerBase, GenerativeModel]:
    """Load the tokenizer and model from the HuggingFace cache.

    The subject requires `Qwen/Qwen3-0.6B`. The code deliberately does not
    download the model here; if it is missing, the caller gets a clear error.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True
        )
        # AutoModel selects a concrete class dynamically; declare the interface
        # required here rather than relying on its incomplete base-class types.
        model = cast(GenerativeModel, AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True
        ))
    except OSError:
        raise OSError(
            f"Model '{model_name}' not found in HuggingFace cache. "
            "Ensure the model is downloaded before running."
        )
    model.eval()
    return tokenizer, model


def build_context(
    chunks: list[MinimalSource],
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Build a bounded context string from retrieved chunks.

    Each chunk is prefixed with its source location. This gives the model
    useful provenance while keeping the exact sources available in the
    structured output.
    """
    if max_context_chars < 0:
        raise ValueError("max_context_chars must not be negative")
    context_parts: list[str] = []
    used_chars = 0

    for index, chunk in enumerate(chunks, start=1):
        text = get_chunk_text(chunk).strip()
        if not text:
            continue

        header = (
            f"[Source {index}: {chunk.file_path}:"
            f"{chunk.first_character_index}-{chunk.last_character_index}]\n"
        )
        separator_chars = 2 if context_parts else 0
        available_chars = (
            max_context_chars - used_chars - len(header) - separator_chars
        )
        if available_chars <= 0:
            break

        # Truncate only the final included source; earlier sources stay intact.
        if len(text) > available_chars:
            text = text[:available_chars]

        part = f"{header}{text}"
        context_parts.append(part)
        used_chars += len(part) + separator_chars

    return "\n\n".join(context_parts)


def _encode_prompt(
    question: str,
    context: str,
    tokenizer: PreTrainedTokenizerBase,
) -> BatchEncoding:
    """Encode the complete chat prompt with named tensors for generation."""
    messages = [
        {
            "role": "system",
            "content": (
                "Answer concisely using only the provided context. "
                "If the context is insufficient, say so."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]
    return cast(BatchEncoding, tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def _fit_prompt(
    question: str,
    context: str,
    tokenizer: PreTrainedTokenizerBase,
    token_budget: int,
) -> BatchEncoding:
    """Fit context to the token budget while preserving the whole question."""
    inputs = _encode_prompt(question, context, tokenizer)
    if inputs["input_ids"].shape[-1] <= token_budget:
        return inputs

    fitted = _encode_prompt(question, "", tokenizer)
    if fitted["input_ids"].shape[-1] > token_budget:
        raise ValueError("question and chat template exceed the token budget")

    # Count the actual templated prompt at every step, including its headers.
    lower, upper = 0, len(context)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        candidate = _encode_prompt(question, context[:middle], tokenizer)
        if candidate["input_ids"].shape[-1] <= token_budget:
            fitted = candidate
            lower = middle
        else:
            upper = middle - 1
    return fitted


def generate(
    question: str,
    chunks: list[MinimalSource],
    tokenizer: PreTrainedTokenizerBase,
    model: GenerativeModel,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate an answer for a question given retrieved chunks as context.

    The prompt explicitly tells the model to answer from provided context only.
    Context is trimmed by token count after applying the chat template, with
    room reserved for the response within the model's context window.
    The returned string is just the generated answer; the CLI wraps it together
    with the sources in a `MinimalAnswer` model.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    limits = [
        limit for limit in (
            getattr(model.config, "max_position_embeddings", None),
            tokenizer.model_max_length,
        )
        if isinstance(limit, int) and 0 < limit < 10**9
    ]
    if not limits:
        raise ValueError("model and tokenizer do not specify a context window")
    token_budget = min(limits) - max_new_tokens
    if token_budget <= 0:
        raise ValueError("response token budget exhausts the context window")

    context = build_context(chunks, max_context_chars=max_context_chars)
    inputs = _fit_prompt(question, context, tokenizer, token_budget)
    inputs = inputs.to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    return str(tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True
    )).strip()
