"""Answer generation with Qwen.

Generation is separated from retrieval so dataset search can be evaluated
without loading the language model. This is important because retrieval should
be fast and deterministic, while generation is slower and model-dependent.
"""

from typing import Any

from src.models import MinimalSource
from src.indexer import get_chunk_text
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

MAX_CONTEXT_CHARS = 12000
MAX_NEW_TOKENS = 512


def load_model(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the tokenizer and model from the HuggingFace cache.

    The subject requires `Qwen/Qwen3-0.6B`. The code deliberately does not
    download the model here; if it is missing, the caller gets a clear error.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
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
        available_chars = max_context_chars - used_chars - len(header)
        if available_chars <= 0:
            break

        # Truncate only the final included source; earlier sources stay intact.
        if len(text) > available_chars:
            text = text[:available_chars]

        part = f"{header}{text}"
        context_parts.append(part)
        used_chars += len(part)

    return "\n\n".join(context_parts)


def generate(
    question: str,
    chunks: list[MinimalSource],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Generate an answer for a question given retrieved chunks as context.

    The prompt explicitly tells the model to answer from provided context only.
    The returned string is just the generated answer; the CLI wraps it together
    with the sources in a `MinimalAnswer` model.
    """
    context = build_context(chunks, max_context_chars=max_context_chars)

    messages = [
        {
            "role": "system",
            "content": "Answer the question using only the provided context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    # Qwen3 supports `enable_thinking=False`; the fallback keeps compatibility
    # with tokenizer versions that do not expose that argument
    try:
        inputs: Any = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    output_ids: Any = model.generate(  # type: ignore[operator]
        inputs, max_new_tokens=MAX_NEW_TOKENS
    )

    # The generated tensor contains prompt + answer, so remove the prompt part.
    prompt_len = len(inputs[0])
    answer: str = tokenizer.decode(  # type: ignore[assignment]
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
    )
    return answer
