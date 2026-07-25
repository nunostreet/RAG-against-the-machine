from typing import Any
from src.models import MinimalSource
from src.indexer import get_chunk_text
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_model(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the tokenizer and model from HuggingFace cache."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except OSError:
        raise OSError(
            f"Model '{model_name}' not found in HuggingFace cache. "
            "Ensure the model is downloaded before running."
        )
    return tokenizer, model


def generate(
    question: str,
    chunks: list[MinimalSource],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> str:
    """Generate an answer for a question given retrieved chunks as context."""
    # Concatenate chunk texts into a single context string
    context = "\n\n".join(get_chunk_text(chunk) for chunk in chunks)

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

    # Tokenize using Qwen3 chat format; disable thinking mode to save tokens
    # Any annotation needed: return type depends on return_tensors value
    inputs: Any = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )

    output_ids: Any = model.generate(  # type: ignore[operator]
        inputs, max_new_tokens=512
    )

    # Slice off the prompt tokens — model returns prompt + answer together
    prompt_len = len(inputs[0])
    answer: str = tokenizer.decode(  # type: ignore[assignment]
        output_ids[0][prompt_len:],
        skip_special_tokens=True,
    )
    return answer
