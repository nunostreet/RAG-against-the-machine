from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "Qwen/Qwen3-0.6B") -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
