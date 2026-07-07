from src.models import MinimalSource
import bm25s


def get_chunk_text(chunk: MinimalSource) -> str:
    with open(chunk.file_path, encoding="utf-8", errors="ignore") as f:
        f.seek(chunk.first_character_index)
        text = f.read(chunk.last_character_index - chunk.first_character_index)
    return text


def build_corpus(chunks: list[MinimalSource]) -> list[str]:
    corpus = [get_chunk_text(chunk) for chunk in chunks]
    return corpus


def build_index(corpus: list[str]) -> bm25s.BM25:
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    retriever.save("data/processed/bm25_index")
    return retriever
