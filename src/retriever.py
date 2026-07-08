from src.models import MinimalSource
import bm25s


def load_index(file_path: str):
    return bm25s.BM25.load(file_path)


def search(
        query: str,
        retriever: bm25s.BM25,
        chunks: list[MinimalSource],
        k: int = 10) -> list[MinimalSource]:

    results, _ = retriever.retrieve(bm25s.tokenize([query]), k=k)

    return [chunks[i] for i in results[0]]
