from src.models import MinimalSource
import bm25s


def load_index(file_path: str) -> bm25s.BM25:
    """Load a previously saved BM25 index from disk."""
    return bm25s.BM25.load(file_path)


def search(
        query: str,
        retriever: bm25s.BM25,
        chunks: list[MinimalSource],
        k: int = 10) -> list[MinimalSource]:
    """Return the top-k chunks most relevant to the query via BM25."""
    # retrieve returns (results, scores); scores are discarded here
    results, _ = retriever.retrieve(bm25s.tokenize([query]), k=k)

    # results[0] contains the indices of the top-k chunks for the first query
    return [chunks[i] for i in results[0]]
