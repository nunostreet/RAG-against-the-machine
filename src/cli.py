import json
import os
import src.retriever as retriever
from src.chunker import build_chunks
from src.models import RagDataset, StudentSearchResults, MinimalSearchResults


class RAGSystem():
    def search(
            self,
            dataset_path: str,
            output_path: str,
            k: int = 10
    ) -> None:
        chunks = build_chunks("data/raw")
        index = retriever.load_index("data/processed/bm25_index")
        with open(dataset_path) as f:
            data = RagDataset.model_validate(json.load(f))

        search_result: list = []
        for question in data.rag_questions:
            results = retriever.search(
                question.question, index, chunks, k=k
                )
            search_result.append(MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=results
            ))
        output = StudentSearchResults(
            search_results=search_result,
            k=k
        )

        # Checking if folder doesn't exist already
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))
