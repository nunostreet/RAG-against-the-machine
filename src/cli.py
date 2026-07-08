import json
from src.chunker import build_chunks
from src.retriever import load_index, search
from src.models import RagDataset, StudentSearchResults, MinimalSearchResults


class RAGSystem():
    def search(
            self,
            dataset_path: str,
            output_path: str,
            k: int = 10
    ) -> None:
        with open(dataset_path) as f:
            data = RagDataset.model_validate(json.load(f))
