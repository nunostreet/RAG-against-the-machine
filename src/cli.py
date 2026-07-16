import json
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
        with open(dataset_path) as f:
            data = RagDataset.model_validate(json.load(f))
            chunks = build_chunks("data/raw")
            index = retriever.load_index("data/processed/bm25_index")
            for question in data.rag_questions:
                results = retriever.search(
                    question.question, index, chunks, k=k
                    )
                
""" 
class MinimalSearchResults(BaseModel):
    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource] """