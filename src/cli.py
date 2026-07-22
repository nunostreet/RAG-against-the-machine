import json
import os
import src.retriever as retriever
from tqdm import tqdm
from src.chunker import build_chunks
from src.generator import load_model, generate
from src.models import (
    RagDataset, StudentSearchResults, MinimalSearchResults,
    MinimalAnswer, StudentSearchResultsAndAnswer,
)

INDEX_PATH = "data/processed/bm25_index"
RAW_DIR = "data/raw"


class RAGSystem():
    """CLI entry point for the RAG system, exposed via python-fire."""

    def search_dataset(
            self,
            dataset_path: str,
            output_path: str,
            k: int = 10,
            max_chunk_size: int = 1000,
    ) -> None:
        """Run BM25 retrieval for every question in a dataset.

        Args:
            dataset_path:   Path to the input JSON dataset file.
            output_path:    Path where the output JSON will be written.
            k:              Number of chunks to retrieve per question.
            max_chunk_size: Maximum characters per chunk.
        """
        chunks = build_chunks(RAW_DIR, max_chunk_size=max_chunk_size)
        index = retriever.load_index(INDEX_PATH)

        with open(dataset_path) as f:
            data = RagDataset.model_validate(json.load(f))

        search_result: list = []
        for question in tqdm(data.rag_questions, desc="Searching"):
            results = retriever.search(
                question.question, index, chunks, k=k
            )
            search_result.append(MinimalSearchResults(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=results,
            ))

        output = StudentSearchResults(search_results=search_result, k=k)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))

    def answer_dataset(
            self,
            dataset_path: str,
            output_path: str,
            k: int = 10,
            max_chunk_size: int = 1000,
    ) -> None:
        """Retrieve chunks and generate an answer for every question.

        Args:
            dataset_path:   Path to the input JSON dataset file.
            output_path:    Path where the output JSON will be written.
            k:              Number of chunks to retrieve per question.
            max_chunk_size: Maximum characters per chunk.
        """
        chunks = build_chunks(RAW_DIR, max_chunk_size=max_chunk_size)
        index = retriever.load_index(INDEX_PATH)
        tokenizer, model = load_model()

        with open(dataset_path) as f:
            data = RagDataset.model_validate(json.load(f))

        answers: list = []
        for question in tqdm(data.rag_questions, desc="Answering"):
            results = retriever.search(
                question.question, index, chunks, k=k
            )
            answers.append(MinimalAnswer(
                question_id=question.question_id,
                question_str=question.question,
                retrieved_sources=results,
                answer=generate(
                    question.question, results, tokenizer, model
                ),
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=k
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output.model_dump_json(indent=2))

    def search(
            self,
            query: str,
            k: int = 10,
            max_chunk_size: int = 1000,
    ) -> None:
        """Search the index with a single query and print the top-k results.

        Args:
            query:          The search query string.
            k:              Number of chunks to retrieve.
            max_chunk_size: Maximum characters per chunk.
        """
        chunks = build_chunks(RAW_DIR, max_chunk_size=max_chunk_size)
        index = retriever.load_index(INDEX_PATH)
        results = retriever.search(query, index, chunks, k=k)
        for i, chunk in enumerate(results):
            print(f"[{i + 1}] {chunk.file_path} "
                  f"({chunk.first_character_index}"
                  f"–{chunk.last_character_index})")

    def answer(
            self,
            query: str,
            k: int = 10,
            max_chunk_size: int = 1000,
    ) -> None:
        """Answer a single query using retrieved context from the index.

        Args:
            query:          The question to answer.
            k:              Number of chunks to retrieve.
            max_chunk_size: Maximum characters per chunk.
        """
        chunks = build_chunks(RAW_DIR, max_chunk_size=max_chunk_size)
        index = retriever.load_index(INDEX_PATH)
        tokenizer, model = load_model()
        results = retriever.search(query, index, chunks, k=k)
        response = generate(query, results, tokenizer, model)
        print(response)
