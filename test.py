from src.cli import RAGSystem

rag = RAGSystem()
rag.search(
    dataset_path="provided/datasets_public/public/UnansweredQuestions/dataset_code_public.json",
    output_path="data/output/search_results/dataset_code_public.json",
    k=10
)
