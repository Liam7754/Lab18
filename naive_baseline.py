"""Naive RAG Baseline — Chạy TRƯỚC để có scores so sánh."""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_fixed_size
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from config import NAIVE_COLLECTION, FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP


def main():
    print("=" * 60)
    print("NAIVE RAG BASELINE")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_fixed_size(doc["text"], FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP, doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} fixed-size chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []
    for item in test_set:
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]
        answers.append(contexts[0] if contexts else "Không tìm thấy.")
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print("\nNAIVE BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results.get(m, 0):.4f}")
    save_report(results, [], path="naive_baseline_report.json")
    print("\nDone! Now implement modules and run: python src/pipeline.py")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
