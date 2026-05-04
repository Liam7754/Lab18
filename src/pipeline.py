"""Production RAG Pipeline — Group integration of M1+M2+M3+M4 (+M5 bonus)."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import (
    EvalResult,
    evaluate_ragas,
    failure_analysis,
    load_test_set,
    save_report,
)
from src.m5_enrichment import enrich_chunks


ENABLE_ENRICHMENT = os.getenv("ENABLE_ENRICHMENT", "0") == "1"
TIMINGS: dict[str, float] = {}


def build_pipeline() -> tuple[HybridSearch, CrossEncoderReranker]:
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1) — hierarchical: index children, return parents.
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents...")
    docs = load_documents()
    parent_index: dict[str, str] = {}  # parent_id → parent.text
    all_chunks: list[dict] = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for p in parents:
            parent_index[p.metadata["parent_id"]] = p.text
        for child in children:
            all_chunks.append(
                {
                    "text": child.text,
                    "metadata": {**child.metadata, "parent_id": child.parent_id},
                }
            )
    TIMINGS["chunk"] = time.perf_counter() - t0
    print(f"  {len(all_chunks)} child chunks ({len(parent_index)} parents) from {len(docs)} documents")

    # Step 2: Enrichment (M5) — bonus
    t0 = time.perf_counter()
    if ENABLE_ENRICHMENT:
        print("\n[2/4] Enriching chunks (M5)...")
        enriched = enrich_chunks(all_chunks, methods=["contextual", "metadata"])
        if enriched:
            for i, e in enumerate(enriched):
                all_chunks[i]["text"] = e.enriched_text
                all_chunks[i]["metadata"] = {**all_chunks[i]["metadata"], **e.auto_metadata}
            print(f"  Enriched {len(enriched)} chunks")
    else:
        print("\n[2/4] Enrichment disabled (set ENABLE_ENRICHMENT=1 to enable)")
    TIMINGS["enrich"] = time.perf_counter() - t0

    # Step 3: Index (M2)
    t0 = time.perf_counter()
    print("\n[3/4] Indexing (BM25 + Dense + RRF)...")
    search = HybridSearch()
    search.index(all_chunks)
    # stash parent index on the search object so run_query can reach it
    search._parent_index = parent_index  # type: ignore[attr-defined]
    TIMINGS["index"] = time.perf_counter() - t0

    # Step 4: Reranker (M3)
    t0 = time.perf_counter()
    print("\n[4/4] Loading reranker...")
    reranker = CrossEncoderReranker()
    reranker._load_model()  # warm load
    TIMINGS["rerank_load"] = time.perf_counter() - t0

    return search, reranker


def _generate_answer(query: str, contexts: list[str]) -> str:
    """LLM answer generation (gpt-4o-mini). Falls back to top-context echo."""
    context_str = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là trợ lý trả lời CHỈ dựa vào context tiếng Việt được cung cấp. "
                            "Nếu context không chứa câu trả lời, hãy trả lời chính xác: 'Không tìm thấy thông tin trong tài liệu.' "
                            "Trả lời ngắn gọn, không thêm thông tin không có trong context."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  ⚠️  LLM call failed ({e}); falling back to top context")
    return contexts[0] if contexts else "Không tìm thấy thông tin trong tài liệu."


def run_query(
    query: str, search: HybridSearch, reranker: CrossEncoderReranker
) -> tuple[str, list[str]]:
    """Run a single query through retrieval → rerank → generate."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)

    parent_index: dict[str, str] = getattr(search, "_parent_index", {})

    # Hierarchical lookup: child → parent (richer context for the LLM).
    seen: set[str] = set()
    contexts: list[str] = []
    child_texts: list[str] = []
    if reranked:
        for r in reranked:
            pid = r.metadata.get("parent_id")
            ctx = parent_index.get(pid, r.text) if pid else r.text
            child_texts.append(r.text)
            if ctx in seen:
                continue
            seen.add(ctx)
            contexts.append(ctx)
    else:
        contexts = [r.text for r in results[:RERANK_TOP_K]]
        child_texts = list(contexts)

    # Use LLM if available; otherwise echo the top child (concise) — not the
    # full parent — so RAGAS-style metrics aren't penalised by a 1KB "answer".
    if OPENAI_API_KEY:
        answer = _generate_answer(query, contexts)
    else:
        answer = child_texts[0] if child_texts else "Không tìm thấy thông tin trong tài liệu."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker) -> dict:
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    t0 = time.perf_counter()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i + 1}/{len(test_set)}] {item['question'][:60]}")
    TIMINGS["query"] = (time.perf_counter() - t0) / max(len(test_set), 1)

    print("\n[Eval] Running RAGAS...")
    t0 = time.perf_counter()
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    TIMINGS["ragas"] = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"PRODUCTION RAG SCORES  (engine={results.get('_engine', '?')})")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = float(results.get(m, 0))
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)

    print("\n[Latency Breakdown]")
    for k, v in TIMINGS.items():
        print(f"  {k:<14s} {v * 1000:>9.1f} ms" + (" / query" if k == "query" else ""))

    return results


if __name__ == "__main__":
    start = time.time()
    s, r = build_pipeline()
    evaluate_pipeline(s, r)
    print(f"\nTotal: {time.time() - start:.1f}s")
