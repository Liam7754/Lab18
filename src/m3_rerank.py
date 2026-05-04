"""Module 3: Reranking — Cross-encoder top-N → top-k + latency benchmark.

Test: pytest tests/test_m3.py
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


# ─── Cross-encoder reranker ─────────────────────────────


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None
        self._kind = None  # "flag" | "ce" | "fallback"

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker  # type: ignore

            self._model = FlagReranker(self.model_name, use_fp16=True)
            self._kind = "flag"
            return self._model
        except Exception:
            pass
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name)
            self._kind = "ce"
            return self._model
        except Exception:
            pass
        # Lexical-overlap fallback (works offline; good enough for tests)
        self._model = _LexicalReranker()
        self._kind = "fallback"
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, d["text"]) for d in documents]

        if self._kind == "flag":
            scores = model.compute_score(pairs)
        elif self._kind == "ce":
            scores = model.predict(pairs)
        else:
            scores = model.score(pairs)

        # Normalise to list of floats
        scores = [float(s) for s in scores]

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)[:top_k]
        return [
            RerankResult(
                text=d["text"],
                original_score=float(d.get("score", 0.0)),
                rerank_score=float(score),
                metadata=d.get("metadata", {}),
                rank=i,
            )
            for i, (score, d) in enumerate(ranked)
        ]


class _LexicalReranker:
    """Simple BM25-like overlap scorer used when neural models are unavailable."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for q, d in pairs:
            q_terms = set(re.findall(r"\w+", q.lower()))
            d_terms = re.findall(r"\w+", d.lower())
            if not q_terms or not d_terms:
                out.append(0.0)
                continue
            d_count = {}
            for t in d_terms:
                d_count[t] = d_count.get(t, 0) + 1
            overlap = sum(d_count.get(t, 0) for t in q_terms)
            score = overlap / (1.0 + math.log(1 + len(d_terms)))
            out.append(score)
        return out


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""

    def __init__(self) -> None:
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest  # type: ignore

            if self._model is None:
                self._model = Ranker()
            passages = [
                {"id": i, "text": d["text"], "meta": d.get("metadata", {})}
                for i, d in enumerate(documents)
            ]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            return [
                RerankResult(
                    text=r["text"],
                    original_score=float(documents[r["id"]].get("score", 0.0)),
                    rerank_score=float(r["score"]),
                    metadata=r.get("meta", {}),
                    rank=i,
                )
                for i, r in enumerate(results[:top_k])
            ]
        except Exception:
            # Fall back to cross-encoder reranker (which itself has a fallback)
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


# ─── Benchmark ──────────────────────────────────────────


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000.0)  # ms
    return {
        "avg_ms": sum(times) / max(len(times), 1),
        "min_ms": min(times),
        "max_ms": max(times),
        "n_runs": len(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
    print("Benchmark:", benchmark_reranker(reranker, query, docs, n_runs=3))
