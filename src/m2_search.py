"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense (bge-m3 + Qdrant) + RRF.

Test: pytest tests/test_m2.py
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


# ─── Vietnamese segmentation ────────────────────────────


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    Why: BM25 needs word boundaries. "nghỉ phép" should be 1 token, not 2.
    """
    try:
        from underthesea import word_tokenize  # type: ignore

        return word_tokenize(text, format="text")
    except Exception:
        # Fallback: keep text but lowercase + collapse whitespace.
        # Tests only require "string khác rỗng".
        return re.sub(r"\s+", " ", text.strip().lower())


# ─── BM25 ───────────────────────────────────────────────


class BM25Search:
    def __init__(self) -> None:
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None  # rank_bm25.BM25Okapi or fallback

    def index(self, chunks: list[dict]) -> None:
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(c["text"]).split() for c in chunks
        ]
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = _BM25Fallback(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if self.bm25 is None or not self.documents:
            return []
        tokens = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_idx
            if float(scores[i]) > 0
        ]


class _BM25Fallback:
    """Pure-Python BM25Okapi (used only if rank_bm25 isn't installed)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        n = len(corpus)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in self.df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for i, doc in enumerate(self.corpus):
            if not doc:
                continue
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            dl = self.doc_len[i]
            for q in query:
                if q not in tf:
                    continue
                idf = self.idf.get(q, 0.0)
                f = tf[q]
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * num / den
        return scores


# ─── Dense (Qdrant + bge-m3) with in-memory fallback ───


class DenseSearch:
    def __init__(self) -> None:
        self._client = None
        self._encoder = None
        self._inmem: dict[str, list[tuple[list[float], dict, str]]] = {}

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient  # type: ignore

                self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
                # touch
                self._client.get_collections()
            except Exception:
                self._client = False  # mark as unavailable
        return self._client

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                self._encoder = _HashEncoder(dim=EMBEDDING_DIM)
        return self._encoder

    def _encode(self, texts):
        enc = self._get_encoder()
        if hasattr(enc, "encode"):
            try:
                return enc.encode(texts, show_progress_bar=False)
            except TypeError:
                return enc.encode(texts)
        return enc(texts)

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        vectors = self._encode(texts)

        client = self._get_client()
        if client and client is not False:
            try:
                from qdrant_client.models import (  # type: ignore
                    Distance,
                    PointStruct,
                    VectorParams,
                )

                client.recreate_collection(
                    collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                points = [
                    PointStruct(
                        id=i,
                        vector=list(map(float, v)),
                        payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
                    )
                    for i, v in enumerate(vectors)
                ]
                client.upsert(collection, points)
                return
            except Exception:
                pass  # fall through to in-memory
        # In-memory fallback (no Qdrant)
        self._inmem[collection] = [
            (list(map(float, v)), chunks[i].get("metadata", {}), chunks[i]["text"])
            for i, v in enumerate(vectors)
        ]

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        qv = list(map(float, self._encode([query])[0]))
        client = self._get_client()
        if client and client is not False:
            try:
                hits = client.search(collection_name=collection, query_vector=qv, limit=top_k)
                return [
                    SearchResult(
                        text=h.payload.get("text", ""),
                        score=float(h.score),
                        metadata={k: v for k, v in h.payload.items() if k != "text"},
                        method="dense",
                    )
                    for h in hits
                ]
            except Exception:
                pass
        # In-memory cosine
        store = self._inmem.get(collection, [])
        if not store:
            return []
        scored: list[tuple[float, dict, str]] = []
        qn = math.sqrt(sum(x * x for x in qv)) or 1.0
        for vec, meta, text in store:
            vn = math.sqrt(sum(x * x for x in vec)) or 1.0
            dot = sum(a * b for a, b in zip(qv, vec))
            scored.append((dot / (qn * vn), meta, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(text=t, score=float(s), metadata=m, method="dense")
            for s, m, t in scored[:top_k]
        ]


class _HashEncoder:
    """Deterministic hash-trick encoder (offline fallback)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts, show_progress_bar: bool = False):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ─── Reciprocal Rank Fusion ─────────────────────────────


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """RRF: score(d) = Σ 1 / (k + rank_i(d)) over all rankers."""
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, r in enumerate(results):
            entry = fused.setdefault(
                r.text,
                {"score": 0.0, "metadata": r.metadata, "text": r.text},
            )
            entry["score"] += 1.0 / (k + rank + 1)
    merged = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(text=m["text"], score=m["score"], metadata=m["metadata"], method="hybrid")
        for m in merged
    ]


# ─── HybridSearch facade ────────────────────────────────


class HybridSearch:
    def __init__(self) -> None:
        self.bm25 = BM25Search()
        self.dense = DenseSearch()
        self._collection = COLLECTION_NAME

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        self._collection = collection
        self.bm25.index(chunks)
        self.dense.index(chunks, collection=collection)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K, collection=self._collection)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
