"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, and structure-aware chunking.
So sánh với basic chunking (baseline).

Test: pytest tests/test_m1.py
"""

from __future__ import annotations

import glob
import math
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown files from data/."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking ───────────────────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Basic paragraph chunking — baseline."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(
                Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)})
            )
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)})
        )
    return chunks


# ─── Embedding helper with fallback ─────────────────────


def _encode_sentences(sentences: list[str]):
    """Try sentence-transformers; fall back to char n-gram TF for offline tests."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(sentences, show_progress_bar=False)
    except Exception:
        # Lightweight deterministic fallback: char-3gram bag-of-features
        import numpy as np

        vocab: dict[str, int] = {}
        rows: list[dict[int, float]] = []
        for s in sentences:
            row: dict[int, float] = {}
            tokens = re.findall(r"\w+", s.lower())
            grams = []
            for tok in tokens:
                grams.append(tok)
                for i in range(len(tok) - 2):
                    grams.append(tok[i : i + 3])
            for g in grams:
                idx = vocab.setdefault(g, len(vocab))
                row[idx] = row.get(idx, 0.0) + 1.0
            rows.append(row)
        dim = max(1, len(vocab))
        mat = np.zeros((len(sentences), dim), dtype="float32")
        for i, row in enumerate(rows):
            for j, v in row.items():
                mat[i, j] = v
        # L2 normalise
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def _cosine(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─── Strategy 1: Semantic Chunking ──────────────────────


def chunk_semantic(
    text: str, threshold: float = SEMANTIC_THRESHOLD, metadata: dict | None = None
) -> list[Chunk]:
    """Group consecutive sentences by similarity → don't cut mid-idea."""
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    embeddings = _encode_sentences(sentences)

    chunks: list[Chunk] = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(
                Chunk(
                    text=" ".join(current_group),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
                )
            )
            current_group = []
        current_group.append(sentences[i])
    if current_group:
        chunks.append(
            Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"},
            )
        )
    return chunks


# ─── Strategy 2: Hierarchical Chunking ─────────────────


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Parent (large) → Children (small). Index children, return parent for context."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Namespace parent_id by source so different documents don't collide.
    source = str(metadata.get("source", "doc"))

    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > parent_size and current:
            pid = f"{source}::parent_{len(parents)}"
            parents.append(
                Chunk(
                    text=current.strip(),
                    metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
                )
            )
            current = ""
        current += para + "\n\n"
    if current.strip():
        pid = f"{source}::parent_{len(parents)}"
        parents.append(
            Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            )
        )

    children: list[Chunk] = []
    overlap = max(child_size // 5, 16)
    for parent in parents:
        pid = parent.metadata["parent_id"]
        ptext = parent.text
        if len(ptext) <= child_size:
            children.append(
                Chunk(
                    text=ptext,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid,
                )
            )
            continue
        start = 0
        while start < len(ptext):
            end = min(start + child_size, len(ptext))
            child_text = ptext[start:end].strip()
            if child_text:
                children.append(
                    Chunk(
                        text=child_text,
                        metadata={**metadata, "chunk_type": "child"},
                        parent_id=pid,
                    )
                )
            if end >= len(ptext):
                break
            start = end - overlap
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ──────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Split by markdown headers → each chunk = 1 logical section."""
    metadata = metadata or {}
    parts = re.split(r"(^#{1,6}\s+.+$)", text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""
    for part in parts:
        if part is None:
            continue
        if re.match(r"^#{1,6}\s+", part):
            if current_content.strip() or current_header:
                chunks.append(
                    Chunk(
                        text=f"{current_header}\n{current_content}".strip(),
                        metadata={
                            **metadata,
                            "section": current_header.lstrip("# ").strip(),
                            "strategy": "structure",
                            "chunk_index": len(chunks),
                        },
                    )
                )
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
    if current_content.strip() or current_header:
        chunks.append(
            Chunk(
                text=f"{current_header}\n{current_content}".strip(),
                metadata={
                    **metadata,
                    "section": current_header.lstrip("# ").strip(),
                    "strategy": "structure",
                    "chunk_index": len(chunks),
                },
            )
        )
    # Drop empties (e.g., when the file starts with a header)
    chunks = [c for c in chunks if c.text.strip()]
    return chunks


# ─── A/B Test ───────────────────────────────────────────


def _stats(chunks: list[Chunk]) -> dict:
    if not chunks:
        return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    lengths = [len(c.text) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_length": int(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def compare_strategies(documents: list[dict]) -> dict:
    """Run all 4 strategies and print comparison table."""
    aggregated: dict[str, list[Chunk]] = {
        "basic": [],
        "semantic": [],
        "hierarchical": [],
        "structure": [],
    }
    parents_total: list[Chunk] = []

    for doc in documents:
        text, meta = doc["text"], doc.get("metadata", {})
        aggregated["basic"].extend(chunk_basic(text, metadata=meta))
        aggregated["semantic"].extend(chunk_semantic(text, metadata=meta))
        parents, children = chunk_hierarchical(text, metadata=meta)
        parents_total.extend(parents)
        aggregated["hierarchical"].extend(children)
        aggregated["structure"].extend(chunk_structure_aware(text, metadata=meta))

    results = {name: _stats(chunks) for name, chunks in aggregated.items()}
    results["hierarchical"]["num_parents"] = len(parents_total)

    header = f"{'Strategy':<15} {'Chunks':>8} {'Avg Len':>9} {'Min':>6} {'Max':>6}"
    print(header)
    print("-" * len(header))
    for name, stats in results.items():
        n = stats["num_chunks"]
        if name == "hierarchical":
            label = f"{stats.get('num_parents', 0)}p/{n}c"
            print(f"{name:<15} {label:>8} {stats['avg_length']:>9} {stats['min_length']:>6} {stats['max_length']:>6}")
        else:
            print(f"{name:<15} {n:>8} {stats['avg_length']:>9} {stats['min_length']:>6} {stats['max_length']:>6}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents\n")
    compare_strategies(docs)
