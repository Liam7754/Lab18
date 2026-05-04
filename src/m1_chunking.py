"""Module 1: Chunking Strategies — Implement 3 strategies + A/B test."""

import os, sys, glob
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP,
                    HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE, SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


def chunk_fixed_size(text: str, chunk_size: int = FIXED_CHUNK_SIZE,
                     overlap: int = FIXED_CHUNK_OVERLAP, metadata: dict | None = None) -> list[Chunk]:
    """Split text into fixed-size chunks with overlap."""
    metadata = metadata or {}
    # TODO: Implement fixed-size chunking
    # 1. start = 0, i = 0
    # 2. While start < len(text):
    #      end = start + chunk_size
    #      chunk_text = text[start:end]
    #      Create Chunk(text=chunk_text, metadata={**metadata, "chunk_index": i})
    #      start += chunk_size - overlap
    #      i += 1
    # 3. Return list of Chunks
    return []


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """Split text by sentence similarity."""
    metadata = metadata or {}
    # TODO: Implement semantic chunking
    # 1. Split text into sentences (split on "\n\n" or ". ")
    # 2. from sentence_transformers import SentenceTransformer
    #    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast for chunking
    # 3. Encode all sentences: embeddings = model.encode(sentences)
    # 4. For i in range(1, len(sentences)):
    #      sim = cosine_similarity(embeddings[i-1], embeddings[i])
    #      if sim < threshold: start new chunk
    # 5. Group consecutive sentences into Chunk objects
    return []


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent-child chunk hierarchy."""
    metadata = metadata or {}
    # TODO: Implement hierarchical chunking
    # 1. Split text into parents of parent_size chars (no overlap)
    # 2. For each parent (index p):
    #      pid = f"parent_{p}"
    #      parent = Chunk(text=parent_text, metadata={**metadata, "chunk_type": "parent", "parent_id": pid})
    # 3. Split each parent into children of child_size chars
    #      child = Chunk(text=child_text, metadata={**metadata, "chunk_type": "child"}, parent_id=pid)
    # 4. Return (parents_list, children_list)
    return [], []


def compare_strategies(documents: list[dict]) -> dict:
    """Run all 3 strategies and compare. Returns stats dict."""
    # TODO: Implement comparison
    # 1. For each doc, run chunk_fixed_size, chunk_semantic, chunk_hierarchical
    # 2. Collect: num_chunks, avg_length, min_length, max_length
    # 3. Print comparison table
    # 4. Return {"fixed": {...}, "semantic": {...}, "hierarchical": {...}}
    return {}


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
