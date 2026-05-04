"""Tests for Module 1: Chunking."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m1_chunking import chunk_fixed_size, chunk_semantic, chunk_hierarchical, compare_strategies, load_documents, Chunk

TEXT = "# Nghỉ phép\n\nNhân viên được nghỉ 12 ngày/năm.\n\nNghỉ không lương tối đa 30 ngày.\n\nCần đăng ký trước 3 ngày."

def test_fixed_returns_chunks():
    assert len(chunk_fixed_size(TEXT, 50, 10)) > 0

def test_fixed_type():
    assert all(isinstance(c, Chunk) for c in chunk_fixed_size(TEXT, 50, 10))

def test_fixed_has_index():
    for c in chunk_fixed_size(TEXT, 50, 10, {"source": "t"}):
        assert "chunk_index" in c.metadata

def test_semantic_returns():
    assert len(chunk_semantic(TEXT, 0.5)) > 0

def test_hierarchical_parents_children():
    p, ch = chunk_hierarchical(TEXT, 80, 30)
    assert len(p) > 0 and len(ch) > 0

def test_hierarchical_parent_id():
    _, ch = chunk_hierarchical(TEXT, 80, 30)
    for c in ch:
        assert c.parent_id is not None

def test_hierarchical_valid_ids():
    p, ch = chunk_hierarchical(TEXT, 80, 30)
    pids = {x.metadata.get("parent_id") for x in p}
    for c in ch:
        assert c.parent_id in pids

def test_compare():
    docs = load_documents()
    if docs:
        r = compare_strategies(docs)
        assert "fixed" in r and "semantic" in r and "hierarchical" in r
