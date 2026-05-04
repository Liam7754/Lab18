# Group Report — Lab 18: Production RAG

**Nhóm:** Liam, Nhân, Bân · **Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Trần Phan Văn Nhân | M1: Chunking            | ✅ | 13/13 |
| Trần Phan Văn Nhân | M2: Hybrid Search       | ✅ | 5/5  |
| Trần Văn Gia Bân  | M3: Reranking           | ✅ | 5/5  |
| Trần Văn Gia Bân  | M4: Evaluation          | ✅ | 4/4  |
| Nguyễn Tùng Lâm | M5: Enrichment (bonus)  | ✅ | 10/10 |
| Nguyễn Tùng Lâm | Pipeline integration    | ✅ | end-to-end runs |

Tổng: **37/37 tests pass** (`pytest tests/ -v`).

## Kết quả RAGAS

(Eval engine: `lexical_fallback` — môi trường offline. Xem chi tiết trong
`failure_analysis.md`.)

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness     | 1.0000 | 0.9851 | -0.0149 |
| Answer Relevancy | 0.1539 | 0.1855 | **+0.0316** |
| Context Precision| 0.0591 | 0.0447 | -0.0144 |
| Context Recall   | 1.0000 | 1.0000 | 0.0000  |

Pass rubric B2: **2 metrics ≥ 0.75** (faithfulness, context_recall).

## Latency Breakdown

| Stage | Time |
|-------|------|
| chunk        | 0.5 ms (one-time) |
| enrich       | disabled (bonus, optional) |
| index        | 18 ms (BM25 + in-memory dense, 17 chunks) |
| rerank load  | 0.5 ms |
| query        | ~2.4 ms / query |
| ragas (lexical) | ~5 ms |

Với deps thật: index ~5–10 s (model warmup), query ~150–300 ms (dense + cross-encoder).

## Key Findings

1. **Biggest improvement:** Production tăng `answer_relevancy` +3.2pp so với
   naive vì hybrid (BM25+RRF) tìm được câu chứa từ khoá hiếm tốt hơn pure
   dense, và rerank nâng câu sát query lên top.
2. **Biggest challenge:** Môi trường offline → không có Qdrant /
   sentence-transformers / RAGAS / OpenAI key. Phải thiết kế từng module
   với dual path: production stack khi đủ deps, fallback nhẹ khi không.
   Bài học: viết RAG production đáng để **lazy-import + try/except**
   các deps nặng để không vỡ cả pipeline khi 1 service down.
3. **Surprise finding:** Lexical-proxy eval phạt mạnh `context_precision`
   khi context dài (parent chunks ~1KB). Naive baseline ngẫu nhiên
   "thắng" precision dù chất lượng retrieval kém hơn — minh chứng vì sao
   RAGAS dùng LLM-judge chứ không dùng overlap thuần.

## Architectural notes

- **Hierarchical chunking** với `parent_size=2048 / child_size=256`: index
  child (chính xác semantic), trả parent (đủ context cho LLM). Đã sửa
  bug parent_id collision giữa các document (namespace bằng source).
- **HybridSearch** = BM25 (Vietnamese segmentation qua underthesea, có
  fallback) + Dense (bge-m3 + Qdrant, có in-memory fallback) + RRF
  fusion (k=60).
- **CrossEncoderReranker** thử FlagReranker → CrossEncoder →
  lexical-overlap fallback. Benchmark API ổn định cho mọi backend.
- **Eval** thử RAGAS thật trước (cần OPENAI_API_KEY), lexical fallback
  nếu không. Report ghi `engine` để minh bạch.
- **M5 enrichment** bonus: 4 techniques (summary, HyQA, contextual,
  metadata) với extractive/heuristic fallback khi không có OpenAI.
  Set `ENABLE_ENRICHMENT=1` để bật trong pipeline.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** xem bảng trên — F=0.99, CR=1.0
   ở production; AR cải thiện 3.2pp.
2. **Biggest win — module nào, tại sao:** **M2 Hybrid Search**. Naive
   chỉ dùng dense (hoặc hash-fallback) → miss câu hỏi có từ khoá hiếm
   ("WireGuard", "helpdesk@..."). BM25 bridge gap đó; RRF kết hợp ổn
   định không cần tune weight.
3. **Case study — Error Tree:** "Email helpdesk nội bộ là gì?". Output
   sai → Context đúng (recall=1) → Query OK → Fix Generator (cần LLM
   gen, hiện đang fallback echo).
4. **Next optimization nếu có thêm 1 giờ:**
   1) Cài đủ deps + bật OPENAI_API_KEY → bge-m3 + bge-reranker + LLM
      gen + RAGAS thật. 2) Bật M5 enrichment (contextual prepend).
   3) Tune `parent_size` 800 cho corpus nhỏ. 4) Regex post-processor
      cho câu hỏi pattern (email/số liệu).

## Bonus checklist

- [x] **Latency breakdown report** (+2): pipeline in `[Latency Breakdown]` mỗi run.
- [x] **Enrichment pipeline integrated** (+3): `src/m5_enrichment.py` với
  4 techniques + flag `ENABLE_ENRICHMENT=1` trong pipeline.
- [ ] **RAGAS Faithfulness ≥ 0.85** (+5): đạt 0.9851 ở lexical-proxy,
  cần xác nhận bằng RAGAS thật.
