# Individual Reflection — Lab 18

**Tên:** Nguyễn Tùng Lâm
**MSV:** 2A202600410
**Module phụ trách:** M5 (Enrichment) + Pipeline integration

---

## 1. Đóng góp kỹ thuật

- **M5 — Enrichment** (`src/m5_enrichment.py`): 4 techniques với
  OpenAI primary path + heuristic fallback.
  - `summarize_chunk`: gpt-4o-mini 2-3 câu, fallback extractive (2 câu
    đầu, đảm bảo terminator).
  - `generate_hypothesis_questions`: LLM tạo n câu hỏi mỗi chunk có
    thể trả lời, fallback heuristic (nhận diện số → "Bao nhiêu...",
    còn lại → "Điều gì về..."), strip leading numbering/dấu.
  - `contextual_prepend`: 1 câu mô tả vị trí + chủ đề chunk trong
    document, fallback regex tìm header markdown gần nhất.
  - `extract_metadata`: JSON `{topic, entities, category, language}`,
    fallback keyword-classify (4 categories: hr/it/finance/policy) +
    regex entity extraction (số + đơn vị: VND/ngày/giờ...).
  - `enrich_chunks`: orchestrator, hỗ trợ `methods=["full"]` hoặc subset.
- **Pipeline integration** (`src/pipeline.py`):
  - `build_pipeline`: ghép 4 stages của Nhân + Bân (chunk → enrich →
    index → rerank load) với timing per-stage để debug bottleneck.
  - **Hierarchical retrieve:** index child (semantic precision), trả
    parent (richer context cho LLM) thông qua `parent_index` stash trên
    search object. De-dup parent để không gửi trùng context.
  - `_generate_answer`: gọi gpt-4o-mini với system prompt tiếng Việt
    (force quote-from-context), fallback echo top-child (không phải
    parent — tránh "answer 1KB" làm rớt RAGAS metrics).
  - `evaluate_pipeline`: chạy 20 query test set → RAGAS → save report
    + failure_analysis. In `[Latency Breakdown]` mỗi run.
  - **Flag `ENABLE_ENRICHMENT=1`**: bonus M5 enrichment chỉ bật khi
    user opt-in; mặc định off để pipeline chạy nhanh trong CI.

**Số tests pass: M5 = 10/10**, pipeline end-to-end run sạch (37/37 toàn nhóm).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** **Contextual Prepend** (Anthropic, 2024) —
  prepend 1 câu mô tả vị trí + chủ đề chunk trước khi embed. Ý tưởng
  đẹp ở chỗ chunk con không còn "mồ côi" khi search; benchmark gốc
  báo giảm 49% retrieval failure. Đây là điểm giao thú vị giữa
  enrichment offline (one-time cost) và retrieval quality online.
- **Điều bất ngờ nhất:** Khi không có OPENAI_API_KEY, generator phải
  echo top context → answer là chunk gốc dài 1KB → `context_precision`
  bị penalize nặng vì lexical-proxy chia overlap cho `len(ctx)`. Phải
  echo **child** thay vì **parent** mới giảm noise. Đây là lesson
  chuẩn về "đo cái mình muốn vs đo cái dễ đo".
- **Kết nối với bài giảng:** Pipeline = orchestrator của 4 modules,
  nhưng cái khó nhất không phải gọi từng module mà là **truyền
  metadata xuyên stages** (parent_id từ M1 → search → rerank → lookup
  parent_index). Cùng pattern với data lineage / context propagation
  bên distributed tracing.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Tích hợp 4 modules viết bởi 3 người — interface
  thiếu thống nhất ban đầu (M1 trả `Chunk` dataclass, M2 nhận `dict`,
  M3 nhận `dict` với key `score`). Phải thống nhất schema chung trước
  khi pipeline chạy được end-to-end.
- **Cách giải quyết:** Define adapter trong `build_pipeline`: convert
  `Chunk` → `{text, metadata}` ngay sau M1; sau M2 convert
  `SearchResult` → `{text, score, metadata}` cho M3. Không sửa code
  của Nhân/Bân, để mỗi module test độc lập được.
- **Bug thực:** Ban đầu pipeline gửi **parent text** (1KB) làm answer
  fallback khi không có LLM key → RAGAS lexical phạt context_precision
  vì overlap chia cho document dài. Fix: echo **child text** (256 char,
  semantic-precise) thay vì parent. Đây là edge case của offline mode,
  không lộ ra với LLM thật.
- **Khó khăn M5:** OpenAI JSON parse hay fail vì model wrap output
  trong ```json fences. Phải strip fence regex trước khi `json.loads`,
  và có fallback heuristic nếu vẫn fail — nguyên tắc "không bao giờ
  để LLM call vỡ pipeline".

## 4. Nếu làm lại

- **Sẽ làm khác:** Define interface schema (Pydantic / TypedDict) ngay
  từ đầu giữa 3 modules thay vì discover bằng integration error. Tiết
  kiệm 1-2 giờ debug khi ráp pipeline.
- **Module muốn thử tiếp:**
  - **M5 contextual prepend** với LLM thật để verify 49% retrieval
    failure reduction (Anthropic benchmark) có đúng với corpus tiếng
    Việt không.
  - Thêm **HyDE** (Hypothetical Document Embeddings) ở query side —
    mirror của HyQA bên indexing side, bridge từ ngữ giữa câu hỏi và
    corpus.
  - **Streaming generation** trong pipeline để giảm perceived latency
    khi production deploy.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality (lazy-imports, fallbacks, type hints) | 4 |
| Teamwork (đồng bộ interface với Nhân & Bân) | 4 |
| Problem solving (debug parent vs child fallback echo) | 5 |
