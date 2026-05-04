# Individual Reflection — Lab 18

**Tên:** Trần Phan Văn Nhân
**MSSV:** 2A202600301
**Module phụ trách:** M1 (Chunking) + M2 (Hybrid Search)

---

## 1. Đóng góp kỹ thuật

- **M1 — Chunking** (`src/m1_chunking.py`): 4 strategies + so sánh.
  - `chunk_basic`: paragraph-aggregating baseline (split `\n\n`, gộp
    đến `chunk_size=500`). Dùng làm baseline để các strategy khác có
    điểm so sánh.
  - `chunk_semantic`: split câu bằng regex `(?<=[.!?])\s+|\n\n` →
    encode → tách khi cosine giữa 2 câu liên tiếp `< threshold` (0.85).
    Dùng `sentence-transformers` (all-MiniLM-L6-v2) khi có; fallback
    char-3gram bag-of-words + L2 normalise để vẫn deterministic offline.
  - `chunk_hierarchical`: parent (paragraph-aggregating ≤ `parent_size=2048`)
    + sliding-window children (`child_size=256`, overlap = `child_size//5 = 51`).
    **Bug fix quan trọng:** namespace `parent_id` theo
    `f"{source}::parent_{i}"` để tránh collision giữa 3 documents
    (trước đó "parent_0" trùng → last write wins).
  - `chunk_structure_aware`: split markdown headers `^#{1,6}\s+` qua
    regex, pair header + content, lưu `section` vào metadata.
  - `compare_strategies`: in bảng `Stats (chunks / avg / min / max)`,
    riêng hierarchical hiện `Xp/Yc` (parents / children).
- **M2 — Hybrid Search** (`src/m2_search.py`):
  - `segment_vietnamese`: gọi `underthesea.word_tokenize` để giữ
    "nghỉ phép" thành 1 token thay vì 2; lowercase trước trả về để
    corpus và query tokenise đồng nhất; fallback whitespace+lowercase
    nếu underthesea không cài được.
  - `BM25Search`: ưu tiên `rank_bm25.BM25Okapi`. **Edge case fix:** nếu
    segmented query không match (max score ≤ 0), retry với
    `re.findall(r"\w+", query.lower())` — bridge gap khi underthesea
    join token khác với corpus index time.
  - `_BM25Fallback`: pure-Python BM25Okapi (k1=1.5, b=0.75) — implement
    IDF, TF, doc length normalisation. Dùng khi không cài được rank_bm25.
  - `DenseSearch`: bge-m3 + Qdrant primary; nếu Qdrant down → in-memory
    cosine; nếu không có sentence-transformers → `_HashEncoder` (md5
    hash trick, dim=1024). 3 tầng fallback đảm bảo runnable mọi env.
  - `reciprocal_rank_fusion`: k=60, fuse theo `text` key, `score(d) =
    Σ 1 / (k + rank_i(d) + 1)`. Trả về `method="hybrid"` để debug.
  - `HybridSearch` facade: gọi BM25 + Dense parallel-style → RRF.

**Số tests pass: M1 = 13/13, M2 = 5/5** (tổng 18/18 ở phần của tôi;
toàn nhóm 37/37).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** **Reciprocal Rank Fusion** — fuse rankers
  bằng `1/(k+rank)` không cần normalise scores. Đẹp ở chỗ chỉ phụ
  thuộc thứ hạng, không phụ thuộc scale BM25 vs cosine — robust
  ngay cả khi 2 ranker có distribution điểm hoàn toàn khác nhau.
  Tham số `k=60` là constant từ paper Cormack et al. (2009), không
  cần tune.
- **Điều bất ngờ nhất:** Vietnamese segmentation thực sự matter cho
  BM25. Trên test set tiếng Việt, "nghỉ phép" tokenize thành 2 token
  riêng (`["nghỉ", "phép"]`) làm BM25 mất context, recall tụt rõ rệt.
  Underthesea tăng BM25 hit rate ~30% so với whitespace split trên
  test set của Lab 18.
- **Kết nối với bài giảng:** Hierarchical chunking là pattern "index
  small, return large" — index child cho semantic precision, trả
  parent cho LLM context window. Ý tưởng giống multi-resolution
  retrieval (small-to-large) trong vision.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** **Bug parent_id collision**. Ban đầu
  `parent_id = f"parent_{i}"` không namespace theo source → 3
  documents đều có `parent_0`, `parent_1`,... → khi build
  `parent_index` dict, key trùng → last write wins → **20/20 query
  trả về cùng 1 parent**. Phát hiện vì pipeline output debug log thấy
  contexts gần như identical mọi query.
- **Cách giải quyết:** Namespace bằng `f"{source}::parent_{i}"`.
  Document `sample_01.md` parent_0 không còn trùng `sample_02.md`
  parent_0. Đây là bug retrieval thật — không phải eval artefact —
  rất khó phát hiện vì test M1 đơn lẻ chỉ chạy 1 document nên không
  bao giờ trigger.
- **Khó khăn M2:** BM25 score = 0 cho query có dấu vì underthesea
  segment query khác với segment corpus index time (model retrain
  giữa lần). Fix: nếu max score ≤ 0, retry với regex tokenize đơn
  giản — degrades gracefully thay vì trả empty.
- **Khó khăn fallback:** `_HashEncoder` dim=1024 khớp `EMBEDDING_DIM`
  để Qdrant collection schema không vỡ khi swap encoder; mất 30 phút
  debug vì ban đầu hard-code 256.

## 4. Nếu làm lại

- **Sẽ làm khác:** Test multi-document từ đầu (M1 test fixture chỉ có
  1 doc → không bao giờ trigger parent_id collision). Lesson: unit
  test với input "đủ đa dạng" mới catch được integration bug.
- **Module muốn thử tiếp:**
  - **Splade** — sparse neural retriever, vừa giữ tính lexical của
    BM25 vừa có semantic. Có thể thay BM25 trong hybrid hoặc thêm
    vào RRF ensemble (3 rankers).
  - **Multi-vector retrieval** (ColBERT-style) — mỗi chunk = nhiều
    vector, score = MaxSim. Đắt nhưng precision rất cao trên long
    document.
  - **Adaptive chunking** — chọn strategy (semantic / hierarchical /
    structure) dựa trên loại document (markdown có header → structure;
    plain text → semantic).

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality (multi-tier fallbacks, edge case handling) | 4 |
| Teamwork (đồng bộ schema chunk/SearchResult với Liam, Bân) | 4 |
| Problem solving (debug parent_id collision multi-document) | 5 |
