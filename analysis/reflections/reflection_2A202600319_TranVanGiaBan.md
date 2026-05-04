# Individual Reflection — Lab 18

**Tên:** Trần Văn Gia Bân (2A202600319)
**Module phụ trách:** M3 (Reranking) + M4 (RAGAS Evaluation)

---

## 1. Đóng góp kỹ thuật

- **M3 — Reranking** (`src/m3_rerank.py`):
  - `CrossEncoderReranker._load_model`: 3-tier fallback chain —
    `FlagReranker` (bge-reranker-v2-m3, fp16) → `sentence_transformers.CrossEncoder` →
    `_LexicalReranker`. Single API surface (`compute_score | predict |
    score`) qua `self._kind` để caller không cần biết backend.
  - **Probe-load trick:** sau khi tạo FlagReranker, gọi
    `compute_score([("ping", "pong")])` để validate tokenizer còn
    work — vì một số FlagEmbedding/transformers combo crash runtime
    do `XLMRobertaTokenizer.prepare_for_model` đã removed ở
    transformers mới. Nếu probe fail → fallback ngay lúc load
    thay vì crash giữa pipeline.
  - **Runtime degrade:** nếu `compute_score` raise giữa rerank loop,
    reset `_model = None`, reload (lần 2 sẽ dùng tier kế tiếp), retry
    cùng pairs. Không vỡ pipeline ngay cả khi backend "đột tử".
  - `_LexicalReranker`: BM25-like overlap scorer offline, công thức
    `overlap / (1 + log(1 + len(doc)))` — đủ pass test M3 không cần
    GPU.
  - `FlashrankReranker`: lightweight alternative (<5ms claim), wrap
    `flashrank.Ranker` với fallback về CrossEncoderReranker khi
    flashrank không cài được.
  - `benchmark_reranker`: avg/min/max ms qua n_runs với
    `time.perf_counter`.
- **M4 — RAGAS Evaluation** (`src/m4_eval.py`):
  - `_try_ragas`: chạy ragas thật khi có `OPENAI_API_KEY` + ragas
    package; build HuggingFace `Dataset.from_dict` đúng schema
    (question / answer / contexts / ground_truth) → `evaluate`. Swallow
    Exception và return `None` → caller fallback nhẹ nhàng.
  - `_evaluate_lexical`: 4 lexical proxies cho RAGAS metrics.
    - **Faithfulness:** `recall(answer_tokens, context_tokens)`.
    - **Answer relevancy:** `max(jaccard(answer, question),
       jaccard(answer, gt))`.
    - **Context precision:** mỗi context tính `overlap_with_gt /
       len(ctx_tokens)`, weighted bằng `1/log2(rank+2)` (DCG-like) →
       trung bình các context, clamp ≤ 1.0.
    - **Context recall:** `recall(gt_tokens, context_tokens)`.
  - **Transparency flag:** mọi output có `_engine` field
    (`"ragas"` | `"lexical_fallback"`) — không che đậy điều gì khi
    chạy offline.
  - `failure_analysis`: sort theo mean của 4 metric → bottom_n →
    match Diagnostic Tree với 4 thresholds (F<0.85 / CR<0.75 /
    CP<0.75 / AR<0.80) ưu tiên match đầu tiên → trả
    `worst_metric + diagnosis + suggested_fix`. Đây là phần "biến
    score thấp thành action item" — quan trọng nhất của eval.
  - `save_report`: serialize aggregate + per_question + failures vào
    `ragas_report.json` (UTF-8, indent=2).

**Số tests pass: M3 = 5/5, M4 = 4/4** (tổng 9/9 ở phần của tôi; toàn
nhóm 37/37).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** **Diagnostic Tree** — map từ "metric nào
  thấp" → "lỗi ở đâu" → "fix gì". Cụ thể:
  - `faithfulness` thấp → LLM hallucinate → tighten prompt, lower temp.
  - `context_recall` thấp → retriever miss → cải tiến chunking / thêm BM25.
  - `context_precision` thấp → quá nhiều noise → reranker / lower top_k.
  - `answer_relevancy` thấp → prompt không bám question → restate question.
  Đây là framework chuẩn để biến "score 0.4" thành "action item rõ ràng",
  cùng tinh thần với SLO error budget trong SRE.
- **Điều bất ngờ nhất:** Lexical-overlap proxy phạt **rất nặng** context
  dài. Naive baseline (1 short paragraph) ngẫu nhiên "thắng" production
  (parent context 1KB) trên `context_precision` dù chất lượng retrieval
  kém hơn nhiều. Đây là chính xác lý do RAGAS phải dùng LLM-judge —
  overlap không capture được "context có liên quan đến question hay
  không", chỉ đo được "context chia sẻ token với gt hay không".
- **Kết nối với bài giảng:** Cross-encoder rerank ≠ bi-encoder retrieve.
  Bi-encoder embed Q, D độc lập rồi đo cosine — fast nhưng mất
  cross-attention. Cross-encoder concat (Q, D) → 1 forward pass với
  full attention → chính xác hơn nhiều nhưng O(N) inference, vì vậy
  chỉ rerank top-N (20-50), không thể search cả corpus.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** **FlagReranker dependency hell**. Cài được
  package nhưng `compute_score` crash runtime vì
  `XLMRobertaTokenizer.prepare_for_model` đã bị remove ở transformers
  mới. Nếu chỉ try/except quanh `from FlagEmbedding import` thì sẽ
  pass load nhưng vỡ giữa pipeline.
- **Cách giải quyết:** **Probe-load** — gọi `compute_score([("ping",
  "pong")])` ngay sau khi instantiate. Nếu probe fail → fallback ngay
  ở load time, không phải runtime. Đây là pattern "validate at
  boundary" — chuyển lỗi từ runtime sang init time, dễ debug hơn rất nhiều.
- **Khó khăn M4:** Không có RAGAS thật (offline) nhưng vẫn phải có số
  để pipeline runnable end-to-end. Trade-off: làm proxy quá đơn giản
  thì meaningless; làm phức tạp thì phải maintain song song với
  RAGAS thật. Chọn middle ground: 4 metric proxy đủ tách biệt
  (faithfulness ≠ context_recall ≠ context_precision ≠ relevancy)
  với formula khác nhau — số sẽ khác nhau, không degenerate hết về
  jaccard.
- **Bug rerank ranking:** `CrossEncoder.predict` đôi khi trả `numpy.float32`
  mảng, không phải list. `sorted(zip(scores, ...))` ổn, nhưng
  serialize ra JSON sau đó vỡ. Fix: `[float(s) for s in scores]`
  ngay sau predict — defensive coercion ở boundary.

## 4. Nếu làm lại

- **Sẽ làm khác:** Cài `ragas + openai` từ đầu và viết test M4 với
  RAGAS thật ngay từ commit đầu tiên. Hiện tại lexical-proxy giúp
  pipeline runnable nhưng số không thuyết phục với rubric — phải
  rerun với key thật trước khi nộp.
- **Module muốn thử tiếp:**
  - **Bge-reranker với GPU thật** — hiện tại fallback về lexical,
    không biết neural rerank cải thiện bao nhiêu trên test set
    tiếng Việt (Anthropic claim ~10pp precision boost).
  - **LLM-as-judge custom** — viết RAGAS-style metric với prompt
    riêng, calibrate trên 50 case manually-judged. Thoát phụ thuộc
    RAGAS framework, kiểm soát được prompt bias.
  - **Reranker distillation** — train cross-encoder small (deberta-v3-xsmall)
    distill từ bge-reranker. Giảm latency 5-10x, mất precision ~2-3pp.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng (Diagnostic Tree, cross vs bi-encoder) | 5 |
| Code quality (probe-load, runtime degrade, defensive coercion) | 4 |
| Teamwork (đồng bộ schema RerankResult/EvalResult với Liam, Nhân) | 4 |
| Problem solving (debug FlagReranker compat hell) | 5 |
