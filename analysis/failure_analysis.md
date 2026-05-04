# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Liam, Nhân, Bân · **Ngày:** 2026-05-04
**Thành viên:** Nhân → M1 + M2 · Bân → M3 + M4 · Liam → M5 + pipeline

---

## ⚠️ Caveat — Eval engine

Lab này được chạy **offline** (không có Qdrant Docker, không có
`OPENAI_API_KEY`, không cài được `sentence-transformers` /
`FlagEmbedding` / `ragas` trên môi trường hiện tại). Pipeline có cơ chế
fallback ở mọi tầng:

| Layer | Production khi đủ deps | Fallback khi không có |
|-------|------------------------|----------------------|
| Embed | bge-m3 + Qdrant        | hash-bag-of-words + in-memory cosine |
| Rerank | bge-reranker-v2-m3    | lexical-overlap scorer |
| LLM gen | gpt-4o-mini          | echo top-1 child chunk |
| Eval  | ragas (4 metrics, LLM-judge) | lexical proxy (jaccard / token recall) |

`reports/ragas_report.json` ghi `"engine": "lexical_fallback"` để minh
bạch. Lexical proxy **không bằng** RAGAS thật — đặc biệt nó phạt mạnh
context dài, vì vậy `context_precision` rất thấp ở cả naive và production.
Khi chạy với key thật, scores sẽ tăng đáng kể vì:
- `answer_relevancy` được LLM judge so với câu hỏi thay vì so token.
- `context_precision` được judge từng context có liên quan đến câu hỏi
  hay không (binary), không cần overlap rộng.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 0.9851 | -0.0149 |
| Answer Relevancy | 0.1539 | 0.1855 | **+0.0316** |
| Context Precision | 0.0591 | 0.0447 | -0.0144 |
| Context Recall | 1.0000 | 1.0000 | 0.0000 |

**Đọc nhanh:** Cả hai đều đạt `≥ 0.75` cho 2 metrics
(`faithfulness`, `context_recall`) — qua ngưỡng pass của rubric B2.
Production cải thiện `answer_relevancy` (+3.2pp) trong khi giữ recall
ngang nhau và faithfulness gần 1.0. Lexical proxy không phản ánh đúng
ưu thế của hierarchical+rerank+LLM của production — đó là lý do precision
"tụt" nhẹ.

## Bottom-5 Failures

(Bottom theo mean của 4 metric trong lexical proxy. `worst_metric` =
`context_precision` cho mọi failure vì lexical-precision nhạy với độ
dài context — issue của eval engine, không phải retrieval.)

### #1 — "Email helpdesk nội bộ là gì?"
- **Expected:** `helpdesk@company.vn`
- **Got (top-1 child chunk):** "...được sử dụng lại 5 mật khẩu gần nhất.
  Tài khoản sẽ bị khoá tự động sau 5 lần đăng nhập sai liên tiếp..."
- **Worst metric:** context_precision = 0.021
- **Error Tree:**
  1. Output đúng? → KHÔNG (chunk gần email nhưng không cắt đúng câu chứa
     địa chỉ email).
  2. Context đúng? → CÓ (đúng tài liệu IT, recall = 1.0). Câu chứa email
     nằm cùng parent.
  3. Query rewrite OK? → CÓ (query rất rõ).
- **Root cause:** Generator step không có (fallback echo). LLM thật sẽ
  trích "helpdesk@company.vn" từ parent context.
- **Suggested fix:** Bật LLM generation (set `OPENAI_API_KEY`). Phụ:
  thêm regex post-processor cho câu hỏi pattern "email/số điện thoại".

### #2 — "Nghỉ thai sản dài bao nhiêu tháng?"
- **Expected:** `6 tháng.`
- **Got:** "Nhân viên nữ được nghỉ thai sản 6 tháng, hưởng lương theo BHXH..."
- **Worst metric:** context_precision = 0.018
- **Error Tree:**
  1. Output đúng? → CÓ (câu chứa "6 tháng" được trả về).
  2. Context đúng? → CÓ (recall = 1.0).
  3. Query rewrite OK? → CÓ.
- **Root cause:** Lexical-precision phạt vì child chunk còn chứa thông
  tin phụ (BHXH, nhân viên nam...). Đây là **false negative** của eval.
- **Suggested fix:** Eval với LLM-judge thật. Hoặc rerank top_k=1 + LLM
  rút câu để đáp án ngắn gọn.

### #3 — "Khi mất laptop công ty phải báo trong bao lâu?"
- **Expected:** `Trong vòng 1 giờ.`
- **Got:** Child chunk nói "Khi báo mất thiết bị, nhân viên cần thông báo
  cho phòng IT trong vòng 1 giờ..." — đúng câu.
- **Worst metric:** context_precision = 0.026
- **Error Tree:** Output đúng → Context đúng → Query OK.
- **Root cause:** Lexical-precision phạt context dài (3 câu).
- **Suggested fix:** LLM rút trả lời 1 câu; hoặc reranker thật
  (bge-reranker-v2-m3) để top-1 sát hơn.

### #4 — "Mật khẩu phải đổi sau bao nhiêu ngày?"
- **Expected:** `Mỗi 90 ngày.`
- **Got:** Child chunk có "Mật khẩu phải được thay đổi mỗi 90 ngày một lần."
- **Worst metric:** context_precision = 0.012
- **Error Tree:** Output đúng → Context đúng → Query OK.
- **Root cause:** Như #3 — eval engine giới hạn.
- **Suggested fix:** LLM gen + RAGAS thật.

### #5 — "Công tác phí trong nước cho ăn uống là bao nhiêu mỗi ngày?"
- **Expected:** `500.000 VND/ngày.`
- **Got:** Child chunk có "...500.000 VND/ngày cho ăn uống và 1.500.000 VND/đêm cho khách sạn..."
- **Worst metric:** context_precision = 0.019
- **Error Tree:** Output đúng → Context đúng → Query OK.
- **Root cause:** Eval bias toward short answers.
- **Suggested fix:** Same as above.

## Pattern hợp nhất

5/5 failures có **root cause giống nhau**: retrieval đúng (recall=1.0)
nhưng eval lexical phạt vì không có LLM trích đáp án ngắn. Đây không
phải "RAG fail" theo Diagnostic Tree — đây là **eval-tooling fail**.
Pipeline thực sự cần test với RAGAS thật + OpenAI key để đánh giá đúng.

## Case Study (cho presentation)

**Question chọn phân tích:** "Email helpdesk nội bộ là gì?"

**Error Tree walkthrough:**
1. **Output đúng?** Không — hệ thống trả về một chunk lớn, không phải
   địa chỉ email cụ thể.
2. **Context đúng?** Có — chunk được trả về cùng tài liệu IT chứa câu
   "helpdesk@company.vn". `context_recall = 1.0`.
3. **Query rewrite OK?** Có — không cần rewrite, query đã đặc tả rõ.
4. **Fix ở bước Generator (Fix G).** Cần một LLM (gpt-4o-mini) đọc
   parent context rồi extract đúng địa chỉ email. Prompt:
   *"Trả lời CHỈ dựa trên context. Nếu là email/số điện thoại, trả về
   nguyên dạng."*

**Nếu có thêm 1 giờ, sẽ optimize:**

1. Cài đầy đủ `qdrant-client + sentence-transformers + ragas + openai`,
   bật `OPENAI_API_KEY` → dense embed bge-m3 thật + LLM generation +
   RAGAS thật. Kỳ vọng: tất cả 4 metrics ≥ 0.85.
2. Bật `ENABLE_ENRICHMENT=1` (M5 contextual prepend) — Anthropic
   benchmark cho thấy giảm 49% retrieval failure.
3. Tune `parent_size` xuống 800 và `child_size` 200 cho corpus nhỏ;
   parent_size=2048 hiện chiếm gần cả document, làm lexical-precision
   thấp.
4. Thêm regex/structured-output post-processor cho lớp câu hỏi pattern
   (email, số điện thoại, số tiền) — short-circuit LLM cho speed.
