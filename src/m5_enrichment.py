"""Module 5: Enrichment Pipeline — Summarize, HyQA, Contextual Prepend, Auto Metadata.

All techniques call gpt-4o-mini when OPENAI_API_KEY is set; otherwise they use
deterministic heuristics so the pipeline is still runnable offline.

Test: pytest tests/test_m5.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


# ─── OpenAI helper (lazy) ───────────────────────────────


def _openai_chat(system: str, user: str, max_tokens: int = 200) -> str | None:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


# ─── Technique 1: Summarize ─────────────────────────────


def summarize_chunk(text: str) -> str:
    out = _openai_chat(
        "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
        text,
        max_tokens=150,
    )
    if out:
        return out
    # Extractive fallback: first 2 sentences.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return text.strip()
    head = " ".join(sentences[:2]).strip()
    if not head.endswith((".", "!", "?")):
        head += "."
    return head


# ─── Technique 2: Hypothesis Questions ──────────────────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    out = _openai_chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        f"Trả về mỗi câu hỏi trên một dòng, không đánh số.",
        text,
        max_tokens=200,
    )
    if out:
        questions = [q.strip().lstrip("0123456789.-) ").strip() for q in out.split("\n")]
        return [q for q in questions if q]

    # Heuristic fallback: turn salient nouns/numbers into questions.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    questions: list[str] = []
    for s in sentences[:n_questions]:
        # Strip terminal punctuation, prepend a question word.
        clean = re.sub(r"[.!?]+$", "", s)
        if re.search(r"\d", clean):
            questions.append(f"Bao nhiêu liên quan đến: {clean}?")
        else:
            questions.append(f"Điều gì về: {clean}?")
    if not questions and text.strip():
        questions.append(f"Đoạn này nói về điều gì?")
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend ────────────────────


def contextual_prepend(text: str, document_title: str = "") -> str:
    title = document_title or "tài liệu"
    out = _openai_chat(
        "Viết MỘT câu ngắn (tiếng Việt) mô tả đoạn văn này nằm ở đâu trong tài liệu "
        "và nói về chủ đề gì. Chỉ trả về 1 câu, không kèm giải thích.",
        f"Tài liệu: {title}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    if not out:
        # Heuristic fallback: pick the nearest section header in the text.
        header_match = re.search(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
        section = header_match.group(1).strip() if header_match else "nội dung chính"
        out = f"Trích từ {title} — chủ đề: {section}."
    return f"{out}\n\n{text}"


# ─── Technique 4: Auto Metadata ─────────────────────────


_CATEGORY_KEYWORDS = {
    "hr": ["nghỉ phép", "nhân viên", "thai sản", "bhxh", "thử việc", "lương"],
    "it": ["mật khẩu", "vpn", "wireguard", "otp", "helpdesk", "bitlocker", "thiết bị"],
    "finance": ["hoá đơn", "công tác", "thanh toán", "vnd", "mua sắm", "kế toán"],
    "policy": ["quy định", "phê duyệt", "chính sách", "quy trình"],
}


def extract_metadata(text: str) -> dict:
    out = _openai_chat(
        'Trích xuất metadata từ đoạn văn sau. Trả về JSON đúng định dạng: '
        '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}. '
        "Chỉ trả về JSON, không kèm giải thích.",
        text,
        max_tokens=200,
    )
    if out:
        try:
            # Strip ```json fences if any
            cleaned = re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.MULTILINE).strip()
            return json.loads(cleaned)
        except Exception:
            pass

    # Heuristic fallback
    low = text.lower()
    category = "policy"
    best_hit = 0
    for cat, kws in _CATEGORY_KEYWORDS.items():
        hit = sum(1 for kw in kws if kw in low)
        if hit > best_hit:
            best_hit = hit
            category = cat
    # Crude entity extraction: capitalised tokens & numeric patterns.
    entities = list({m.group(0) for m in re.finditer(r"\b\d+(?:[.,]\d+)?\s*(?:VND|ngày|tháng|năm|giờ|phút|ký tự)?\b", text)})
    # Topic = first sentence
    topic_match = re.split(r"[.!?\n]", text.strip(), maxsplit=1)
    topic = (topic_match[0] if topic_match else text)[:80].strip()
    language = "vi" if re.search(r"[ăâđêôơưáàảãạ]", low) else "en"
    return {
        "topic": topic,
        "entities": entities[:10],
        "category": category,
        "language": language,
    }


# ─── Full Enrichment Pipeline ───────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched: list[EnrichedChunk] = []
    methods_set = set(methods)
    do_all = "full" in methods_set

    for chunk in chunks:
        text = chunk["text"]
        meta = chunk.get("metadata", {})
        title = meta.get("source", "")

        summary = summarize_chunk(text) if (do_all or "summary" in methods_set) else ""
        questions = (
            generate_hypothesis_questions(text)
            if (do_all or "hyqa" in methods_set)
            else []
        )
        if do_all or "contextual" in methods_set:
            enriched_text = contextual_prepend(text, title)
        else:
            enriched_text = text
        auto_meta_extracted = (
            extract_metadata(text) if (do_all or "metadata" in methods_set) else {}
        )

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**meta, **auto_meta_extracted},
                method="+".join(methods),
            )
        )
    return enriched


if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )
    print("Original:", sample)
    print("Summary :", summarize_chunk(sample))
    print("HyQA    :", generate_hypothesis_questions(sample))
    print("Context :", contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024"))
    print("Meta    :", extract_metadata(sample))
