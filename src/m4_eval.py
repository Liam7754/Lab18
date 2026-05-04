"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis.

Test: pytest tests/test_m4.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── RAGAS evaluation (with offline lexical fallback) ──


def _try_ragas(questions, answers, contexts, ground_truths) -> dict | None:
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception:
        return None

    dataset = Dataset.from_dict(
        {
            "question": list(questions),
            "answer": list(answers),
            "contexts": list(contexts),
            "ground_truth": list(ground_truths),
        }
    )
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
    except Exception:
        return None

    per_question: list[EvalResult] = []
    for _, row in df.iterrows():
        per_question.append(
            EvalResult(
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                contexts=list(row.get("contexts", [])),
                ground_truth=row.get("ground_truth", ""),
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
        )
    agg = lambda key: float(sum(getattr(r, key) for r in per_question) / max(len(per_question), 1))
    return {
        "faithfulness": agg("faithfulness"),
        "answer_relevancy": agg("answer_relevancy"),
        "context_precision": agg("context_precision"),
        "context_recall": agg("context_recall"),
        "per_question": per_question,
        "_engine": "ragas",
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _recall(reference: list[str], candidate: list[str]) -> float:
    sr, sc = set(reference), set(candidate)
    if not sr:
        return 0.0
    return len(sr & sc) / len(sr)


def _evaluate_lexical(questions, answers, contexts, ground_truths) -> dict:
    """Lexical proxy metrics when RAGAS / OpenAI aren't available.

    These are NOT identical to RAGAS but produce sensible 0–1 numbers so
    the pipeline stays runnable end-to-end.
    """
    per_question: list[EvalResult] = []
    for q, a, ctxs, gt in zip(questions, answers, contexts, ground_truths):
        a_tok = _tokenize(a)
        q_tok = _tokenize(q)
        gt_tok = _tokenize(gt)
        ctx_tok = _tokenize(" ".join(ctxs))

        # Faithfulness: how much of the answer is supported by the contexts.
        faith = _recall(a_tok, ctx_tok) if a_tok else 0.0
        # Answer relevancy: how much of the answer overlaps with the question/gt.
        rel = max(_jaccard(a_tok, q_tok), _jaccard(a_tok, gt_tok))
        # Context precision: per-context, fraction overlapping with gt, weighted by rank.
        cp_scores = []
        for i, ctx in enumerate(ctxs):
            ct_tok = _tokenize(ctx)
            overlap = len(set(ct_tok) & set(gt_tok)) / max(len(set(ct_tok)), 1)
            cp_scores.append(overlap / math.log2(i + 2))
        cp = sum(cp_scores) / max(len(cp_scores), 1) if ctxs else 0.0
        cp = min(cp, 1.0)
        # Context recall: fraction of gt tokens covered by contexts.
        cr = _recall(gt_tok, ctx_tok)

        per_question.append(
            EvalResult(
                question=q,
                answer=a,
                contexts=list(ctxs),
                ground_truth=gt,
                faithfulness=float(faith),
                answer_relevancy=float(rel),
                context_precision=float(cp),
                context_recall=float(cr),
            )
        )

    n = max(len(per_question), 1)
    return {
        "faithfulness": sum(r.faithfulness for r in per_question) / n,
        "answer_relevancy": sum(r.answer_relevancy for r in per_question) / n,
        "context_precision": sum(r.context_precision for r in per_question) / n,
        "context_recall": sum(r.context_recall for r in per_question) / n,
        "per_question": per_question,
        "_engine": "lexical_fallback",
    }


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Evaluate with RAGAS if available, else lexical fallback."""
    if not (
        len(questions) == len(answers) == len(contexts) == len(ground_truths)
    ):
        raise ValueError("questions / answers / contexts / ground_truths length mismatch")

    has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if has_openai:
        ragas_out = _try_ragas(questions, answers, contexts, ground_truths)
        if ragas_out is not None:
            return ragas_out
    return _evaluate_lexical(questions, answers, contexts, ground_truths)


# ─── Failure analysis ──────────────────────────────────


_DIAGNOSTICS = [
    (
        "faithfulness",
        0.85,
        "LLM hallucinating — answer not grounded in context",
        "Tighten prompt, lower temperature, force quote-from-context",
    ),
    (
        "context_recall",
        0.75,
        "Missing relevant chunks — retriever didn't surface the source",
        "Improve chunking (smaller children) or add BM25 to recall lexical hits",
    ),
    (
        "context_precision",
        0.75,
        "Too many irrelevant chunks crowding the context window",
        "Add a cross-encoder reranker or metadata filters; lower top_k",
    ),
    (
        "answer_relevancy",
        0.80,
        "Answer doesn't actually address the question asked",
        "Improve prompt template; require the answer to restate the question",
    ),
]


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Sort by avg score → bottom_n → diagnose worst metric per question."""
    if not eval_results:
        return []

    def avg(r: EvalResult) -> float:
        return (
            r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall
        ) / 4.0

    ranked = sorted(eval_results, key=avg)[:bottom_n]

    out: list[dict] = []
    for r in ranked:
        scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(scores, key=lambda k: scores[k])
        worst_score = scores[worst_metric]

        diagnosis = "Acceptable across all metrics — review prompt"
        suggested_fix = "Investigate manually; no single metric below threshold."
        for metric, threshold, diag, fix in _DIAGNOSTICS:
            if scores[metric] < threshold:
                diagnosis = diag
                suggested_fix = fix
                worst_metric = metric
                worst_score = scores[metric]
                break

        out.append(
            {
                "question": r.question,
                "answer": r.answer,
                "ground_truth": r.ground_truth,
                "worst_metric": worst_metric,
                "score": float(worst_score),
                "all_scores": scores,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    return out


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json") -> None:
    aggregate = {
        k: float(v)
        for k, v in results.items()
        if k != "per_question" and not k.startswith("_") and isinstance(v, (int, float))
    }
    per_q = []
    for r in results.get("per_question", []):
        per_q.append(
            {
                "question": r.question,
                "answer": r.answer,
                "ground_truth": r.ground_truth,
                "contexts": r.contexts,
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
            }
        )
    report = {
        "engine": results.get("_engine", "unknown"),
        "aggregate": aggregate,
        "num_questions": len(per_q),
        "per_question": per_q,
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
