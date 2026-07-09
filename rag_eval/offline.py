"""Offline evaluation: run the full RAG triad on eval_dataset.json."""

import json
import os
import sys
import random
from pathlib import Path

from rag_eval.retriever import retrieve_chunks
from rag_eval.generator import generate_answer
from rag_eval.metrics import rouge_l, hit_rate, mrr
from rag_eval.ragas_runner import score_batch
from rag_eval.langfuse_reporter import post_scores

FAST_SAMPLE_DOMAINS = {
    "regulatory_capital": 2,
    "liquidity": 2,
    "profitability": 2,
    "lvmh": 2,
    "off_topic": 2,
}


def _fast_subset(dataset: list[dict]) -> list[dict]:
    """Pick 10 representative samples across domains for the CI fast gate."""
    by_domain: dict[str, list] = {}
    for s in dataset:
        by_domain.setdefault(s.get("domain", "other"), []).append(s)

    chosen = []
    for domain, count in FAST_SAMPLE_DOMAINS.items():
        pool = by_domain.get(domain, [])
        chosen.extend(random.sample(pool, min(count, len(pool))))

    # Fill to 10 if domains are sparse
    remaining = [s for s in dataset if s not in chosen]
    while len(chosen) < 10 and remaining:
        chosen.append(remaining.pop(0))
    return chosen[:10]


def run_offline_eval() -> None:
    dataset_path = Path(os.environ.get("EVAL_DATASET_PATH", "/config/eval_dataset.json"))
    collection_uuid = os.environ["RAG_COLLECTION_UUID"]
    fast_mode = os.environ.get("EVAL_FAST_MODE", "false").lower() == "true"
    faithfulness_threshold = float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.70"))
    hit_rate_threshold = float(os.environ.get("HIT_RATE_THRESHOLD", "0.80"))

    dataset: list[dict] = json.loads(dataset_path.read_text())
    subset = _fast_subset(dataset) if fast_mode else dataset

    mode_label = f"FAST ({len(subset)}/{len(dataset)} samples)" if fast_mode else f"FULL ({len(subset)} samples)"
    print(f"\n=== RAG Eval — {mode_label} ===\n")

    # --- Retrieval + Generation ---
    queries, answers, contexts, ground_truths, trace_ids = [], [], [], [], []
    for i, sample in enumerate(subset, 1):
        print(f"[{i}/{len(subset)}] {sample['query'][:80]}")
        chunks = retrieve_chunks(sample["query"], collection_uuid)
        answer, trace_id = generate_answer(sample["query"], chunks)
        queries.append(sample["query"])
        answers.append(answer)
        contexts.append(chunks)
        ground_truths.append(sample.get("ground_truth", ""))
        trace_ids.append(trace_id)

    # --- Ragas batch (LLM judge) ---
    print("\n[Ragas] Running LLM-as-judge metrics…")
    ragas_scores = score_batch(queries, answers, contexts, ground_truths)

    # --- Deterministic metrics + Langfuse reporting ---
    results = []
    for i, sample in enumerate(subset):
        det = {
            "rouge_l": rouge_l(answers[i], ground_truths[i]),
            "hit_rate": hit_rate(ground_truths[i], contexts[i]),
            "mrr": mrr(ground_truths[i], contexts[i]),
        }
        all_scores = {**ragas_scores[i], **det}

        if trace_ids[i]:
            try:
                post_scores(trace_ids[i], all_scores)
            except Exception as e:
                print(f"  [warn] Langfuse post failed: {e}")

        results.append({**sample, "answer": answers[i], "metrics": all_scores})

    # --- Aggregate ---
    def _mean(key: str) -> float:
        vals = [r["metrics"][key] for r in results if key in r["metrics"]]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    agg = {
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "context_precision": _mean("context_precision"),
        "context_recall": _mean("context_recall"),
        "rouge_l": _mean("rouge_l"),
        "hit_rate": _mean("hit_rate"),
        "mrr": _mean("mrr"),
    }

    # --- Print summary ---
    print("\n┌─────────────────────────────────────────────┐")
    print("│            RAG Evaluation Summary            │")
    print("├──────────────────────────┬──────────────────┤")
    for metric, value in agg.items():
        threshold = ""
        if metric == "faithfulness":
            threshold = f"  (threshold: {faithfulness_threshold})"
        elif metric == "hit_rate":
            threshold = f"  (threshold: {hit_rate_threshold})"
        flag = "✅" if value >= 0.70 else "⚠️ "
        print(f"│ {metric:<24} │ {flag} {value:.4f}{threshold:<8}│")
    print("└──────────────────────────┴──────────────────┘\n")

    # --- CI gate: fail if below threshold ---
    failures = []
    if agg["faithfulness"] < faithfulness_threshold:
        failures.append(f"faithfulness {agg['faithfulness']:.4f} < {faithfulness_threshold}")
    if agg["hit_rate"] < hit_rate_threshold:
        failures.append(f"hit_rate {agg['hit_rate']:.4f} < {hit_rate_threshold}")

    if failures:
        print("❌ REGRESSION DETECTED:")
        for f in failures:
            print(f"   {f}")
        sys.exit(1)

    print("✅ All thresholds passed.")
