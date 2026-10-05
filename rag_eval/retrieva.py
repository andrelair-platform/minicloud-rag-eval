"""Retrieva target (RTV-75): harvest 👎-rated chat-RAG traces into a Langfuse dataset.

OFFLINE / harvest ONLY, by design and to avoid redundancy with what Retrieva already has:
  - Retrieva's runtime `llmJudge` already scores faithfulness/groundedness inline → this does
    NOT re-score faithfulness online.
  - Retrieva's `scripts/evaluate.js` already gates retrieval Recall@K in CI → this does NOT
    re-implement retrieval metrics.
This module's only job is to turn the RTV-73 feedback signal (a `user_rating` score on each
chat answer's trace) into a curated `rag-chat-negatives` dataset. The offline RAGAS experiment
(answer_relevancy + prompt A/B) runs on top via the framework's existing `offline` mode.

Points at RETRIEVA's own Langfuse project (per-product LLMOps standard) via the standard
LANGFUSE_* env — run it with that project's keys, not the phi3-financial ones.
"""

import os
from typing import Optional

from rag_eval.langfuse_reporter import (
    get_traces,
    ensure_dataset,
    list_dataset_source_trace_ids,
    upsert_dataset_item,
)

RAG_CHAT_TAG = "feature:rag-chat"  # set by retrieva-backend startTrace (services/rag.ts)
FEEDBACK_SCORE = "user_rating"  # posted by RTV-73 (positive→1, negative→0)
DATASET = os.environ.get("RETRIEVA_NEGATIVES_DATASET", "rag-chat-negatives")


def _user_rating(trace: dict) -> Optional[float]:
    for s in trace.get("scores") or []:
        if s.get("name") == FEEDBACK_SCORE:
            try:
                return float(s.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def filter_negative_feedback(traces: list[dict]) -> list[dict]:
    """Chat-RAG traces a user rated negatively (user_rating ≤ 0; RTV-73 maps 👎→0)."""
    out = []
    for t in traces:
        if RAG_CHAT_TAG not in (t.get("tags") or []):
            continue
        rating = _user_rating(t)
        if rating is not None and rating <= 0:
            out.append(t)
    return out


def extract_sample(trace: dict) -> tuple[Optional[str], Optional[str]]:
    """(question, answer) from a Retrieva chat-RAG trace (input=question, output=answer)."""
    inp = trace.get("input")
    if isinstance(inp, str):
        question = inp
    elif isinstance(inp, dict):
        question = inp.get("question") or inp.get("query") or inp.get("content")
    else:
        question = None

    out = trace.get("output")
    if isinstance(out, str):
        answer = out
    elif isinstance(out, dict):
        answer = out.get("content") or out.get("answer") or out.get("text")
    else:
        answer = None
    return question, answer


def run_harvest_negatives() -> None:
    window = int(os.environ.get("HARVEST_WINDOW_MINUTES", str(24 * 60)))
    # get_traces caps at ~20 per call (a documented Langfuse-v3 slow-scan quirk); run the
    # harvester frequently rather than fetching a huge window. Pagination is a follow-up.
    limit = int(os.environ.get("HARVEST_TRACE_LIMIT", "20"))

    print(f"[harvest-negatives] dataset={DATASET} window={window}min limit={limit}")
    ensure_dataset(
        DATASET,
        "Retrieva chat-RAG answers users rated 👎 (RTV-73). Curate expectedOutput + a failure "
        "tag in the UI, then run offline RAGAS experiments (answer_relevancy, prompt A/B).",
    )
    already = list_dataset_source_trace_ids(DATASET)
    traces = get_traces(minutes=window, limit=limit)
    negatives = filter_negative_feedback(traces)
    print(
        f"[harvest-negatives] {len(traces)} traces, {len(negatives)} negative, "
        f"{len(already)} already harvested"
    )

    added = skipped = 0
    for t in negatives:
        tid = t.get("id")
        if not tid or tid in already:
            skipped += 1
            continue
        question, answer = extract_sample(t)
        if not question:
            skipped += 1
            continue
        upsert_dataset_item(
            DATASET,
            item_id=f"neg-{tid}",
            input_=question,
            metadata={
                "traceId": tid,
                "proposedAnswer": answer,  # kept for reviewer context; expectedOutput is curated
                "feedbackValue": _user_rating(t),
                "workspaceId": (t.get("metadata") or {}).get("workspaceId"),
            },
            source_trace_id=tid,
        )
        added += 1
        print(f"  + {tid[:8]}… harvested")

    print(f"[harvest-negatives] done: added={added} skipped={skipped}")
