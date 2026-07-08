"""Online sampling: score 5% of recent phi3-financial production traces."""

import os
import random

from rag_eval.retriever import retrieve_chunks
from rag_eval.ragas_runner import score_faithfulness_single
from rag_eval.langfuse_reporter import (
    get_traces,
    filter_phi3_financial,
    extract_query_and_output,
    post_scores,
)


def run_online_eval() -> None:
    collection_uuid = os.environ["RAG_COLLECTION_UUID"]
    sample_rate = float(os.environ.get("SAMPLE_RATE", "0.05"))
    window_minutes = int(os.environ.get("SAMPLING_WINDOW_MINUTES", "15"))

    print(f"[online-eval] Fetching traces from last {window_minutes} min…")
    traces = get_traces(minutes=window_minutes)
    candidates = filter_phi3_financial(traces)
    print(f"[online-eval] {len(traces)} total traces, {len(candidates)} phi3-financial unscored")

    if not candidates:
        print("[online-eval] Nothing to score.")
        return

    n = max(1, int(len(candidates) * sample_rate))
    sampled = random.sample(candidates, min(n, len(candidates)))
    print(f"[online-eval] Scoring {len(sampled)} sampled traces ({sample_rate*100:.0f}%)")

    for trace in sampled:
        query, output = extract_query_and_output(trace)
        if not query or not output:
            continue
        try:
            chunks = retrieve_chunks(query, collection_uuid)
            score = score_faithfulness_single(query, output, chunks)
            post_scores(trace["id"], {"online_faithfulness": score})
            print(f"  trace={trace['id'][:8]}… faithfulness={score:.4f}")
        except Exception as e:
            print(f"  [warn] trace={trace['id'][:8]}… error: {e}")

    print("[online-eval] Done.")
