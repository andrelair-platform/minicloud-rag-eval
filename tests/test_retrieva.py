"""Unit tests for the Retrieva 👎-harvester (RTV-75). Pure logic + the harvest loop with
the Langfuse REST helpers stubbed — no network, no ragas import."""

from rag_eval import retrieva


def _trace(tid, tags, rating, inp="q?", out="a."):
    scores = [] if rating is None else [{"name": "user_rating", "value": rating}]
    return {"id": tid, "tags": tags, "scores": scores, "input": inp, "output": out}


# ── filter_negative_feedback ──────────────────────────────────────────────────

def test_keeps_only_tagged_negative_traces():
    traces = [
        _trace("t1", ["feature:rag-chat"], 0),          # 👎 → keep
        _trace("t2", ["feature:rag-chat"], 1),          # 👍 → drop
        _trace("t3", ["feature:rag-chat"], None),       # unrated → drop
        _trace("t4", ["other"], 0),                      # wrong feature → drop
        _trace("t5", ["feature:rag-chat"], -1),         # negative → keep
    ]
    kept = {t["id"] for t in retrieva.filter_negative_feedback(traces)}
    assert kept == {"t1", "t5"}


# ── extract_sample ────────────────────────────────────────────────────────────

def test_extract_sample_handles_string_and_dict_shapes():
    assert retrieva.extract_sample({"input": "why?", "output": "because"}) == ("why?", "because")
    assert retrieva.extract_sample(
        {"input": {"question": "q"}, "output": {"content": "c"}}
    ) == ("q", "c")
    # no usable question → (None, ...)
    q, _ = retrieva.extract_sample({"input": None, "output": "x"})
    assert q is None


# ── run_harvest_negatives (loop: dedupe + skip-unusable + upsert) ─────────────

def test_harvest_upserts_new_negatives_and_dedupes(monkeypatch):
    calls = []
    monkeypatch.setattr(retrieva, "ensure_dataset", lambda *a, **k: None)
    monkeypatch.setattr(retrieva, "list_dataset_source_trace_ids", lambda name: {"t_old"})
    monkeypatch.setattr(
        retrieva,
        "get_traces",
        lambda minutes, limit: [
            _trace("t_new", ["feature:rag-chat"], 0),           # harvest
            _trace("t_old", ["feature:rag-chat"], 0),           # already in dataset → skip
            _trace("t_noq", ["feature:rag-chat"], 0, inp=None),  # no question → skip
            _trace("t_pos", ["feature:rag-chat"], 1),           # 👍 → filtered out
        ],
    )
    monkeypatch.setattr(
        retrieva,
        "upsert_dataset_item",
        lambda dataset, **kw: calls.append(kw),
    )

    retrieva.run_harvest_negatives()

    assert len(calls) == 1
    only = calls[0]
    assert only["item_id"] == "neg-t_new"
    assert only["source_trace_id"] == "t_new"
    assert only["input_"] == "q?"
    assert only["metadata"]["traceId"] == "t_new"
    assert only["metadata"]["feedbackValue"] == 0.0
