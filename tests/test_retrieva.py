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


# ── score-negatives (RTV-75b): item mapping + scoring loop ────────────────────

def test_item_qa_prefers_curated_expected_output_over_harvested_answer():
    # curated expectedOutput wins
    q, a, tid = retrieva._item_qa(
        {"input": "why?", "expectedOutput": "ideal", "metadata": {"proposedAnswer": "prod", "traceId": "t1"}}
    )
    assert (q, a, tid) == ("why?", "ideal", "t1")
    # falls back to the production answer when not curated; dict input is unwrapped
    q, a, tid = retrieva._item_qa(
        {"input": {"question": "q2"}, "sourceTraceId": "t2", "metadata": {"proposedAnswer": "prod2"}}
    )
    assert (q, a, tid) == ("q2", "prod2", "t2")


def test_score_negatives_scores_scorable_items_and_posts(monkeypatch):
    items = [
        {"input": "q1", "metadata": {"proposedAnswer": "a1", "traceId": "t1"}},
        {"input": "q2", "expectedOutput": "a2", "sourceTraceId": "t2"},
        {"input": None, "metadata": {"proposedAnswer": "a3", "traceId": "t3"}},  # no question → skip
        {"input": "q4", "metadata": {"traceId": "t4"}},  # no answer → skip
    ]
    posted = []
    monkeypatch.setattr(retrieva, "get_dataset_items", lambda name, limit_total=None: items)
    monkeypatch.setattr(retrieva, "post_scores", lambda tid, scores: posted.append((tid, scores)))

    captured = {}

    def fake_scorer(queries, answers):
        captured["queries"] = queries
        captured["answers"] = answers
        return [0.9, 0.3]

    retrieva.run_score_negatives(scorer=fake_scorer)

    # only the two scorable items are sent to the scorer, in order
    assert captured["queries"] == ["q1", "q2"]
    assert captured["answers"] == ["a1", "a2"]
    # each score posted to its source trace under the offline metric name
    assert ("t1", {"offline_answer_relevancy": 0.9}) in posted
    assert ("t2", {"offline_answer_relevancy": 0.3}) in posted
    assert len(posted) == 2
