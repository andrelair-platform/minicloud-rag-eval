"""Post evaluation scores to Langfuse /api/public/scores."""

import os
import requests
from typing import Optional


def _auth() -> tuple[str, str]:
    return os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"]


def _host() -> str:
    return os.environ.get(
        "LANGFUSE_HOST", "http://langfuse-web.langfuse.svc.cluster.local:3000"
    ).rstrip("/")


def post_scores(trace_id: str, scores: dict[str, float]) -> None:
    if not trace_id:
        return
    import math
    pub, sec = _auth()
    for name, value in scores.items():
        safe_value = 0.0 if (value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))) else float(value)
        requests.post(
            f"{_host()}/api/public/scores",
            auth=(pub, sec),
            json={"traceId": trace_id, "name": name, "value": safe_value},
            timeout=10,
        ).raise_for_status()


# ── Datasets (generic; used by the Retrieva negatives harvester, RTV-75) ──────────
# Langfuse v3 public API: datasets live under /api/public/v2/datasets, items under
# /api/public/dataset-items (same host/auth as traces + scores above).


def ensure_dataset(name: str, description: Optional[str] = None) -> None:
    """Create the dataset if absent. Langfuse upserts by name, so a re-run is a no-op (200)."""
    pub, sec = _auth()
    body: dict = {"name": name}
    if description:
        body["description"] = description
    resp = requests.post(
        f"{_host()}/api/public/v2/datasets", auth=(pub, sec), json=body, timeout=15
    )
    if resp.status_code not in (200, 201, 409):  # 409 = already exists on some versions
        resp.raise_for_status()


def list_dataset_source_trace_ids(name: str) -> set[str]:
    """Trace ids already represented in the dataset — the idempotency key for harvesting."""
    pub, sec = _auth()
    seen: set[str] = set()
    page = 1
    while True:
        resp = requests.get(
            f"{_host()}/api/public/dataset-items",
            auth=(pub, sec),
            params={"datasetName": name, "limit": 50, "page": page},
            timeout=30,
        )
        if resp.status_code == 404:
            break
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        for item in data:
            tid = item.get("sourceTraceId") or (item.get("metadata") or {}).get("traceId")
            if tid:
                seen.add(tid)
        if len(data) < 50:
            break
        page += 1
    return seen


def upsert_dataset_item(
    dataset_name: str,
    *,
    item_id: str,
    input_: object,
    expected_output: Optional[object] = None,
    metadata: Optional[dict] = None,
    source_trace_id: Optional[str] = None,
) -> None:
    """Create/overwrite a dataset item. A stable item_id makes re-POST an upsert, not a dup."""
    pub, sec = _auth()
    body: dict = {"datasetName": dataset_name, "id": item_id, "input": input_}
    if expected_output is not None:
        body["expectedOutput"] = expected_output
    if metadata is not None:
        body["metadata"] = metadata
    if source_trace_id:
        body["sourceTraceId"] = source_trace_id
    requests.post(
        f"{_host()}/api/public/dataset-items", auth=(pub, sec), json=body, timeout=15
    ).raise_for_status()


def get_traces(minutes: int = 15, limit: int = 20) -> list[dict]:
    # limit=100 causes Langfuse v3 to issue a slow full-table scan → internal DB timeout → 422.
    # Keep limit ≤20; the online sampler only needs a handful of recent traces.
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    from_ts = (now - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    pub, sec = _auth()
    resp = requests.get(
        f"{_host()}/api/public/traces",
        auth=(pub, sec),
        params={"limit": limit, "fromTimestamp": from_ts, "toTimestamp": to_ts},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def filter_phi3_financial(traces: list[dict]) -> list[dict]:
    """Keep only traces from gpt-4o-mini model, not already scored online."""
    out = []
    for t in traces:
        name = (t.get("name") or "").lower()
        tags = t.get("tags") or []
        # LiteLLM sets trace name to the model name
        if "gpt-4o-mini" in name or "gpt-4o-mini" in tags:
            if "online_faithfulness" not in [s.get("name") for s in t.get("scores", [])]:
                out.append(t)
    return out


def extract_query_and_output(trace: dict) -> tuple[Optional[str], Optional[str]]:
    query, output = None, None
    inp = trace.get("input")
    if isinstance(inp, list):
        for msg in reversed(inp):
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = msg.get("content")
                break
    elif isinstance(inp, dict):
        query = inp.get("query") or inp.get("content")
    elif isinstance(inp, str):
        query = inp

    out = trace.get("output")
    if isinstance(out, dict):
        output = out.get("content") or str(out)
    elif isinstance(out, str):
        output = out
    return query, output
