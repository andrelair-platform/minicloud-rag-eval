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
    pub, sec = _auth()
    for name, value in scores.items():
        requests.post(
            f"{_host()}/api/public/scores",
            auth=(pub, sec),
            json={"traceId": trace_id, "name": name, "value": value},
            timeout=10,
        ).raise_for_status()


def get_traces(minutes: int = 15, limit: int = 200) -> list[dict]:
    from datetime import datetime, timedelta, timezone

    from_ts = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    pub, sec = _auth()
    resp = requests.get(
        f"{_host()}/api/public/traces",
        auth=(pub, sec),
        params={"limit": limit, "fromTimestamp": from_ts},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def filter_phi3_financial(traces: list[dict]) -> list[dict]:
    """Keep only traces from phi3-financial model, not already scored online."""
    out = []
    for t in traces:
        name = (t.get("name") or "").lower()
        tags = t.get("tags") or []
        # LiteLLM sets trace name to the model name
        if "phi3-financial" in name or "phi3-financial" in tags:
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
