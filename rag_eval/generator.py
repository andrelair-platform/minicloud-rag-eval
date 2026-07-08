"""Generate an answer from phi3-financial via LiteLLM and return (answer, trace_id)."""

import os
import json
import requests


def generate_answer(query: str, chunks: list[str]) -> tuple[str, str]:
    base_url = os.environ["LITELLM_BASE_URL"].rstrip("/")
    api_key = os.environ["LITELLM_API_KEY"]
    model = os.environ.get("GENERATION_MODEL", "phi3-financial")

    context = "\n\n".join(f"[Source {i + 1}]: {c}" for i, c in enumerate(chunks))
    messages = [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Tag traces so Langfuse online_sampler can filter them
            "X-Langfuse-Tags": json.dumps(["rag-eval"]),
        },
        json={"model": model, "messages": messages, "stream": False},
        timeout=360,
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    trace_id = resp.headers.get("x-litellm-call-id", data.get("id", ""))
    return answer, trace_id
