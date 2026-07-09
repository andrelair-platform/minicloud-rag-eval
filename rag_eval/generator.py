"""Generate an answer from phi3-financial via LiteLLM and return (answer, trace_id)."""

import os
import json
import time
import requests


def generate_answer(query: str, chunks: list[str]) -> tuple[str, str]:
    base_url = os.environ["LITELLM_BASE_URL"].rstrip("/")
    api_key = os.environ["LITELLM_API_KEY"]
    model = os.environ.get("GENERATION_MODEL", "phi3-financial")

    context = "\n\n".join(f"[Source {i + 1}]: {c}" for i, c in enumerate(chunks))
    instruction = (
        "Réponds UNIQUEMENT en te basant sur le contexte fourni. "
        "N'ajoute aucune information absente du contexte. "
        "Si l'information n'est pas dans le contexte, indique-le clairement et brièvement.\n\n"
    )
    messages = [{"role": "user", "content": f"{instruction}Contexte:\n{context}\n\nQuestion : {query}"}]

    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Langfuse-Tags": json.dumps(["rag-eval"]),
                },
                json={"model": model, "messages": messages, "stream": False},
                timeout=420,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            trace_id = resp.headers.get("x-litellm-call-id", data.get("id", ""))
            return answer, trace_id
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as exc:
            last_exc = exc
            wait = 15 * (attempt + 1)
            print(f"  [gen-retry {attempt+1}/3] {type(exc).__name__}, retrying in {wait}s…", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"generate_answer failed after 3 attempts: {last_exc}") from last_exc
