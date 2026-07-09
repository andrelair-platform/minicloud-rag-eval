"""Retrieve chunks from Open WebUI /api/v1/retrieval/query/collection."""

import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=3,          # 3s, 6s, 12s, 24s, 48s
        status_forcelist=[502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def retrieve_chunks(query: str, collection_name: str, k: int = 10) -> list[str]:
    base_url = os.environ["OPENWEBUI_BASE_URL"].rstrip("/")
    api_key = os.environ["OPENWEBUI_API_KEY"]

    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            resp = _session().post(
                f"{base_url}/api/v1/retrieval/query/collection",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"collection_names": [collection_name], "query": query, "k": k},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("documents") or data.get("results", {}).get("documents") or []
            if raw and isinstance(raw[0], list):
                chunks = raw[0]
            else:
                chunks = raw
            return [str(c) for c in chunks if c]
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
            wait = 10 * (attempt + 1)
            print(f"  [retry {attempt+1}/5] Open WebUI connection error, retrying in {wait}s…")
            time.sleep(wait)

    raise RuntimeError(f"retrieve_chunks failed after 5 attempts: {last_exc}") from last_exc
