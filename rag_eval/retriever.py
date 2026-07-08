"""Retrieve chunks from Open WebUI /api/v1/retrieval/query/collection."""

import os
import requests


def retrieve_chunks(query: str, collection_name: str, k: int = 10) -> list[str]:
    base_url = os.environ["OPENWEBUI_BASE_URL"].rstrip("/")
    api_key = os.environ["OPENWEBUI_API_KEY"]

    resp = requests.post(
        f"{base_url}/api/v1/retrieval/query/collection",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"collection_name": collection_name, "query": query, "k": k},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Open WebUI returns ChromaDB-style nested lists: {"documents": [["chunk1", "chunk2"]]}
    raw = data.get("documents") or data.get("results", {}).get("documents") or []
    if raw and isinstance(raw[0], list):
        chunks = raw[0]
    else:
        chunks = raw

    return [str(c) for c in chunks if c]
