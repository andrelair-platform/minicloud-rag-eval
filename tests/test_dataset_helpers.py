"""Unit tests for the generic Langfuse dataset REST helpers (RTV-75), with `requests` stubbed."""

import pytest

from rag_eval import langfuse_reporter as lr


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "http://lf:3000")


def test_upsert_dataset_item_builds_the_right_request(monkeypatch):
    sent = {}

    def fake_post(url, auth=None, json=None, timeout=None):
        sent.update(url=url, auth=auth, json=json)
        return _Resp(200)

    monkeypatch.setattr(lr.requests, "post", fake_post)
    lr.upsert_dataset_item(
        "rag-chat-negatives",
        item_id="neg-t1",
        input_="why?",
        metadata={"traceId": "t1"},
        source_trace_id="t1",
    )
    assert sent["url"] == "http://lf:3000/api/public/dataset-items"
    assert sent["auth"] == ("pk", "sk")
    assert sent["json"]["id"] == "neg-t1"
    assert sent["json"]["datasetName"] == "rag-chat-negatives"
    assert sent["json"]["sourceTraceId"] == "t1"


def test_ensure_dataset_tolerates_already_exists(monkeypatch):
    monkeypatch.setattr(lr.requests, "post", lambda *a, **k: _Resp(409))
    lr.ensure_dataset("rag-chat-negatives")  # must not raise on 409


def test_list_source_trace_ids_paginates_and_dedupes(monkeypatch):
    pages = {
        1: _Resp(200, {"data": [{"sourceTraceId": "t1"}, {"metadata": {"traceId": "t2"}}] + [{} for _ in range(48)]}),
        2: _Resp(200, {"data": [{"sourceTraceId": "t3"}]}),
    }

    def fake_get(url, auth=None, params=None, timeout=None):
        return pages[params["page"]]

    monkeypatch.setattr(lr.requests, "get", fake_get)
    assert lr.list_dataset_source_trace_ids("rag-chat-negatives") == {"t1", "t2", "t3"}
