import pytest
from fastapi.testclient import TestClient
from hub.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_post_segment_returns_201(client):
    payload = {
        "session_id": "s1",
        "participant_id": "p01",
        "start_time": 1.0,
        "end_time": 3.5,
        "text": "회의 시작합니다",
        "confidence": 0.95,
    }
    resp = client.post("/segments", json=payload)
    assert resp.status_code == 201


def test_get_merged_returns_sorted_list(client):
    client.post("/segments", json={
        "session_id": "s1", "participant_id": "p02",
        "start_time": 10.0, "end_time": 12.0,
        "text": "나중 발화", "confidence": 0.9,
    })
    client.post("/segments", json={
        "session_id": "s1", "participant_id": "p01",
        "start_time": 2.0, "end_time": 4.0,
        "text": "이른 발화", "confidence": 0.9,
    })
    resp = client.get("/sessions/s1/transcript")
    assert resp.status_code == 200
    data = resp.json()
    starts = [s["start_time"] for s in data]
    assert starts == sorted(starts)


def test_export_srt(client):
    resp = client.get("/sessions/s1/export?fmt=srt")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "WEBVTT" not in resp.text
    assert "-->" in resp.text


def test_unknown_session_returns_empty(client):
    resp = client.get("/sessions/nonexistent/transcript")
    assert resp.status_code == 200
    assert resp.json() == []
