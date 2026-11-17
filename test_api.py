from fastapi.testclient import TestClient

from app.evaluation import keyword_recall
from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_required():
    assert client.post("/ask", json={"question": "hello there"}).status_code == 401


def test_keyword_recall():
    assert keyword_recall("RAG uses embeddings", ["rag", "embeddings"]) == 1.0
    assert keyword_recall("RAG only", ["rag", "embeddings"]) == 0.5
