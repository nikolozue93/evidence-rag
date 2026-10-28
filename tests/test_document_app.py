"""Offline tests. Run: python -m pytest tests/test_document_app.py -q"""
import uuid
from types import SimpleNamespace
import pytest

from src.chunking import chunk_text, fixed_chunks, paragraph_chunks
from src.documents import content_hash, extract_text, make_chunks, DocumentError
from src.search_multi import search_scope
from app_investigation import app, require_token
from fastapi import HTTPException
from fastapi.testclient import TestClient


def test_chunking_both_strategies():
    text = (" ".join(f"alpha{i}" for i in range(60)) + "\n\n" +
            " ".join(f"beta{i}" for i in range(40)))
    fixed = fixed_chunks(text, words=30, overlap=5)
    para = paragraph_chunks(text, words=30)
    assert len(fixed) > 1 and len(para) > 1
    assert fixed[0].split()[-5:] == fixed[1].split()[:5]
    assert chunk_text(text, "paragraph", words=30) == para


def test_extract_and_dedup():
    text1, fmt = extract_text("note.MD", ("Intro\n\n" + "hello world " * 5).encode())
    text2, _ = extract_text("note.txt", ("Intro " + "hello world " * 5).encode())
    assert fmt == "md"
    assert content_hash(text1) == content_hash(text2)
    assert make_chunks(text1, "fixed")
    with pytest.raises(DocumentError):
        extract_text("evil.exe", b"lots of text " * 5)
    with pytest.raises(DocumentError):
        extract_text("bad.pdf", b"Not a real PDF")


def test_authenticated_api(monkeypatch):
    monkeypatch.setenv("DEMO_TOKEN", "test-private")
    client = TestClient(app)
    assert client.post("/api/ask", json={"question": "Which article?"}).status_code == 401
    assert client.post("/api/ask", json={"question": "Hi"}).status_code == 422
    assert client.get("/").status_code == 200


def test_search_scope_merges(monkeypatch):
    import src.search_multi as m
    monkeypatch.setattr(m, "execute_search", lambda db, query, mode, collection, k:
        [{"id": collection, "payload": {"title": collection, "text": "Evidence"}}])
    db = SimpleNamespace(collection_exists=lambda name: True)
    result = search_scope(db, "test", scope="both", mode="dense", k=8)
    assert len(result) == 2
    assert result[0]["id"] == "user_documents_v1"


def test_authorized_ask_route(monkeypatch):
    import app_investigation as m
    monkeypatch.setenv('DEMO_TOKEN', 'local-test-token')
    monkeypatch.setattr(m, 'investigate_question', lambda q: {
        'question': q.question, 'answer': 'Test answer [1]', 'status': 'cited',
        'search_queries': ['test'], 'sources': [{'id': 1, 'title': 'Source', 'text': 'Test answer'}],
    })
    client = TestClient(app)
    response = client.post('/api/ask', headers={'Authorization': 'Bearer local-test-token'},
                           json={'question': 'What caused the outage?', 'mode': 'hybrid', 'scope': 'uploaded'})
    assert response.status_code == 200, response.text
    assert response.json()['answer'] == 'Test answer [1]'


def test_document_routes_mocked(monkeypatch):
    import app_investigation as m
    monkeypatch.setenv('DEMO_TOKEN', 'local-test-token')
    monkeypatch.setattr(m, 'get_db', lambda: object())
    doc_id = str(uuid.uuid4())
    created = {'document_id': doc_id, 'filename': 'note.md', 'title': 'note', 'chunk_count': 1}
    monkeypatch.setattr(m, 'save_document', lambda *a, **k: created)
    monkeypatch.setattr(m, 'list_documents', lambda *a, **k: [created])
    monkeypatch.setattr(m, 'delete_document', lambda *a, **k: True)
    headers = {'Authorization': 'Bearer local-test-token'}
    client = TestClient(app)
    upload = client.post('/api/documents', headers=headers,
                         data={'strategy': 'paragraph'},
                         files={'file': ('note.md', b'The billing gateway was down on March 10 for 30 minutes.')})
    assert upload.status_code == 201, upload.text
    assert upload.json()['document_id'] == doc_id
    listed = client.get('/api/documents', headers=headers)
    assert len(listed.json()['documents']) == 1
    deleted = client.delete(f'/api/documents/{doc_id}', headers=headers)
    assert deleted.status_code == 200 and deleted.json()['deleted']
