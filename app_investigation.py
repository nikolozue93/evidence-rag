"""Run: uvicorn app_investigation:app --host 0.0.0.0 --port 8000

Standalone application: does not overwrite the earlier `main.py` probe.
A DEMO_TOKEN is required for all cost-incurring operations and all mutations.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
import hmac
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.documents import (DocumentError, MAX_FILE_BYTES, delete_document,
                           get_document, list_documents, save_document)
from src.investigation import investigate
from src.search_multi import search_scope

load_dotenv()
app = FastAPI(title="Evidence Investigation Engine", version="0.2.0",
              description="Multi-source RAG with persistent document CRUD")
WEB = Path(__file__).parent / "static" / "index.html"


def require_token(authorization: str | None):
    configured = os.getenv("DEMO_TOKEN", "")
    if not configured:
        raise HTTPException(503, "DEMO_TOKEN is not configured on the server")
    supplied = (authorization or "").removeprefix("Bearer ")
    if not hmac.compare_digest(configured, supplied):
        raise HTTPException(401, "Missing or invalid demo token")


@lru_cache(maxsize=1)
def get_db():
    from qdrant_client import QdrantClient
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                        cloud_inference=True, timeout=90)


@lru_cache(maxsize=1)
def get_llm():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"], timeout=45.0)


@app.get("/")
def ui():
    return FileResponse(WEB, media_type="text/html")


@app.get("/health")
def health():
    try:
        db = get_db()
        info = db.get_collection("hotpot_evidence_v1")
        return {"status": "ok", "hotpot_records": info.points_count,
                "document_collection_exists": db.collection_exists("user_documents_v1")}
    except Exception as exc:
        raise HTTPException(503, "Qdrant unavailable; check server logs") from exc


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["hybrid", "dense"] = "hybrid"
    scope: Literal["both", "hotpot", "uploaded"] = "both"
    rounds: int = Field(default=2, ge=1, le=3)


def investigate_question(item: Question):
    db = get_db()
    return investigate(
        item.question, db, get_llm(), mode=item.mode,
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        search_rounds=item.rounds,
        search_fn=lambda query: search_scope(db, query, mode=item.mode, scope=item.scope, k=8),
    )


@app.post("/api/ask")
async def ask(item: Question, authorization: str | None = Header(None)):
    require_token(authorization)
    try:
        return await asyncio.to_thread(investigate_question, item)
    except Exception as exc:
        # Avoid returning internal infrastructure/key details to the browser.
        raise HTTPException(502, f"Investigation failed ({type(exc).__name__}); inspect server logs") from exc


@app.get("/api/documents")
async def documents(authorization: str | None = Header(None)):
    require_token(authorization)
    try:
        return {"documents": await asyncio.to_thread(list_documents, get_db())}
    except Exception as exc:
        raise HTTPException(502, "Could not list documents") from exc


@app.get("/api/documents/{document_id}")
async def document_detail(document_id: str, authorization: str | None = Header(None)):
    require_token(authorization)
    try:
        doc = await asyncio.to_thread(get_document, get_db(), document_id)
    except DocumentError as exc:
        raise HTTPException(400, str(exc)) from exc
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc


async def _read_file(file: UploadFile, request: Request) -> tuple[str, bytes]:
    if request.headers.get("content-length"):
        try:
            if int(request.headers["content-length"]) > MAX_FILE_BYTES + 250_000:
                raise HTTPException(413, "Upload too large")
        except ValueError:
            pass
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "File exceeds 5 MB")
    return file.filename or "untitled.txt", data


@app.post("/api/documents", status_code=201)
async def upload_document(request: Request, file: UploadFile = File(...),
                          strategy: Literal["fixed", "paragraph"] = Form("paragraph"),
                          authorization: str | None = Header(None)):
    require_token(authorization)
    name, data = await _read_file(file, request)
    try:
        return await asyncio.to_thread(save_document, get_db(), name, data, strategy=strategy)
    except DocumentError as exc:
        raise HTTPException(409 if "Duplicate" in str(exc) else 400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, "Document indexing failed; inspect server logs") from exc


@app.put("/api/documents/{document_id}")
async def replace_document(document_id: str, request: Request,
                           file: UploadFile = File(...),
                           strategy: Literal["fixed", "paragraph"] = Form("paragraph"),
                           authorization: str | None = Header(None)):
    require_token(authorization)
    name, data = await _read_file(file, request)
    try:
        return await asyncio.to_thread(save_document, get_db(), name, data,
                                       strategy=strategy, document_id=document_id)
    except DocumentError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, "Document update failed; inspect server logs") from exc


@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str, authorization: str | None = Header(None)):
    require_token(authorization)
    try:
        deleted = await asyncio.to_thread(delete_document, get_db(), document_id)
    except DocumentError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, "Document deletion failed; inspect server logs") from exc
    if not deleted:
        raise HTTPException(404, "Document not found")
    return {"deleted": True, "document_id": document_id}
