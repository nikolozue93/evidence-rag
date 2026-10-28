"""Persistent three-format document CRUD backed by Qdrant.

HotpotQA is untouched in hotpot_evidence_v1. User documents are stored as dense
and BM25 chunk points; document registry lives in its own tiny Qdrant collection.
Raw uploaded files are NOT retained; only extracted text and metadata persist.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, io, uuid
from pathlib import Path
from src.chunking import chunk_text, normalize

CHUNKS_COLLECTION = "user_documents_v1"
REGISTRY_COLLECTION = "user_document_registry_v1"
DENSE_MODEL = "sentence-transformers/all-minilm-l6-v2"
SPARSE_MODEL = "qdrant/bm25"
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_CHUNKS = 180


class DocumentError(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError("File exceeds the 5 MB upload limit")
    if not data:
        raise DocumentError("File is empty")
    if suffix in (".txt", ".md"):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError("Text files must be UTF-8 encoded") from exc
    elif suffix == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            if len(reader.pages) > 150:
                raise DocumentError("PDF has more than 150 pages")
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("Could not read PDF. Password-protected and malformed PDFs are unsupported") from exc
    else:
        raise DocumentError("Supported types: .pdf, .txt, .md")
    if len(normalize(text)) < 30:
        raise DocumentError("Insufficient extracted text. Scanned PDFs require OCR (not supported)")
    if len(text) > 250_000:
        raise DocumentError("Extracted document exceeds 250,000 characters")
    return text, suffix[1:]


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def make_chunks(text: str, strategy: str, *, chunk_words: int = 180) -> list[str]:
    chunks = chunk_text(text, strategy, words=chunk_words)
    if not chunks:
        raise DocumentError("Text did not produce any chunks")
    if len(chunks) > MAX_CHUNKS:
        raise DocumentError(f"Document would require {len(chunks)} chunks; limit is {MAX_CHUNKS}")
    return chunks


def build_chunk_points(doc_id: str, title: str, filename: str, source_type: str,
                       chunks: list[str], revision: str, strategy: str):
    from qdrant_client import models
    items = []
    for i, text in enumerate(chunks):
        sid = f"upload:{doc_id}:{revision}:{i}"
        items.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, sid)),
            vector={"dense": models.Document(text=f"{title}\n{text}", model=DENSE_MODEL),
                    "bm25": models.Document(text=f"{title}\n{text}", model=SPARSE_MODEL)},
            payload={"source_id": sid, "document_id": doc_id, "title": title,
                     "filename": filename, "source_type": source_type,
                     "text": text, "chunk_index": i, "revision": revision,
                     "chunk_strategy": strategy, "record_type": "chunk"},
        ))
    return items


def create_collections(db) -> None:
    from qdrant_client import models
    if not db.collection_exists(CHUNKS_COLLECTION):
        db.create_collection(
            collection_name=CHUNKS_COLLECTION,
            vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE,
                                                          on_disk=True)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=True))},
        )
    if not db.collection_exists(REGISTRY_COLLECTION):
        db.create_collection(
            collection_name=REGISTRY_COLLECTION,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
        )
    # Qdrant Cloud may require payload indexes for filtered deletion (strict mode).
    # Do this for both NEW and EXISTING collections, not only on first creation.
    existing_indexes = db.get_collection(CHUNKS_COLLECTION).payload_schema or {}
    for field in ("document_id", "revision"):
        if field not in existing_indexes:
            db.create_payload_index(
                collection_name=CHUNKS_COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )


def get_document(db, document_id: str) -> dict | None:
    try:
        uuid.UUID(document_id)
    except ValueError as exc:
        raise DocumentError("Invalid document ID") from exc
    if not db.collection_exists(REGISTRY_COLLECTION):
        return None
    records = db.retrieve(collection_name=REGISTRY_COLLECTION, ids=[document_id],
                          with_payload=True, with_vectors=False)
    return (records[0].payload or {}) if records else None


def list_documents(db, *, max_docs: int = 200) -> list[dict]:
    if not db.collection_exists(REGISTRY_COLLECTION):
        return []
    docs, offset = db.scroll(collection_name=REGISTRY_COLLECTION, limit=min(max_docs, 100),
                             with_payload=True, with_vectors=False)
    all_docs = [p.payload or {} for p in docs]
    while offset is not None and len(all_docs) < max_docs:
        docs, offset = db.scroll(collection_name=REGISTRY_COLLECTION, offset=offset,
                                 limit=min(100, max_docs - len(all_docs)),
                                 with_payload=True, with_vectors=False)
        all_docs.extend(p.payload or {} for p in docs)
    return sorted(all_docs, key=lambda x: x.get("uploaded_at", ""), reverse=True)


def _delete_revision(db, document_id: str, revision: str) -> None:
    from qdrant_client import models
    db.delete(collection_name=CHUNKS_COLLECTION,
              points_selector=models.FilterSelector(filter=models.Filter(must=[
                  models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                  models.FieldCondition(key="revision", match=models.MatchValue(value=revision)),
              ])), wait=True)


def save_document(db, filename: str, data: bytes, *, strategy: str = "paragraph",
                  document_id: str | None = None, title: str | None = None,
                  chunk_words: int = 180) -> dict:
    """Create or replace document, upserting new chunks before updating the registry.

    On an update old chunks are removed only after registry update succeeds.
    No cross-collection transaction is claimed; concurrent writes to one document
    should be avoided and partial errors may require a retry/cleanup.
    """
    text, source_type = extract_text(filename, data)
    digest = content_hash(text)
    chunks = make_chunks(text, strategy, chunk_words=chunk_words)
    title = (title or Path(filename).stem).strip()[:150]
    if not title:
        raise DocumentError("Missing title")
    if len(filename) > 200:
        raise DocumentError("Filename too long")
    create_collections(db)
    previous = get_document(db, document_id) if document_id else None
    if document_id and not previous:
        raise DocumentError("Document to update does not exist")
    # Deduplicate exact normalized content across uploaded docs (not HotpotQA).
    for item in list_documents(db):
        if item.get("content_sha256") == digest and item.get("document_id") != document_id:
            raise DocumentError(f"Duplicate content: already uploaded as {item.get('document_id')}")
    if previous and previous.get("content_sha256") == digest and previous.get("chunk_strategy") == strategy:
        return {**previous, "unchanged": True}
    if not document_id:
        document_id = str(uuid.uuid4())
    revision = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    points = build_chunk_points(document_id, title, filename, source_type, chunks, revision, strategy)
    db.upload_points(collection_name=CHUNKS_COLLECTION, points=points, batch_size=12,
                     parallel=1, max_retries=2, wait=True)
    meta = {"document_id": document_id, "title": title, "filename": filename,
            "source_type": source_type, "content_sha256": digest,
            "chunk_strategy": strategy, "chunk_words": chunk_words,
            "chunk_count": len(chunks), "revision": revision, "uploaded_at": now}
    from qdrant_client import models
    db.upsert(collection_name=REGISTRY_COLLECTION, points=[models.PointStruct(
        id=document_id, vector=[1.0], payload=meta)], wait=True)
    if previous:
        _delete_revision(db, document_id, previous["revision"])
    return meta


def delete_document(db, document_id: str) -> bool:
    previous = get_document(db, document_id)
    if not previous:
        return False
    create_collections(db)  # Upgrade existing collections if indexes were missing.
    # Remove ALL revisions, including chunks left behind by a failed update.
    from qdrant_client import models
    db.delete(collection_name=CHUNKS_COLLECTION,
              points_selector=models.FilterSelector(filter=models.Filter(must=[
                  models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
              ])), wait=True)
    db.delete(collection_name=REGISTRY_COLLECTION, points_selector=[document_id], wait=True)
    return True
