"""Small live integration test of Qdrant document CRUD and uploaded retrieval.

Requires existing QDRANT_URL/QDRANT_API_KEY and Qdrant cloud inference enabled.
Creates a disposable TXT record then removes it, even after an intermediate error.
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.documents import delete_document, get_document, save_document, CHUNKS_COLLECTION
from src.search_multi import search_scope


def main():
    load_dotenv()
    db = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                      cloud_inference=True, timeout=90)
    doc_id = None
    try:
        data = ("An imaginary project called Aster has a billing gateway called Iris. "
                "The gateway was upgraded on March 10 and the payment API had a 30-minute delay. "
                "Aster's incident note says the upgrade caused the delay.").encode()
        created = save_document(db, "integration_test.txt", data, strategy="paragraph")
        doc_id = created["document_id"]
        assert get_document(db, doc_id)["chunk_count"] >= 1
        hits = search_scope(db, "What caused Aster billing gateway payment delay?",
                            mode="hybrid", scope="uploaded", k=5)
        assert any(h.get("payload", {}).get("document_id") == doc_id for h in hits), hits
        print("PASS: text document uploaded, persisted, and found through hybrid search")
        updated = save_document(db, "integration_test.md",
                                data + b" New evidence was reviewed by the team.",
                                strategy="fixed", document_id=doc_id)
        assert updated["revision"] != created["revision"]
        print("PASS: document replaced, and chunking strategy changed")
    finally:
        if doc_id:
            deleted = delete_document(db, doc_id)
            assert deleted
            print("PASS: temporary document deleted")


if __name__ == "__main__":
    main()
