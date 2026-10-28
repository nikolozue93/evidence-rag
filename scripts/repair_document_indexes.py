"""One-time upgrade and optional cleanup of smoke-test documents.

Run from project root:
    python -m scripts.repair_document_indexes
    python -m scripts.repair_document_indexes --cleanup-smoke

Does NOT change the HotpotQA collection. Only deletes explicitly named smoke-test
records after creating the required payload indexes.
"""
import argparse
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.documents import (create_collections, list_documents, CHUNKS_COLLECTION,
                           REGISTRY_COLLECTION)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup-smoke', action='store_true')
    args = parser.parse_args()
    load_dotenv()
    db = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'],
                      cloud_inference=True, timeout=90)
    create_collections(db)
    print('PASS: document_id and revision indexes are available in', CHUNKS_COLLECTION)
    print('Indexed fields:', ', '.join(sorted((db.get_collection(CHUNKS_COLLECTION).payload_schema or {}).keys())))
    if args.cleanup_smoke:
        for item in list_documents(db):
            # Only artifacts created by the known integration smoke test.
            if (item.get('title') == 'integration_test' and
                    item.get('filename') in {'integration_test.txt', 'integration_test.md'}):
                doc_id = item['document_id']
                db.delete(collection_name=CHUNKS_COLLECTION,
                          points_selector=models.FilterSelector(filter=models.Filter(must=[
                              models.FieldCondition(key='document_id',
                                                    match=models.MatchValue(value=doc_id)),
                          ])), wait=True)
                db.delete(collection_name=REGISTRY_COLLECTION,
                          points_selector=[doc_id], wait=True)
                print('Cleaned up smoke-test document:', doc_id)


if __name__ == '__main__':
    main()
