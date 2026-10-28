"""Hybrid dense/BM25 search with *application-side* weighted RRF and title diversity.

This is our ranking component, rather than using Qdrant's built-in fusion.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

DENSE_MODEL = "sentence-transformers/all-minilm-l6-v2"
SPARSE_MODEL = "qdrant/bm25"


def weighted_rrf(dense_hits, sparse_hits, *, top_k=8,
                 dense_weight=0.55, sparse_weight=0.45,
                 rrf_k=60, max_per_title=2):
    """Fuse ordered hits by ID, then limit repeated titles (not scores).

    RRF scores cannot be interpreted as probabilities. Dense and BM25 raw
    similarities are NOT directly comparable, hence the use of rank positions.
    """
    entries = {}
    for weight, hits in ((dense_weight, dense_hits), (sparse_weight, sparse_hits)):
        for rank, hit in enumerate(hits, start=1):
            key = str(hit.id)
            item = entries.setdefault(key, {"id": key, "score": 0.0,
                                            "payload": hit.payload or {},
                                            "channels": []})
            item["score"] += weight / (rrf_k + rank)
            item["channels"].append("dense" if weight == dense_weight else "bm25")
    ranked = sorted(entries.values(), key=lambda x: (-x["score"], x["id"]))
    out = []
    per_title = {}
    for item in ranked:
        title = item["payload"].get("title", item["id"])
        if per_title.get(title, 0) >= max_per_title:
            continue
        per_title[title] = per_title.get(title, 0) + 1
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def hybrid_search(client, question, *, collection="hotpot_evidence_v1", k=8,
                  candidates=30):
    from qdrant_client import models
    dense = client.query_points(
        collection_name=collection,
        query=models.Document(text=question, model=DENSE_MODEL),
        using="dense", limit=candidates, with_payload=True,
    ).points
    sparse = client.query_points(
        collection_name=collection,
        query=models.Document(text=question, model=SPARSE_MODEL),
        using="bm25", limit=candidates, with_payload=True,
    ).points
    return weighted_rrf(dense, sparse, top_k=k)


def main():
    import argparse
    from qdrant_client import QdrantClient
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="Which magazine started first, Arthur's Magazine or First for Women?")
    parser.add_argument("--collection", default="hotpot_evidence_v1")
    args = parser.parse_args()
    load_dotenv()
    if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in .env")
    client = QdrantClient(url=os.environ["QDRANT_URL"],
                          api_key=os.environ["QDRANT_API_KEY"],
                          cloud_inference=True, timeout=60)
    for i, hit in enumerate(hybrid_search(client, args.question, collection=args.collection), start=1):
        p = hit["payload"]
        print(f"[{i}] {p.get('title','?')} | rrf={hit['score']:.5f} | {p.get('source_id','')[:10]}")
        print(p.get("text", "")[:200], "\n")


if __name__ == "__main__":
    main()
