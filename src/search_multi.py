"""Compose the existing hybrid retriever across HotpotQA and uploaded documents."""
from __future__ import annotations

from src.retrieval import hybrid_search
from src.investigation import execute_search
from src.documents import CHUNKS_COLLECTION

HOTPOT_COLLECTION = "hotpot_evidence_v1"


def search_scope(db, query: str, *, mode: str = "hybrid", scope: str = "both", k: int = 8):
    if scope not in ("hotpot", "uploaded", "both"):
        raise ValueError("scope must be hotpot, uploaded, or both")
    if mode not in ("hybrid", "dense"):
        raise ValueError("mode must be hybrid or dense")

    def one(collection: str, n: int):
        if not db.collection_exists(collection):
            return []
        return execute_search(db, query, mode=mode, collection=collection, k=n)

    if scope == "hotpot":
        return one(HOTPOT_COLLECTION, k)
    if scope == "uploaded":
        return one(CHUNKS_COLLECTION, k)

    # Two independently ranked collections. A round-robin mix gives the user
    # docs a fair chance alongside a much larger Hotpot corpus. It is a
    # heuristic, NOT a calibrated ranking or directly comparable RRF score.
    user_hits = one(CHUNKS_COLLECTION, k)
    hotpot_hits = one(HOTPOT_COLLECTION, k)
    out = []
    seen = set()
    for index in range(max(len(user_hits), len(hotpot_hits))):
        for hits in (user_hits, hotpot_hits):
            if index >= len(hits):
                continue
            hit = hits[index]
            identity = str(hit.get("id"))
            if identity not in seen:
                seen.add(identity)
                out.append(hit)
                if len(out) >= k:
                    return out
    return out
