"""Measure retrieval on HotpotQA held-out questions, using known support IDs.

Compares dense and hybrid search on the SAME questions. Does not test answer
correctness or fabricated unanswerable/conflicting cases; report separately.
"""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
from dotenv import load_dotenv
from src.investigation import execute_search


def scores(hits, gold_source_ids):
    gold = set(gold_source_ids)
    retrieved = [str(h.get("payload", {}).get("source_id", "")) for h in hits]
    found = sum(x in retrieved for x in gold)
    first_rank = next((i for i, sid in enumerate(retrieved, 1) if sid in gold), 0)
    return {"recall": found / len(gold) if gold else 0.0,
            "hit": float(found > 0),
            "all_supporting_hit": float(found == len(gold) and bool(gold)),
            "mrr": 1 / first_rank if first_rank else 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path,
                   default=Path("data/processed/hotpot_eval.jsonl"))
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("reports/retrieval_eval.csv"))
    p.add_argument("--collection", default="hotpot_evidence_v1")
    a = p.parse_args()
    load_dotenv()
    from qdrant_client import QdrantClient
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                          cloud_inference=True, timeout=90)
    questions = []
    with a.questions.open(encoding="utf-8") as f:
        for line in f:
            if len(questions) >= a.limit:
                break
            item = json.loads(line)
            if item.get("supporting_source_ids"):
                questions.append(item)
    if len(questions) < 40:
        print(f"WARNING: only {len(questions)} supported QA examples available")
    rows = []
    for i, item in enumerate(questions, 1):
        for mode in ("dense", "hybrid"):
            try:
                hits = execute_search(client, item["question"], mode=mode,
                                      collection=a.collection, k=a.k)
                m = scores(hits, item["supporting_source_ids"])
                rows.append({"question_id": item["question_id"], "question": item["question"],
                             "type": item.get("question_type", ""), "mode": mode,
                             "k": a.k, **m, "error": ""})
            except Exception as exc:
                rows.append({"question_id": item["question_id"], "question": item["question"],
                             "type": item.get("question_type", ""), "mode": mode,
                             "k": a.k, "recall": 0, "hit": 0, "all_supporting_hit": 0,
                             "mrr": 0, "error": f"{type(exc).__name__}: {str(exc)[:150]}"})
        if i % 5 == 0:
            print(f"Evaluated {i}/{len(questions)} questions", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for mode in ("dense", "hybrid"):
        rr = [r for r in rows if r["mode"] == mode and not r["error"]]
        if rr:
            metrics = {
                key: round(sum(r[key] for r in rr) / len(rr), 4)
                for key in ("recall", "hit", "all_supporting_hit", "mrr")
            }

            print(f"{mode} | n={len(rr)} | {metrics}")
    print("Saved:", a.out)


if __name__ == "__main__":
    main()
