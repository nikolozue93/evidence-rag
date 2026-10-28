"""Compare chunking strategies on sample source passages WITHOUT reindexing Qdrant.

Structural comparison only, not retrieval accuracy. Run:
    python -m scripts.compare_chunking --max-docs 300
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from pathlib import Path

from src.chunking import chunk_text


def calculate(texts: list[str], words: int = 180) -> list[dict]:
    out = []
    for strategy in ("fixed", "paragraph"):
        per_doc = [chunk_text(text, strategy=strategy, words=words) for text in texts]
        lengths = [len(chunk.split()) for doc in per_doc for chunk in doc]
        total_words = sum(len(text.split()) for text in texts)
        out.append({"strategy": strategy, "documents": len(texts), "chunks": len(lengths),
                    "avg_chunks_per_document": round(sum(map(len, per_doc)) / len(texts), 3) if texts else 0,
                    "mean_chunk_words": round(statistics.mean(lengths), 2) if lengths else 0,
                    "median_chunk_words": round(statistics.median(lengths), 2) if lengths else 0,
                    "min_chunk_words": min(lengths, default=0), "max_chunk_words": max(lengths, default=0),
                    "chunk_words_total": sum(lengths), "original_words_total": total_words,
                    "redundancy_ratio": round(sum(lengths) / total_words, 3) if total_words else 0})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/processed/hotpot_records.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/chunking_comparison.csv"))
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--words", type=int, default=180)
    args = parser.parse_args()
    if args.max_docs < 1 or args.words < 1:
        parser.error("--max-docs and --words must be positive")
    texts = []
    with args.source.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("text", "").strip():
                texts.append(record["text"])
            if len(texts) >= args.max_docs:
                break
    rows = calculate(texts, words=args.words)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("Saved:", args.out)
    print("NOTE: descriptive comparison ONLY; no search-quality conclusion can be drawn from this output.")


if __name__ == "__main__":
    main()
