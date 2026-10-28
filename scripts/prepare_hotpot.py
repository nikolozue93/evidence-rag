"""Build an exact-deduplicated HotpotQA passage corpus and held-out QA set.

The original HotpotQA context passages are the source records. Questions/answers
are stored separately for evaluation and never indexed as source evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sqlite3
from pathlib import Path


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def passage_hash(text: str) -> str:
    # content-only deduplication, same text in different questions counts once.
    return hashlib.sha256(normalize_text(text).casefold().encode("utf-8")).hexdigest()


def context_passages(example: dict):
    context = example["context"]
    for title, sentences in zip(context["title"], context["sentences"]):
        body = normalize_text(" ".join(sentences))
        if len(body) >= 80:
            yield normalize_text(title), body


def build_corpus(train_examples, validation_examples, *, records_path: Path,
                 eval_path: Path, target_records: int, eval_questions: int,
                 progress_every: int = 2500) -> dict:
    """Accept iterables (including streaming HF datasets) for easy testing."""
    records_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    seen_path = records_path.with_suffix(".seen.sqlite")
    seen_path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(seen_path))
    conn.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY)")
    count = 0
    questions = 0
    train_processed = 0

    def add_passage(title, content, example_id, writer, source_split):
        nonlocal count
        digest = passage_hash(content)
        before = conn.total_changes
        conn.execute("INSERT OR IGNORE INTO seen(hash) VALUES (?)", (digest,))
        if conn.total_changes == before:
            return digest
        record = {
            "source_id": digest,
            "title": title,
            "text": content,
            "source_type": "hotpotqa",
            "origin_split": source_split,
            "origin_example_id": str(example_id),
        }
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        if count % progress_every == 0:
            conn.commit()
            print(f"Unique source passages: {count:,}", flush=True)
        return digest

    with records_path.open("w", encoding="utf-8") as out, \
         eval_path.open("w", encoding="utf-8") as eval_out:
        for example in train_examples:
            train_processed += 1
            for title, content in context_passages(example):
                add_passage(title, content, example["id"], out, "train")
            if count >= target_records:
                break
        if count < target_records:
            raise RuntimeError(
                f"Only {count:,} distinct passages after {train_processed:,} train examples; "
                f"requested {target_records:,}."
            )
        conn.commit()
        print(f"Train complete: {count:,} unique passages from {train_processed:,} examples", flush=True)

        # Validation questions remain held out from model/prompt selection.
        # Their source texts ARE indexed so retrieval is evaluable, but the
        # question/answer labels and supporting annotations never enter Qdrant.
        for example in validation_examples:
            if questions >= eval_questions:
                break
            evidence_by_title = {}
            for title, content in context_passages(example):
                digest = add_passage(title, content, example["id"], out, "validation")
                evidence_by_title.setdefault(title, []).append(digest)
            supporting_titles = sorted(set(example["supporting_facts"]["title"]))
            gold_ids = sorted({hash_id for title in supporting_titles
                               for hash_id in evidence_by_title.get(title, [])})
            if len(gold_ids) < len(supporting_titles):
                # Skip questions with unsupported/empty passages.
                continue
            evaluation = {
                "question_id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "question_type": example.get("type", ""),
                "difficulty": example.get("level", ""),
                "supporting_titles": supporting_titles,
                "supporting_source_ids": gold_ids,
            }
            eval_out.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
            questions += 1
    conn.commit()
    conn.close()
    summary = {
        "unique_passage_records": count,
        "train_examples_processed": train_processed,
        "validation_questions_saved": questions,
        "records_path": str(records_path),
        "evaluation_path": str(eval_path),
        "deduplication": "SHA256(normalized, casefolded passage text)",
        "minimum_passage_chars": 80,
    }
    records_path.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-records", type=int, default=50_000)
    parser.add_argument("--eval-questions", type=int, default=60)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    if args.target_records < 1 or args.eval_questions < 1:
        parser.error("Targets must be positive")

    from datasets import load_dataset
    print("Streaming HotpotQA distractor/train and distractor/validation...", flush=True)
    train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
    validation = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    summary = build_corpus(
        train, validation,
        records_path=args.out_dir / "hotpot_records.jsonl",
        eval_path=args.out_dir / "hotpot_eval.jsonl",
        target_records=args.target_records,
        eval_questions=args.eval_questions,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
