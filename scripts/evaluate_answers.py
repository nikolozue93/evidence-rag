"""Resumable HotpotQA investigation answer evaluation.

Usage:
    python -u -m scripts.evaluate_answers --limit 40 --max-new 5
    python -u -m scripts.evaluate_answers --limit 40 --max-new 35

Gold labels are not indexed. Lexical match is a PROXY, not semantic correctness.
Source-ID citation checks are NOT claim-level entailment verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import time
from pathlib import Path

FIELDS = [
    "question_id", "mode", "question_type", "question", "gold_answer", "answer",
    "gold_in_answer", "token_f1", "citations_valid", "cited_gold_fraction",
    "all_gold_retrieved", "all_gold_cited", "gold_sources", "retrieved_sources",
    "cited_sources", "tool_searches", "status", "error",
]


def normalized_tokens(value: str) -> list[str]:
    value = re.sub(r"\[\d+\]", " ", value.casefold())
    return re.findall(r"\w+", value, flags=re.UNICODE)


def gold_in_answer(gold: str, predicted: str) -> float:
    """Conservative literal answer check; not a semantic judge."""
    g = normalized_tokens(gold)
    p = normalized_tokens(predicted)
    if not g or not p:
        return 0.0
    if g in (["yes"], ["no"]):
        # An isolated 'no' elsewhere in a long sentence must not count.
        return float(p[0] == g[0] or p[:2] == ["answer", g[0]])
    return float(any(p[i:i + len(g)] == g for i in range(len(p) - len(g) + 1)))


def token_f1(gold: str, predicted: str) -> float:
    """Lexical F1 on short reference vs full answer: supplementary proxy only."""
    from collections import Counter
    a, b = Counter(normalized_tokens(gold)), Counter(normalized_tokens(predicted))
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(b.values()), overlap / sum(a.values())
    return 2 * precision * recall / (precision + recall)


def score_result(item: dict, result: dict) -> dict:
    gold = set(map(str, item["supporting_source_ids"]))
    source_by_index = {int(s["id"]): str(s["source_id"]) for s in result["sources"]}
    retrieved = set(source_by_index.values())
    cited_numbers = set(map(int, re.findall(r"\[(\d+)\]", result["answer"])))
    valid_citations = cited_numbers & set(source_by_index)
    cited_sources = {source_by_index[i] for i in valid_citations}
    answer = result["answer"]
    return {
        "question_id": item["question_id"],
        "question_type": item.get("question_type", ""),
        "question": item["question"],
        "gold_answer": item["answer"],
        "answer": answer,
        "gold_in_answer": gold_in_answer(item["answer"], answer),
        "token_f1": round(token_f1(item["answer"], answer), 4),
        "citations_valid": float(bool(cited_numbers) and cited_numbers <= set(source_by_index)),
        "cited_gold_fraction": round(len(cited_sources & gold) / len(cited_sources), 4) if cited_sources else 0.,
        "all_gold_retrieved": float(bool(gold) and gold <= retrieved),
        "all_gold_cited": float(bool(gold) and gold <= cited_sources),
        "gold_sources": json.dumps(sorted(gold)),
        "retrieved_sources": json.dumps(sorted(retrieved)),
        "cited_sources": json.dumps(sorted(cited_sources)),
        "tool_searches": json.dumps(result.get("search_queries", []), ensure_ascii=False),
        "status": result.get("status", ""),
        "error": "",
    }


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {row["question_id"]: row for row in csv.DictReader(file)}


def save_rows(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement: don't lose earlier work if interrupted during a write.
    fd, name = tempfile.mkstemp(prefix=".answer_eval_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows.values())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def summarize(rows: dict[str, dict]):
    good = [r for r in rows.values() if not r.get("error")]
    if not good:
        return {"successes": 0, "errors": len(rows)}
    metrics = ("gold_in_answer", "citations_valid", "cited_gold_fraction", "all_gold_retrieved", "all_gold_cited")
    return {"successes": len(good), "errors": len(rows) - len(good),
            **{key: round(sum(float(r[key]) for r in good) / len(good), 4) for key in metrics}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/processed/hotpot_eval.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/answer_eval.csv"))
    parser.add_argument("--limit", type=int, default=40, help="First N held-out questions")
    parser.add_argument("--max-new", type=int, default=5, help="Maximum new API evaluations THIS run")
    parser.add_argument("--mode", choices=("hybrid", "dense"), default="hybrid")
    parser.add_argument("--rounds", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--cooldown", type=float, default=0., help="Seconds between questions, for rate limits")
    a = parser.parse_args()
    if a.limit < 1 or a.max_new < 1 or a.cooldown < 0:
        parser.error("Use positive --limit/--max-new and nonnegative --cooldown")

    from dotenv import load_dotenv
    load_dotenv()
    from groq import Groq
    from qdrant_client import QdrantClient
    from src.investigation import investigate
    required = ("QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY")
    if any(not os.getenv(k) for k in required):
        parser.error("Missing required QDRANT_URL, QDRANT_API_KEY or GROQ_API_KEY")

    questions = []
    with a.questions.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            if item.get("supporting_source_ids"):
                questions.append(item)
            if len(questions) >= a.limit:
                break
    rows = load_rows(a.out)
    if rows and any(r.get("mode") != a.mode for r in rows.values()):
        parser.error("Existing output was generated with a different retrieval mode; choose another --out filename")
    if rows and any(r.get("status") and r.get("status") not in ("", "cited", "uncited_or_invalid_citations", "no_evidence") for r in rows.values()):
        print("NOTE: Existing output file has unfamiliar status values; verify it before interpreting results.")
    db = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                      cloud_inference=True, timeout=90)
    groq = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=60.)
    new = 0
    for item in questions:
        qid = str(item["question_id"])
        if qid in rows and not rows[qid].get("error"):
            continue
        if new >= a.max_new:
            break
        try:
            result = investigate(item["question"], db, groq, mode=a.mode,
                                 search_rounds=a.rounds,
                                 model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
            rows[qid] = {**score_result(item, result), "mode": a.mode}
            print(f"PASS {new + 1}/{a.max_new}: {qid} | gold_in_answer={rows[qid]['gold_in_answer']} | all_gold_cited={rows[qid]['all_gold_cited']}", flush=True)
        except Exception as exc:
            rows[qid] = {**{k: "" for k in FIELDS}, "question_id": qid,
                         "mode": a.mode, "question_type": item.get("question_type", ""),
                         "question": item["question"], "gold_answer": item["answer"],
                         "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
            print(f"ERROR {qid}: {rows[qid]['error']}", flush=True)
            save_rows(a.out, rows)
            print("Stopped on error. Fix API/rate limits and rerun; successful rows are skipped.")
            break
        new += 1
        save_rows(a.out, rows)
        if a.cooldown:
            time.sleep(a.cooldown)
    print("SUMMARY:", json.dumps(summarize(rows), indent=2))
    print("Saved:", a.out)
    print("NOTE: gold_in_answer is lexical, NOT a semantic correctness judgment;")
    print("cited_gold_fraction verifies known gold source IDs, NOT sentence-level factual entailment.")


if __name__ == "__main__":
    main()
