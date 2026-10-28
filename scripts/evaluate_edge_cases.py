"""Evaluate LLM uncertainty and disagreement handling using controlled evidence.

Uses the REAL Groq LLM and actual structured tool calls, but intentionally uses
fixture evidence instead of Qdrant. This is a reasoning-only stress test, NOT
end-to-end retrieval validation. Results are automatically checked by keywords
and should be manually reviewed for faithful source interpretation.

Run: python -u -m scripts.evaluate_edge_cases
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

# The source texts deliberately contain no real-world claims.
CASES = [
    {"id": "missing_1", "kind": "unanswerable",
     "question": "Who signed off the Orion Ridge migration on 17 March?",
     "sources": [("Orion Ridge schedule", "Orion Ridge migration was planned for 17 March. The schedule does not record any sign-off or approval decision.")]},
    {"id": "missing_2", "kind": "unanswerable",
     "question": "What was the exact financial loss from the Sapphire Queue outage?",
     "sources": [("Sapphire incident note", "Sapphire Queue had a processing delay. The incident note reports no financial loss estimate or monetary figures.")]},
    {"id": "missing_3", "kind": "unanswerable",
     "question": "Which engineer fixed the Amber Relay incident?",
     "sources": [("Amber Relay change log", "Amber Relay was restored at 15:20 after a configuration rollback. The author and responder are not identified.")]},
    {"id": "missing_4", "kind": "unanswerable",
     "question": "What was the exact root cause of the Delta Crate timeout?",
     "sources": [("Delta Crate triage", "Delta Crate requests timed out. The team is still investigating the cause. The root cause is unknown.")]},
    {"id": "conflict_1", "kind": "contradictory",
     "question": "Which vendor processed the fictional Indigo Harbor settlement on 4 April?",
     "sources": [("Indigo Harbor report A", "The Indigo Harbor settlement on 4 April was processed by Horizon Systems."),
                 ("Indigo Harbor report B", "The Indigo Harbor settlement on 4 April was processed by Meridian Analytics. This note does not mention Horizon Systems.")]},
    {"id": "conflict_2", "kind": "contradictory",
     "question": "How many nodes failed during the fictional Cedar Beacon incident?",
     "sources": [("Cedar Beacon operations log", "Exactly 3 nodes failed during the Cedar Beacon incident."),
                 ("Cedar Beacon incident report", "Exactly 7 nodes failed during the Cedar Beacon incident.")]},
    {"id": "conflict_3", "kind": "contradictory",
     "question": "On what date did the fictional Cobalt Finch service launch?",
     "sources": [("Cobalt Finch launch memo", "Cobalt Finch launched on 12 June 2024."),
                 ("Cobalt Finch product note", "Cobalt Finch launched on 18 June 2024.")]},
    {"id": "conflict_4", "kind": "contradictory",
     "question": "What caused the fictional Violet Gate outage?",
     "sources": [("Violet Gate report 1", "The Violet Gate outage was caused by a malformed cache configuration."),
                 ("Violet Gate report 2", "The Violet Gate outage was caused by a failed database migration, not a cache configuration.")]},
]


def make_search(case: dict):
    """All evidence always available regardless of query; isolates LLM behavior."""
    docs = []
    for index, (title, body) in enumerate(case["sources"]):
        sid = f"fixture:{case['id']}:{index}"
        docs.append({"id": sid, "payload": {"source_id": sid, "title": title,
                    "text": body, "source_type": "synthetic_fixture"}})
    return lambda query: docs


def classify(kind: str, answer: str, citation_count: int) -> bool:
    a = answer.casefold()
    if kind == "unanswerable":
        return bool(re.search(r"insufficient evidence|cannot determine|can't determine|not (?:specified|provided|known|available|recorded)|unknown|no evidence|does not (?:identify|state|specify)", a))
    if kind == "contradictory":
        return bool(re.search(r"conflicting evidence|conflict|contradict|disagree|inconsistent|different accounts|discrepancy", a)) and citation_count >= 2
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/edge_case_eval.csv"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    from groq import Groq
    from src.investigation import investigate
    if not os.getenv("GROQ_API_KEY"):
        parser.error("GROQ_API_KEY is required")
    groq = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=60.)
    fields = ("case_id", "kind", "question", "answer", "expected_behavior_detected", "citations_valid", "citations_used", "error")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if args.out.exists():
        with args.out.open(newline="", encoding="utf-8") as file:
            previous = {r["case_id"]: r for r in csv.DictReader(file)}
    rows_by_id = previous.copy()
    for case in CASES[:args.limit]:
        if case["id"] in rows_by_id and not rows_by_id[case["id"]].get("error"):
            print(f"SKIP {case['id']} (already evaluated)")
            continue
        try:
            r = investigate(case["question"], None, groq, search_fn=make_search(case),
                            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), search_rounds=args.rounds)
            citation_count = len(r["citation_ids_used"])
            row = {"case_id": case["id"], "kind": case["kind"], "question": case["question"],
                   "answer": r["answer"],
                   "expected_behavior_detected": int(classify(case["kind"], r["answer"], citation_count)),
                   "citations_valid": int(not r["invalid_citation_ids"] and citation_count > 0),
                   "citations_used": citation_count, "error": ""}
        except Exception as exc:
            row = {"case_id": case["id"], "kind": case["kind"], "question": case["question"],
                   "answer": "", "expected_behavior_detected": 0, "citations_valid": 0,
                   "citations_used": 0, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        rows_by_id[case["id"]] = row
        with args.out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows_by_id.values())
        print(f"{case['id']} | {case['kind']} | check={row['expected_behavior_detected']} | error={row['error']}", flush=True)
        if row["error"]:
            print("Stopping on API error; rerun when ready. Existing output saved.")
            break
    print("Saved:", args.out)
    print("NOTE: pattern checks are approximate; inspect the answers for genuine abstention or explicit conflict with BOTH sources.")


if __name__ == "__main__":
    main()
