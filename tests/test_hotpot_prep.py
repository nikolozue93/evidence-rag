"""Offline toy test for deduplication and ground-truth mapping."""
import json
import tempfile
from pathlib import Path
from scripts.prepare_hotpot import build_corpus


def fake(id_, titles, texts, question="Which came first?"):
    return {"id": id_, "context": {"title": titles, "sentences": [[t] for t in texts]},
            "question": question, "answer": titles[0], "type": "comparison", "level": "easy",
            "supporting_facts": {"title": titles, "sent_id": [0] * len(titles)}}


def test_build_corpus():
    a = "Arthur's Magazine was published in the nineteenth century, starting in 1844 in Philadelphia."
    b = "First for Women magazine started in 1989 and was published by Bauer Media in the United States."
    c = "Another independent source passage describing unrelated contextual information in adequate detail."
    train = [fake("train1", ["Arthur's Magazine", "First for Women"], [a, b]),
             fake("train2", ["Repeated Arthur", "Other"], [a, c])]
    val = [fake("val1", ["Arthur's Magazine", "First for Women"], [a, b])]
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        summary = build_corpus(train, val, records_path=root / "records.jsonl",
                               eval_path=root / "eval.jsonl", target_records=3,
                               eval_questions=1, progress_every=1)
        assert summary["unique_passage_records"] == 3
        assert summary["validation_questions_saved"] == 1
        rows = [json.loads(s) for s in (root / "records.jsonl").read_text().splitlines()]
        assert len({r["source_id"] for r in rows}) == 3
        qa = json.loads((root / "eval.jsonl").read_text().strip())
        assert len(qa["supporting_source_ids"]) == 2


if __name__ == "__main__":
    test_build_corpus()
    print("Hotpot preparation test: PASSED")
