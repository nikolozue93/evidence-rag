"""Run: pytest tests/test_quality_eval.py -q (no cloud access)."""
from scripts.evaluate_answers import gold_in_answer, score_result, summarize, token_f1
from scripts.evaluate_edge_cases import CASES, make_search, classify
from scripts.compare_chunking import calculate


def test_answer_literal_match_and_f1():
    assert gold_in_answer("Arthur's Magazine", "Arthur's Magazine started in 1844 [2].") == 1
    assert gold_in_answer("Arthur's Magazine", "First for Women came first.") == 0
    assert gold_in_answer("no", "No, there is no record.") == 1
    assert gold_in_answer("yes", "yesterday was cold") == 0
    assert token_f1("Arthur's Magazine", "Arthur's Magazine [1]") == 1


def test_citations_gold_overlap_are_not_claim_level():
    item = {"question_id": "x", "question": "Q", "answer": "Arthur's Magazine",
            "supporting_source_ids": ["a", "b"]}
    result = {"answer": "Arthur's Magazine [1] and First for Women [2]",
              "sources": [{"id": 1, "source_id": "a"}, {"id": 2, "source_id": "other"}],
              "status": "cited", "search_queries": ["Arthur", "First"]}
    row = score_result(item, result)
    assert row["gold_in_answer"] == 1
    assert row["citations_valid"] == 1
    assert row["cited_gold_fraction"] == 0.5
    assert row["all_gold_retrieved"] == 0
    assert row["all_gold_cited"] == 0
    assert summarize({"x": row})["successes"] == 1


def test_invalid_citation_is_detected():
    item = {"question_id": "y", "question": "Who?", "answer": "One", "supporting_source_ids": ["a"]}
    result = {"answer": "One [99]", "sources": [{"id": 1, "source_id": "a"}], "status": "uncited_or_invalid_citations"}
    assert score_result(item, result)["citations_valid"] == 0


def test_fixture_search_exposes_both_sides():
    fixture = next(c for c in CASES if c["kind"] == "contradictory")
    found = make_search(fixture)("arbitrary query")
    assert len(found) == 2
    assert found[0]["payload"]["source_id"] != found[1]["payload"]["source_id"]
    assert classify("contradictory", "Conflicting evidence [1] [2]", 2)
    assert not classify("contradictory", "Both sources agree [1]", 1)
    assert classify("unanswerable", "Insufficient evidence: unknown", 0)


def test_structural_chunking_comparison():
    data = [" ".join("word" + str(i) for i in range(290)),
            "First paragraph of sample evidence.\n\nSecond paragraph with separate evidence."]
    stats = calculate(data, words=40)
    assert len(stats) == 2
    assert stats[0]["strategy"] == "fixed"
    assert stats[1]["strategy"] == "paragraph"
    assert stats[0]["redundancy_ratio"] >= 1
    assert stats[1]["redundancy_ratio"] == 1
