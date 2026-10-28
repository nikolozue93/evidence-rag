from types import SimpleNamespace
import json
from src.investigation import investigate
from scripts.evaluate_retrieval import scores

class FakeGroq:
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        if kwargs.get("tool_choice") == "required":
            query = "Arthur's Magazine" if self.calls == 1 else "First for Women"
            tool_call = SimpleNamespace(
                id=f"call{self.calls}",
                function=SimpleNamespace(name="search_evidence", arguments=json.dumps({"query": query})),
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            message = SimpleNamespace(content="Arthur's Magazine began in 1844 [1]; First for Women began in 1989 [2].", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_search(query):
    if "Arthur" in query:
        p = {"source_id": "a", "title": "Arthur's Magazine", "text": "Arthur's Magazine began in 1844."}
    else:
        p = {"source_id": "b", "title": "First for Women", "text": "First for Women started in 1989."}
    return [{"id": p["source_id"], "payload": p}]


def test_tools_and_citations():
    result = investigate("Which came first?", None, FakeGroq(), search_fn=fake_search)
    assert result["status"] == "cited", result
    assert len(result["search_queries"]) == 2
    assert len(result["sources"]) == 2
    assert result["citation_ids_used"] == [1, 2]


def test_retrieval_scores():
    hits = [{"payload": {"source_id": "wrong"}},
            {"payload": {"source_id": "a"}},
            {"payload": {"source_id": "b"}}]
    m = scores(hits, ["a", "b"])
    assert m["recall"] == 1
    assert m["mrr"] == 0.5
    assert m["all_supporting_hit"] == 1


if __name__ == "__main__":
    test_tools_and_citations()
    test_retrieval_scores()
    print("All investigation tests passed")
