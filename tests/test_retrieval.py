from types import SimpleNamespace as Hit
from src.retrieval import weighted_rrf


def test_weighted_rrf():
    a = Hit(id="1", payload={"title": "Arthur"})
    b = Hit(id="2", payload={"title": "Women"})
    c = Hit(id="3", payload={"title": "Arthur"})
    result = weighted_rrf([a,b,c], [b,a,c], top_k=3)
    assert len(result) == 3
    assert result[0]["id"] in {"1", "2"}
    assert all("score" in x and x["score"] > 0 for x in result)
    assert len({x["id"] for x in result}) == 3


if __name__ == "__main__":
    test_weighted_rrf()
    print("Hybrid ranking test: PASSED")
