"""Pure citation parsing and saved-results audit regression tests."""
import csv
import json
from pathlib import Path

from src.citations import cited_indices, normalize_citations, numeric_citation_check
from scripts.evaluate_answers import score_result
from scripts.evaluate_edge_cases import classify
from scripts.audit_citation_formats import audit_answers, audit_edges


def test_accepts_plain_and_unicode_citations():
    s = 'A [2], B 【10】, C 【3†L1-L3】 and D 【4†L4–L7】, E 【5†text】.'
    assert cited_indices(s) == {2, 10, 3, 4, 5}
    assert normalize_citations(s) == 'A [2], B [10], C [3] and D [4], E [5].'


def test_bad_references_are_not_silently_fixed():
    assert numeric_citation_check('【3】', 2)['invalid'] == [3]
    assert not numeric_citation_check('No references', 2)['valid']
    assert numeric_citation_check('【1†L1-L2】 【2】', 2)['valid']


def test_score_result_recognizes_unicode_citations():
    item = {'question_id': 'abc', 'question': 'Which?', 'answer': 'Magazine A',
            'supporting_source_ids': ['source-A', 'source-B']}
    result = {'answer': 'Magazine A【1†L1-L3】 and Magazine B【2】',
              'sources': [{'id': 1, 'source_id': 'source-A'},
                          {'id': 2, 'source_id': 'source-B'}]}
    scored = score_result(item, result)
    assert scored['citations_valid'] == 1.0
    assert scored['all_gold_cited'] == 1.0
    assert scored['answer'] == 'Magazine A[1] and Magazine B[2]'


def test_all_four_conflict_answers_from_user_csv_are_detected():
    # Pattern checker requires 2 correctly parsed citation numbers.
    answers = [
        'Conflicting evidence: three nodes [1], seven nodes [2].',
        'Different accounts: 12 June 2024【1】 vs 18 June 2024【2】.',
        'The cause is disputed: cache【1】, migration【2】. Conflicting evidence.'
    ]
    assert all(classify('contradictory', a, len(cited_indices(a))) for a in answers)


def test_audit_does_not_require_live_api(tmp_path):
    a = tmp_path/'answer.csv'
    with a.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question_id', 'answer', 'retrieved_sources', 'error'])
        writer.writeheader()
        writer.writerow({'question_id': 'one', 'answer': 'Yes 【1】', 'retrieved_sources': json.dumps(['a']), 'error': ''})
        writer.writerow({'question_id': 'two', 'answer': 'No source', 'retrieved_sources': json.dumps(['b']), 'error': ''})
    r = audit_answers(a)
    assert r['successful_questions'] == 2
    assert r['answers_with_valid_numbered_source_references'] == 1
    assert r['numbered_reference_validity_rate'] == 0.5
