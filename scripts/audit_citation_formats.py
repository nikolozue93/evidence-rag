"""Audit already-saved answer & edge-case CSVs WITHOUT making new API calls.

IMPORTANT: The old answer CSV saves source IDs as an unordered list; it does
NOT preserve citation-number-to-source-ID mapping. Gold-source citation recall
cannot be repaired accurately from that CSV alone. We report only numerical
source reference validity and do not overwrite the original evaluation CSVs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.citations import numeric_citation_check
from scripts.evaluate_edge_cases import CASES, classify


def audit_answers(path: Path) -> dict:
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    success = [r for r in rows if not r.get('error')]
    formats = Counter()
    valid, with_refs = 0, 0
    failures = []
    for r in success:
        answer = r.get('answer') or ''
        if '【' in answer:
            formats['unicode_fullwidth'] += 1
        elif '[' in answer:
            formats['ascii_square_brackets'] += 1
        else:
            formats['no_obvious_citation'] += 1
        retrieved_ids = json.loads(r.get('retrieved_sources') or '[]')
        check = numeric_citation_check(answer, len(retrieved_ids))
        valid += int(check['valid'])
        with_refs += int(bool(check['used']))
        if not check['valid']:
            failures.append({'question_id': r['question_id'], 'used_ids': check['used'],
                             'invalid_ids': check['invalid']})
    return {
        'successful_questions': len(success), 'errors': len(rows) - len(success),
        'citation_formats': dict(formats),
        'answers_with_numeric_source_references': with_refs,
        'answers_with_valid_numbered_source_references': valid,
        'numbered_reference_validity_rate': round(valid / len(success), 4) if success else None,
        'invalid_or_missing_reference_questions': failures,
        'caveat': 'Numerical source-reference validity only; does not verify source factual support. '
                  'The old CSV lacks a numeric-reference-to-source-ID map, so gold-source '
                  'citation coverage metrics cannot be retrospectively repaired.',
    }


def audit_edges(path: Path) -> dict:
    fixtures = {case['id']: case for case in CASES}
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        fixture = fixtures.get(r['case_id'])
        if fixture is None or r.get('error'):
            continue
        check = numeric_citation_check(r['answer'], len(fixture['sources']))
        detected = classify(r['kind'], r['answer'], len(check['used']))
        out.append({'case_id': r['case_id'], 'kind': r['kind'],
                    'expected_behavior_detected': int(detected),
                    'numeric_citations_valid': int(check['valid']),
                    'citation_ids': check['used']})
    cats = sorted({r['kind'] for r in out})
    return {'cases': out,
            'by_kind': {kind: {'passed': sum(r['expected_behavior_detected'] for r in out if r['kind'] == kind),
                               'total': sum(r['kind'] == kind for r in out)} for kind in cats},
            'caveat': 'Synthetic evidence supplied directly; heuristic checks, not end-to-end Qdrant testing or full claim entailment.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', type=Path, default=Path('reports/answer_eval.csv'))
    parser.add_argument('--edges', type=Path, default=Path('reports/edge_case_eval_after_fix.csv'))
    parser.add_argument('--out', type=Path, default=Path('reports/citation_format_audit.json'))
    args = parser.parse_args()
    report = {'answers': audit_answers(args.answers), 'edge_cases': audit_edges(args.edges)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Saved:', args.out)
    print('Original evaluation CSVs were NOT modified; no API calls were made.')


if __name__ == '__main__':
    main()
