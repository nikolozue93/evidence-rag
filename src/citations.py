"""Normalize citation typography returned by LLMs to the project's [N] convention.

This checks REFERENCE SYNTAX only. Valid reference IDs are not proof that a
source entails the claim, and model-produced line numbers are not independently
verified against line-numbered source text.
"""
from __future__ import annotations

import re

# Examples: [2], 【2】, 【2†L1-L3】, 【2†text】.
# Never infer or invent citation numbers; retain the numbers emitted by the LLM.
_CITATION = re.compile(r"\[(?P<ascii>\d{1,3})\]|【(?P<wide>\d{1,3})(?:†[^】]*)?】")


def normalize_citations(answer: str) -> str:
    """Convert supported citation formats to [N]; discard unsupported model-generated location labels.

    Indexed passages are not line-numbered, so the LLM's pseudo-line labels and
    informal tags (e.g. †text) must not be presented as verified locations. Preserve unknown syntax untouched.
    """
    return _CITATION.sub(lambda m: f"[{int(m.group('ascii') or m.group('wide'))}]", answer)


def cited_indices(answer: str) -> set[int]:
    """Get all explicitly referenced numeric source IDs (regardless of format)."""
    return {int(m.group('ascii') or m.group('wide')) for m in _CITATION.finditer(answer)}


def numeric_citation_check(answer: str, source_count: int) -> dict:
    """Check numeric ID references against the 1..source_count source numbering.

    This is *not* factual citation accuracy or source-attribution correctness.
    """
    ids = cited_indices(answer)
    accepted = set(range(1, source_count + 1))
    return {"used": sorted(ids), "invalid": sorted(ids - accepted),
            "valid": bool(ids) and ids <= accepted}
