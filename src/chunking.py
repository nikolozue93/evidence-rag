"""Two transparent, reproducible word-level chunking strategies.

Fixed: overlapping word windows. Paragraph: greedy paragraph grouping, splitting
long paragraphs into sentences/word windows where necessary. No LLM required.
"""
from __future__ import annotations
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fixed_chunks(text: str, *, words: int = 180, overlap: int = 30) -> list[str]:
    if words < 1 or overlap < 0 or overlap >= words:
        raise ValueError("Require words > 0 and 0 <= overlap < words")
    tokens = normalize(text).split()
    if not tokens:
        return []
    step = words - overlap
    result = []
    for start in range(0, len(tokens), step):
        section = tokens[start:start + words]
        if not section:
            break
        result.append(" ".join(section))
        if start + words >= len(tokens):
            break
    return result


def paragraph_chunks(text: str, *, words: int = 180) -> list[str]:
    """Preserve paragraph boundaries where possible; no overlap by design."""
    if words < 1:
        raise ValueError("words must be >= 1")
    paragraphs = [normalize(p) for p in re.split(r"\n\s*\n", text) if normalize(p)]
    result = []
    pending = []
    count = 0

    def flush():
        nonlocal pending, count
        if pending:
            result.append("\n\n".join(pending))
            pending, count = [], 0

    for paragraph in paragraphs:
        tokens = paragraph.split()
        if len(tokens) > words:
            flush()
            for start in range(0, len(tokens), words):
                result.append(" ".join(tokens[start:start + words]))
            continue
        if pending and count + len(tokens) > words:
            flush()
        pending.append(paragraph)
        count += len(tokens)
    flush()
    return result


def chunk_text(text: str, strategy: str = "paragraph", words: int = 180) -> list[str]:
    if strategy == "fixed":
        return fixed_chunks(text, words=words, overlap=min(30, max(0, words // 6)))
    if strategy == "paragraph":
        return paragraph_chunks(text, words=words)
    raise ValueError("strategy must be 'fixed' or 'paragraph'")
