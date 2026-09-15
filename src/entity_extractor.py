from __future__ import annotations

import re

MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]{3,}")
ENGLISH_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&.-]*)(?:\s+[A-Z][A-Za-z0-9&.-]*){0,3}\b")
STOPWORDS = {"The", "This", "That", "India", "Kerala", "World", "News", "Latest"}


def extract_entities(text: str, limit: int = 12) -> list[str]:
    """Extract lightweight named-entity candidates without a large NLP model."""
    candidates = MALAYALAM_RE.findall(text or "") + ENGLISH_ENTITY_RE.findall(text or "")
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        value = " ".join(item.split()).strip(" .,;:()[]{}")
        if len(value) < 3 or value in STOPWORDS or value.lower() in {x.lower() for x in STOPWORDS}:
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
        if len(result) >= limit:
            break
    return result
