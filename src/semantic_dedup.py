from __future__ import annotations

from typing import Any


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Return embedding cosine similarity when sentence-transformers is available.

    The dependency is optional so GitHub Actions can keep using the TF-IDF
    fallback when an embedding model is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return 0.0
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(vectors[0] @ vectors[1])


def semantic_duplicate_flags(articles: list[dict[str, Any]], threshold: float = 0.88) -> set[str]:
    """Find later articles that are near-duplicates of an earlier article."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return set()
    if len(articles) < 2:
        return set()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"{a.get('title', '')} {a.get('summary', '')}" for a in articles]
    vectors = model.encode(texts, normalize_embeddings=True)
    duplicates: set[str] = set()
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            if float(vectors[i] @ vectors[j]) >= threshold:
                duplicates.add(articles[j].get("id", ""))
    return {x for x in duplicates if x}
