from __future__ import annotations

"""Lightweight mathematical AI for NewsBrief AI.

No LLM/API is required. The engine uses TF-IDF vectors, cosine similarity,
prototype classification, novelty detection, and a weighted importance model.
It is intentionally small enough to run inside GitHub Actions on CPU.
"""

import math
import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u0D00-\u0D7F]+", re.UNICODE)

TECH_TERMS = {
    "ai", "artificial", "intelligence", "machine", "learning", "model",
    "software", "developer", "programming", "python", "java", "android",
    "linux", "github", "opensource", "open", "source", "cybersecurity",
    "cyber", "security", "vulnerability", "malware", "cloud", "chip",
    "processor", "semiconductor", "robot", "robotics", "database", "api",
    "quantum", "space", "satellite", "technology", "tech",
}

GENERAL_TERMS = {
    "kerala", "india", "government", "minister", "election", "court", "law",
    "budget", "economy", "market", "flood", "cyclone", "earthquake", "weather",
    "rain", "warning", "disaster", "war", "attack", "missile", "health",
    "education", "school", "college", "railway", "road", "police", "crime",
}

IMPORTANT_TERMS = {
    "breaking": 2.5, "urgent": 3.0, "warning": 3.0, "alert": 2.5,
    "earthquake": 4.0, "cyclone": 4.0, "flood": 4.0, "disaster": 4.0,
    "war": 4.0, "attack": 3.5, "missile": 3.5, "election": 3.5,
    "government": 2.0, "policy": 2.0, "supreme": 3.0, "court": 2.0,
    "law": 2.0, "budget": 2.5, "breach": 3.0, "vulnerability": 3.0,
    "security": 1.5, "artificial": 1.0, "intelligence": 1.0, "research": 1.5,
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


def tfidf_vectors(documents: list[str]) -> list[dict[str, float]]:
    tokenized = [tokenize(doc) for doc in documents]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    n = max(len(documents), 1)
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        vector = {}
        for term, count in counts.items():
            idf = math.log((1 + n) / (1 + document_frequency[term])) + 1.0
            vector[term] = (count / total) * idf
        vectors.append(vector)
    return vectors


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _prototype_vectors() -> tuple[dict[str, float], dict[str, float]]:
    general_text = " ".join(sorted(GENERAL_TERMS)) * 3
    tech_text = " ".join(sorted(TECH_TERMS)) * 3
    general, tech = tfidf_vectors([general_text, tech_text])
    return general, tech


GENERAL_PROTO, TECH_PROTO = _prototype_vectors()


def classify(text: str, current_category: str) -> tuple[str, float]:
    vector = tfidf_vectors([text])[0]
    general_score = cosine_similarity(vector, GENERAL_PROTO)
    tech_score = cosine_similarity(vector, TECH_PROTO)

    # Category from the feed remains a useful prior, but the mathematical
    # classifier can override it when the article strongly matches another topic.
    if current_category == "tech":
        tech_score += 0.08
    else:
        general_score += 0.08

    total = general_score + tech_score
    confidence = abs(tech_score - general_score) / total if total else 0.0
    return ("tech" if tech_score > general_score else "general", round(min(0.99, confidence), 4))


def novelty_score(article_text: str, other_texts: list[str]) -> float:
    if not other_texts:
        return 1.0
    vectors = tfidf_vectors([article_text, *other_texts])
    max_similarity = max(cosine_similarity(vectors[0], vector) for vector in vectors[1:])
    return round(max(0.0, 1.0 - max_similarity), 4)


def importance_score(title: str, summary: str, source: str, published_at: str, category: str) -> int:
    text = f"{title} {summary}".lower()
    score = 1.0

    for term, weight in IMPORTANT_TERMS.items():
        if term in text:
            score += weight

    # Source/feed quality is deliberately modest; it must not dominate the story.
    trusted_fragments = ("bbc", "reuters", "the hindu", "indian express", "ptI", "manorama", "mathrubhumi")
    if any(fragment.lower() in source.lower() for fragment in trusted_fragments):
        score += 1.0

    if category == "tech" and any(term in text for term in TECH_TERMS):
        score += 1.0

    # Recency is calculated from ISO timestamps without external dependencies.
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600)
        score += max(0.0, 2.0 - min(age_hours, 48.0) / 24.0)
    except (ValueError, TypeError):
        pass

    return max(1, min(10, round(score)))


def enrich_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply mathematical AI fields to already-fetched articles."""
    texts = [f"{a['title']} {a.get('summary', '')}" for a in articles]
    for index, article in enumerate(articles):
        text = texts[index]
        predicted_category, confidence = classify(text, article.get("category", "general"))
        article["ai_category"] = predicted_category
        article["ai_confidence"] = confidence
        article["ai_importance"] = importance_score(
            article["title"], article.get("summary", ""), article.get("source", ""),
            article.get("published_at", ""), predicted_category,
        )
        article["ai_novelty"] = novelty_score(text, texts[:index] + texts[index + 1:])
        article["ai"] = {
            "method": "tfidf_cosine_prototype",
            "version": 1,
        }
    return articles
