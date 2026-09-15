from __future__ import annotations

"""Lightweight mathematical NLP engine for NewsBrief AI."""

import math
import re
from datetime import datetime, timezone
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u0D00-\u0D7F]+", re.UNICODE)
TECH_TERMS = {"ai", "artificial intelligence", "machine learning", "software", "developer", "programming", "python", "java", "android", "linux", "github", "open source", "cybersecurity", "malware", "cloud", "chip", "processor", "semiconductor", "robotics", "database", "api", "quantum", "satellite", "technology"}
GENERAL_TERMS = {"kerala", "india", "government", "minister", "election", "court", "law", "budget", "economy", "market", "flood", "cyclone", "earthquake", "weather", "rain", "warning", "disaster", "war", "attack", "missile", "health", "education", "school", "college", "railway", "road", "police", "crime"}
IMPORTANT_TERMS = {"breaking": 2.5, "urgent": 3.0, "warning": 3.0, "alert": 2.5, "earthquake": 4.0, "cyclone": 4.0, "flood": 4.0, "disaster": 4.0, "war": 4.0, "attack": 3.5, "missile": 3.5, "election": 3.5, "government": 2.0, "policy": 2.0, "supreme": 3.0, "court": 2.0, "law": 2.0, "budget": 2.5, "breach": 3.0, "vulnerability": 3.0, "security": 1.5, "artificial intelligence": 2.0, "research": 1.5}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)


def cosine_similarity(a: Any, b: Any) -> float:
    denominator = math.sqrt(float(a.multiply(a).sum())) * math.sqrt(float(b.multiply(b).sum()))
    if denominator == 0:
        return 0.0
    return float(a.multiply(b).sum()) / denominator


def classify(text: str, current_category: str) -> tuple[str, float]:
    matrix = _vectorizer().fit_transform([" ".join(sorted(GENERAL_TERMS)), " ".join(sorted(TECH_TERMS)), text])
    general_score = cosine_similarity(matrix[2], matrix[0])
    tech_score = cosine_similarity(matrix[2], matrix[1])
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
    matrix = _vectorizer().fit_transform([article_text, *other_texts])
    max_similarity = max(cosine_similarity(matrix[0], matrix[i]) for i in range(1, matrix.shape[0]))
    return round(max(0.0, 1.0 - max_similarity), 4)


def importance_score(title: str, summary: str, source: str, published_at: str, category: str) -> int:
    text = f"{title} {summary}".lower()
    score = 1.0
    for term, weight in IMPORTANT_TERMS.items():
        if term in text:
            score += weight
    trusted = ("bbc", "reuters", "the hindu", "indian express", "pti", "manorama", "mathrubhumi")
    if any(fragment in source.lower() for fragment in trusted):
        score += 1.0
    if category == "tech" and any(term in text for term in TECH_TERMS):
        score += 1.0
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600)
        score += max(0.0, 2.0 - min(age_hours, 48.0) / 24.0)
    except (ValueError, TypeError):
        pass
    return max(1, min(10, round(score)))


def enrich_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = [f"{a['title']} {a.get('summary', '')}" for a in articles]
    for index, article in enumerate(articles):
        predicted_category, confidence = classify(texts[index], article.get("category", "general"))
        article["ai_category"] = predicted_category
        article["ai_confidence"] = confidence
        article["ai_importance"] = importance_score(article["title"], article.get("summary", ""), article.get("source", ""), article.get("published_at", ""), predicted_category)
        article["ai_novelty"] = novelty_score(texts[index], texts[:index] + texts[index + 1:])
        article["ai"] = {"method": "tfidf_cosine_prototype", "features": "1-2grams", "version": 2}
    return articles
