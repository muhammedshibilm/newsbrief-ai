from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_FILE = ROOT / "data" / "feedback.json"
MIN_LABELS = 8


def load_feedback() -> list[dict[str, Any]]:
    if not FEEDBACK_FILE.exists():
        return []
    try:
        data = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def record_feedback(article: dict[str, Any], label: str) -> None:
    if label not in {"like", "skip"}:
        raise ValueError("label must be 'like' or 'skip'")
    items = load_feedback()
    items.append({
        "url": article.get("url", ""),
        "text": f"{article.get('title', '')} {article.get('summary', '')}".strip(),
        "category": article.get("category", "general"),
        "source": article.get("source", ""),
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_personal_classifier() -> tuple[TfidfVectorizer | None, LogisticRegression | None, dict[str, Any]]:
    """Train a personal relevance classifier from saved feedback.

    It activates only when there are enough examples of both like and skip.
    Otherwise the deterministic ranking system remains the fallback.
    """
    rows = [x for x in load_feedback() if x.get("text") and x.get("label") in {"like", "skip"}]
    likes = sum(x["label"] == "like" for x in rows)
    skips = len(rows) - likes
    metadata = {"trained": False, "samples": len(rows), "likes": likes, "skips": skips}
    if len(rows) < MIN_LABELS or likes < 2 or skips < 2:
        return None, None, metadata

    texts = [x["text"] for x in rows]
    labels = [x["label"] for x in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(matrix, labels)
    metadata["trained"] = True
    metadata["features"] = len(vectorizer.vocabulary_)
    return vectorizer, model, metadata


def personal_score(article: dict[str, Any], vectorizer: TfidfVectorizer | None, model: LogisticRegression | None) -> float:
    if vectorizer is None or model is None:
        return 0.0
    text = f"{article.get('title', '')} {article.get('summary', '')}".strip()
    probabilities = model.predict_proba(vectorizer.transform([text]))[0]
    classes = list(model.classes_)
    if "like" not in classes:
        return 0.0
    like_probability = float(probabilities[classes.index("like")])
    return round((like_probability - 0.5) * 3.0, 3)


def adaptive_source_reliability() -> dict[str, float]:
    """Estimate source reliability from user feedback, bounded to avoid domination."""
    stats: dict[str, list[int]] = {}
    for row in load_feedback():
        source = str(row.get("source", "")).strip().lower()
        if not source:
            continue
        stats.setdefault(source, [0, 0])
        stats[source][0] += 1
        if row.get("label") == "like":
            stats[source][1] += 1
    reliability: dict[str, float] = {}
    for source, (count, likes) in stats.items():
        rate = (likes + 2) / (count + 4)  # Bayesian smoothing toward neutral 0.5.
        reliability[source] = round(0.75 + rate * 0.5, 3)
    return reliability


def feedback_boost(article: dict[str, Any]) -> float:
    """Compatibility fallback: source/category preference signal."""
    items = load_feedback()
    if not items:
        return 0.0
    source = str(article.get("source", "")).lower()
    category = article.get("category", "general")
    relevant = [x for x in items if x.get("source", "").lower() == source or x.get("category") == category]
    if not relevant:
        return 0.0
    score = sum(1 if x.get("label") == "like" else -1 for x in relevant)
    return max(-1.5, min(1.5, score * 0.25))
