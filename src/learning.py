from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_FILE = ROOT / "data" / "feedback.json"


def load_feedback() -> list[dict]:
    if not FEEDBACK_FILE.exists():
        return []
    try:
        data = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def record_feedback(article: dict, label: str) -> None:
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


def feedback_boost(article: dict) -> float:
    """Return a small source/category preference signal from saved feedback."""
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
