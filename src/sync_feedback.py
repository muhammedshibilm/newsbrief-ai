from __future__ import annotations

import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_FILE = ROOT / "data" / "feedback.json"
COLLECTION = "feedback"


def main() -> None:
    project_id = os.environ.get("FIREBASE_PROJECT_ID", "").strip()
    access_token = os.environ.get("FIRESTORE_ACCESS_TOKEN", "").strip()
    if not project_id or not access_token:
        raise RuntimeError("Firebase project credentials are not configured")

    url = (
        f"https://firestore.googleapis.com/v1/projects/{project_id}"
        f"/databases/(default)/documents/{COLLECTION}"
    )
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"pageSize": 1000},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    rows = []
    for document in payload.get("documents", []):
        fields = document.get("fields", {})
        action = fields.get("action", {}).get("stringValue", "").strip().lower()
        if action not in {"like", "skip"}:
            continue

        title = fields.get("title", {}).get("stringValue", "").strip()
        summary = fields.get("summary", {}).get("stringValue", "").strip()
        text = fields.get("text", {}).get("stringValue", "").strip()
        if not text:
            text = f"{title} {summary}".strip()

        rows.append(
            {
                "article_id": fields.get("article_id", {}).get("stringValue", "").strip(),
                "url": fields.get("url", {}).get("stringValue", "").strip(),
                "text": text,
                "category": fields.get("category", {}).get("stringValue", "general").strip() or "general",
                "source": fields.get("source", {}).get("stringValue", "").strip(),
                "label": action,
                "created_at": (
                    fields.get("timestamp", {}).get("timestampValue")
                    or fields.get("created_at", {}).get("timestampValue")
                    or fields.get("created_at", {}).get("stringValue", "")
                ),
            }
        )

    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_FILE.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Synced {len(rows)} valid feedback records from Firestore")


if __name__ == "__main__":
    main()
