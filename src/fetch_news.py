from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from ai_engine import enrich_articles
from learning import feedback_boost

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "sources.json"
OUTPUT_FILE = ROOT / "data" / "news.json"
MAX_ARTICLES_PER_SOURCE = 30
MAX_ARTICLES_PER_CATEGORY = 50
MAX_AI_CANDIDATES = 50
REQUEST_TIMEOUT = 20
USER_AGENT = "NewsBriefAI/1.0 (+personal RSS reader)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IMPORTANT_KEYWORDS = {"government": 5, "prime minister": 5, "chief minister": 5, "supreme court": 5, "high court": 4, "election": 5, "budget": 5, "earthquake": 5, "cyclone": 5, "flood": 5, "warning": 5, "alert": 4, "disaster": 5, "war": 5, "attack": 5, "missile": 5, "policy": 3, "law": 4, "economy": 3, "market": 2, "scientific breakthrough": 4, "research": 2, "artificial intelligence": 3, "cybersecurity": 3, "security vulnerability": 5, "data breach": 5, "open source": 2, "kerala": 2, "india": 2}


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = html.unescape(value)
    value = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", value).strip()


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    blocked = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid", "ref"}
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in blocked]
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), "")).rstrip("/")


def normalize_title(title: str) -> str:
    title = clean_text(title).lower()
    title = re.sub(r"[^a-z0-9\u0D00-\u0D7F\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def make_id(title: str, url: str) -> str:
    return hashlib.sha256(f"{normalize_title(title)}|{normalize_url(url)}".encode("utf-8")).hexdigest()[:16]


def parse_date(entry: Any) -> str:
    for field in ("published", "updated", "created"):
        value = entry.get(field)
        if value:
            try:
                parsed = date_parser.parse(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError, OverflowError):
                pass
    return datetime.now(timezone.utc).isoformat()


def calculate_importance(title: str, summary: str, category: str) -> int:
    text = f"{title} {summary}".lower()
    score = 1
    for keyword, points in IMPORTANT_KEYWORDS.items():
        if keyword in text:
            score += points
    if category == "tech" and any(k in text for k in ("ai", "android", "linux", "python", "java", "github", "cybersecurity", "open source", "cloud", "chip")):
        score += 1
    return min(score, 10)


def fetch_source(source: dict[str, str]) -> list[dict[str, Any]]:
    name, url = source["name"], source["url"]
    logger.info("Fetching %s", name)
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed %s: %s", name, exc)
        return []
    feed = feedparser.parse(response.content)
    articles = []
    for entry in feed.entries[:MAX_ARTICLES_PER_SOURCE]:
        title = clean_text(entry.get("title", ""))
        link = normalize_url(entry.get("link", ""))
        summary = clean_text(entry.get("summary", entry.get("description", "")))
        if not title or not link:
            continue
        source_name = name
        source_data = entry.get("source")
        if isinstance(source_data, dict):
            source_name = clean_text(source_data.get("title")) or name
        articles.append({"id": make_id(title, link), "title": title, "summary": summary[:1500], "url": link, "source": source_name, "feed": name, "published_at": parse_date(entry)})
    logger.info("%s -> %d articles", name, len(articles))
    return articles


def deduplicate(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    result = []
    for article in articles:
        title = normalize_title(article["title"])
        if article["id"] in seen_ids or title in seen_titles:
            continue
        seen_ids.add(article["id"])
        seen_titles.add(title)
        result.append(article)
    return result


def process_category(sources: list[dict[str, str]], category: str) -> list[dict[str, Any]]:
    articles = []
    for source in sources:
        for article in fetch_source(source):
            article["category"] = category
            article["importance"] = calculate_importance(article["title"], article["summary"], category)
            articles.append(article)
    candidates = deduplicate(articles)
    candidates.sort(key=lambda item: (item["importance"], item["published_at"]), reverse=True)
    candidates = enrich_articles(candidates[:MAX_AI_CANDIDATES])

    for article in candidates:
        article["category"] = article["ai_category"]
        article["feedback_boost"] = round(feedback_boost(article), 3)
        article["importance"] = article["ai_importance"]
        article["personal_score"] = round(article["ai_importance"] + article["feedback_boost"], 3)

    candidates.sort(key=lambda item: (item["personal_score"], item.get("ai_novelty", 0.0), item["published_at"]), reverse=True)
    return candidates[:MAX_ARTICLES_PER_CATEGORY]


def main() -> None:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = json.load(file)
    output = {
        "version": 3,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ai_engine": {"enabled": True, "type": "mathematical_ml", "method": "tfidf_cosine_prototype", "features": "1-2grams"},
        "learning": {"enabled": True, "feedback_file": "data/feedback.json"},
        "general": process_category(config.get("general", []), "general"),
        "tech": process_category(config.get("tech", []), "tech"),
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT_FILE.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temp.replace(OUTPUT_FILE)
    logger.info("Generated %s | general=%d | tech=%d", OUTPUT_FILE, len(output["general"]), len(output["tech"]))


if __name__ == "__main__":
    main()
