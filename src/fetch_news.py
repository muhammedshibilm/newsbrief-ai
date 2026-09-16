from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from ai_engine import enrich_articles
from entity_extractor import extract_entities
from learning import adaptive_source_reliability, feedback_boost, personal_score, train_personal_classifier
from semantic_dedup import semantic_duplicate_flags

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "sources.json"
OUTPUT_FILE = ROOT / "data" / "news.json"
MAX_ARTICLES_PER_SOURCE = 30
MAX_ARTICLES_PER_CATEGORY = 50
MAX_AI_CANDIDATES = 50
REQUEST_TIMEOUT = 20
IMAGE_REQUEST_TIMEOUT = 10
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


def _valid_image_url(url: str | None, base_url: str = "") -> str | None:
    if not url:
        return None
    candidate = urljoin(base_url, html.unescape(str(url)).strip())
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if candidate.startswith("data:"):
        return None
    return candidate


def _first_url(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("url", "href", "src"):
            if value.get(key):
                return str(value[key])
    return None


def extract_rss_image(entry: Any) -> str | None:
    """Extract an image URL from common RSS media/enclosure fields."""
    for field in ("media_content", "media_thumbnail", "media:content", "media:thumbnail"):
        values = entry.get(field)
        if not isinstance(values, list):
            values = [values] if values else []
        for item in values:
            if isinstance(item, dict):
                media_type = str(item.get("type", "")).lower()
                candidate = _first_url(item)
                if candidate and (not media_type or media_type.startswith("image/")):
                    image_url = _valid_image_url(candidate)
                    if image_url:
                        return image_url

    enclosures = entry.get("enclosures")
    if isinstance(enclosures, list):
        for enclosure in enclosures:
            if not isinstance(enclosure, dict):
                continue
            media_type = str(enclosure.get("type", "")).lower()
            candidate = _first_url(enclosure)
            if candidate and (media_type.startswith("image/") or not media_type):
                image_url = _valid_image_url(candidate)
                if image_url:
                    return image_url

    image_data = entry.get("image")
    candidate = _first_url(image_data)
    return _valid_image_url(candidate)


def extract_image_from_html(page_html: str, article_url: str) -> str | None:
    """Extract the publisher's main image from article HTML."""
    soup = BeautifulSoup(page_html, "html.parser")

    meta_candidates = [
        soup.find("meta", attrs={"property": "og:image"}),
        soup.find("meta", attrs={"property": "og:image:url"}),
        soup.find("meta", attrs={"name": "twitter:image"}),
        soup.find("meta", attrs={"name": "twitter:image:src"}),
    ]
    for tag in meta_candidates:
        if tag and tag.get("content"):
            image_url = _valid_image_url(tag.get("content"), article_url)
            if image_url:
                return image_url

    article = soup.find("article")
    containers = [article] if article else []
    containers.extend(soup.find_all(attrs={"class": re.compile(r"article|story|content|post", re.I)}))

    seen: set[int] = set()
    for container in containers:
        if container is None or id(container) in seen:
            continue
        seen.add(id(container))
        for image in container.find_all("img"):
            for attr in ("src", "data-src", "data-lazy-src", "data-original"):
                candidate = image.get(attr)
                image_url = _valid_image_url(candidate, article_url)
                if image_url:
                    return image_url
            srcset = image.get("srcset") or image.get("data-srcset")
            if srcset:
                candidate = str(srcset).split(",")[-1].strip().split(" ")[0]
                image_url = _valid_image_url(candidate, article_url)
                if image_url:
                    return image_url

    return None


def fetch_webpage_image(article_url: str) -> str | None:
    """Fetch an article page and extract its primary image without failing the pipeline."""
    try:
        response = requests.get(
            article_url,
            headers={"User-Agent": USER_AGENT},
            timeout=IMAGE_REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type:
            return None
        return extract_image_from_html(response.text, response.url)
    except requests.RequestException as exc:
        logger.debug("Image extraction failed for %s: %s", article_url, exc)
        return None
    except (ValueError, UnicodeError) as exc:
        logger.debug("Invalid article page for image extraction %s: %s", article_url, exc)
        return None


def add_article_images(articles: list[dict[str, Any]]) -> None:
    """Fill missing image URLs using the actual article page."""
    image_cache: dict[str, str | None] = {}
    extracted = 0
    missing = 0

    for article in articles:
        if article.get("image_url"):
            extracted += 1
            continue

        article_url = article.get("url", "")
        if not article_url:
            article["image_url"] = None
            missing += 1
            continue

        if article_url not in image_cache:
            image_cache[article_url] = fetch_webpage_image(article_url)

        article["image_url"] = image_cache[article_url]
        if article["image_url"]:
            extracted += 1
        else:
            missing += 1

    logger.info("Article images -> extracted=%d missing=%d", extracted, missing)


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
        articles.append({
            "id": make_id(title, link),
            "title": title,
            "summary": summary[:1500],
            "url": link,
            "image_url": extract_rss_image(entry),
            "source": source_name,
            "feed": name,
            "published_at": parse_date(entry),
        })
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


def process_category(sources: list[dict[str, str]], category: str, classifier: tuple[Any, Any], reliability: dict[str, float]) -> list[dict[str, Any]]:
    articles = []
    for source in sources:
        for article in fetch_source(source):
            article["category"] = category
            article["importance"] = calculate_importance(article["title"], article["summary"], category)
            articles.append(article)

    candidates = deduplicate(articles)
    candidates.sort(key=lambda item: (item["importance"], item["published_at"]), reverse=True)
    candidates = candidates[:MAX_AI_CANDIDATES]

    semantic_removed = semantic_duplicate_flags(candidates)
    if semantic_removed:
        candidates = [a for a in candidates if a["id"] not in semantic_removed]
        logger.info("Removed %d semantic duplicates", len(semantic_removed))

    candidates = enrich_articles(candidates)
    vectorizer, model = classifier
    for article in candidates:
        article["category"] = article["ai_category"]
        source_key = str(article.get("source", "")).lower()
        article["entities"] = extract_entities(f"{article['title']} {article.get('summary', '')}")
        article["source_reliability"] = reliability.get(source_key, 1.0)
        article["feedback_boost"] = round(feedback_boost(article), 3)
        article["personal_model_score"] = personal_score(article, vectorizer, model)
        article["importance"] = article["ai_importance"]
        article["personal_score"] = round(article["ai_importance"] * article["source_reliability"] + article["feedback_boost"] + article["personal_model_score"], 3)

    candidates.sort(key=lambda item: (item["personal_score"], item.get("ai_novelty", 0.0), item["published_at"]), reverse=True)
    candidates = candidates[:MAX_ARTICLES_PER_CATEGORY]
    add_article_images(candidates)
    return candidates


def main() -> None:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = json.load(file)

    classifier = train_personal_classifier()
    vectorizer, model, training = classifier
    reliability = adaptive_source_reliability()
    output = {
        "version": 5,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "ai_engine": {"enabled": True, "type": "mathematical_ml", "method": "tfidf_cosine_prototype", "features": "1-2grams"},
        "learning": {"enabled": True, "classifier": training, "adaptive_source_reliability": True},
        "semantic_deduplication": {"enabled": True, "method": "sentence-transformers_optional", "threshold": 0.88},
        "image_extraction": {"enabled": True, "sources": ["rss_media", "og:image", "twitter:image", "article_img"], "fallback": "null"},
        "general": process_category(config.get("general", []), "general", (vectorizer, model), reliability),
        "tech": process_category(config.get("tech", []), "tech", (vectorizer, model), reliability),
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
