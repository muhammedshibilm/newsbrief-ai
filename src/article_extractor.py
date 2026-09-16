from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

USER_AGENT = "NewsBriefAI/1.0 (+personal RSS reader)"
REQUEST_TIMEOUT = 12
MAX_ARTICLE_CHARS = 30000

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _find_article_body(value: Any) -> str | None:
    if isinstance(value, dict):
        article_body = value.get("articleBody")
        if isinstance(article_body, str) and len(article_body.strip()) >= 200:
            return article_body
        for child in value.values():
            result = _find_article_body(child)
            if result:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_article_body(child)
            if result:
                return result
    return None


def _json_ld_article_body(soup: BeautifulSoup) -> str | None:
    for script in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        raw = script.string or script.get_text()
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        result = _find_article_body(data)
        if result:
            return result
    return None


def _paragraph_text(container: Any) -> str:
    paragraphs = []
    for paragraph in container.find_all("p"):
        text = _clean_text(paragraph.get_text(" ", strip=True))
        if len(text) >= 35:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _clean_page(soup: BeautifulSoup) -> None:
    for tag in soup.find_all([
        "script", "style", "noscript", "nav", "header", "footer", "aside",
        "form", "button", "svg", "iframe", "figure", "figcaption",
    ]):
        tag.decompose()

    noise = re.compile(
        r"advert|ads|banner|social|share|comment|related|recommended|newsletter|subscribe|cookie|popup|modal|breadcrumb|navigation|menu",
        re.I,
    )
    for tag in soup.find_all(attrs={"class": noise}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": noise}):
        tag.decompose()


def extract_article_content(url: str) -> dict[str, Any]:
    """Fetch and extract the main readable article body from a publisher page."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()
        if "html" not in response.headers.get("Content-Type", "").lower():
            return {"content": "", "available": False, "length": 0, "final_url": response.url}
    except requests.RequestException as exc:
        logger.warning("Article extraction failed for %s: %s", url, exc)
        return {"content": "", "available": False, "length": 0, "final_url": url}

    soup = BeautifulSoup(response.text, "html.parser")

    # JSON-LD is often the cleanest representation of the publisher's article body.
    json_ld_body = _json_ld_article_body(soup)
    if json_ld_body:
        content = _clean_text(BeautifulSoup(json_ld_body, "html.parser").get_text(" ", strip=True))
        content = content[:MAX_ARTICLE_CHARS]
        return {
            "content": content,
            "available": len(content) >= 200,
            "length": len(content),
            "final_url": response.url,
        }

    _clean_page(soup)

    article = soup.find("article")
    candidates = []
    if article:
        candidates.append(article)
    main = soup.find("main")
    if main and main is not article:
        candidates.append(main)

    candidates.extend(
        soup.find_all(
            attrs={
                "class": re.compile(
                    r"article[-_ ]?(body|content|text)?|story[-_ ]?(body|content)?|post[-_ ]?(body|content)?|entry[-_ ]?content|articleBody",
                    re.I,
                )
            }
        )
    )

    best = ""
    for candidate in candidates:
        text = _paragraph_text(candidate)
        if len(text) > len(best):
            best = text

    if len(best) < 200:
        best = _paragraph_text(soup)

    best = _clean_text(best)[:MAX_ARTICLE_CHARS]
    return {
        "content": best,
        "available": len(best) >= 200,
        "length": len(best),
        "final_url": response.url,
    }
