from __future__ import annotations

import logging
import os
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

SUMMARY_SENTENCES = 5
MIN_ARTICLE_CHARS = 200
MAX_LLM_INPUT_CHARS = 30000


def _sentences(text: str) -> list[str]:
    # Supports Malayalam and English sentence punctuation without requiring a language model.
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text or "")
    return [re.sub(r"\s+", " ", p).strip() for p in parts if len(p.strip()) >= 35]


def _extractive_summary(text: str, max_sentences: int = SUMMARY_SENTENCES) -> str:
    sentences = _sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(sentences)
    scores = matrix.sum(axis=1).A1
    selected = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:max_sentences])
    return " ".join(sentences[i] for i in selected)


def _language_hint(text: str) -> str:
    malayalam = len(re.findall(r"[\u0D00-\u0D7F]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    return "Malayalam" if malayalam > max(20, latin * 0.2) else "English"


def _openai_summary(text: str, title: str, source: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=30.0)
        model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        language = _language_hint(text)
        prompt = f"""Summarize this news article for a personal news briefing.

Rules:
- Read the supplied article content, not just the title.
- Write 3-6 concise sentences covering the main event, important people/organizations, relevant place/time, key facts, and consequences when stated.
- Preserve important numbers, dates, names, and technical terms.
- Do not invent facts or add information not present in the article.
- Do not copy long passages verbatim.
- Keep the summary in the article's language. The detected language is {language}.
- Return only the summary, with no heading or preamble.

Title: {title}
Source: {source}

Article content:
{text[:MAX_LLM_INPUT_CHARS]}
"""
        response = client.responses.create(model=model, input=prompt)
        result = (response.output_text or "").strip()
        if result:
            return result
    except Exception as exc:  # Keep the scheduled news pipeline resilient.
        logger.warning("LLM summary failed for %s: %s", title, exc)
    return None


def summarize_article(content: str, title: str, source: str, rss_summary: str = "") -> dict[str, Any]:
    content = (content or "").strip()
    if len(content) >= MIN_ARTICLE_CHARS:
        llm_summary = _openai_summary(content, title, source)
        if llm_summary:
            return {
                "summary": llm_summary,
                "summary_method": "openai_full_article",
                "article_content_available": True,
                "article_content_length": len(content),
            }

        extractive = _extractive_summary(content)
        if extractive:
            return {
                "summary": extractive,
                "summary_method": "tfidf_extractive_full_article",
                "article_content_available": True,
                "article_content_length": len(content),
            }

    fallback = (rss_summary or title).strip()
    return {
        "summary": fallback,
        "summary_method": "rss_fallback",
        "article_content_available": False,
        "article_content_length": len(content),
    }
