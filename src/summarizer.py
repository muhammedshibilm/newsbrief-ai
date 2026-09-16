from __future__ import annotations

import logging
import os
import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

SUMMARY_SENTENCES = 4
MIN_ARTICLE_CHARS = 200
MAX_LLM_INPUT_CHARS = 24000


def _sentences(text: str) -> list[str]:
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


def _clean_model_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(summary|സംഗ്രഹം)\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _openai_summary(text: str, title: str, source: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=30.0)
        model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        language = _language_hint(text)

        prompt = f"""You are the summarization engine for a personal news reader.

TASK
Read the publisher article body below and produce a precise short news summary.
The article body is the source of truth. The RSS title is only a reference for context.

STRICT RULES
1. Use ONLY facts explicitly supported by the article body.
2. Do not infer motives, causes, outcomes, allegations, identities, or future events that the article does not state.
3. Clearly preserve whether a statement is an allegation, complaint, police instruction, official statement, report, or confirmed event.
4. Preserve names, organizations, places, dates, quantities, money amounts, and important technical terms accurately.
5. Do not turn allegations or claims into established facts.
6. Do not repeat the headline as the summary.
7. Do not mention that you are an AI and do not add a preamble.
8. Do not include advertisements, related-article recommendations, navigation text, author metadata, tags, or disclaimers.
9. Write exactly 3-5 concise sentences.
10. Keep the same language as the article. Detected language: {language}.
11. Return only the summary text.

TITLE
{title}

SOURCE
{source}

PUBLISHER ARTICLE BODY
{text[:MAX_LLM_INPUT_CHARS]}
"""

        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=220,
        )
        result = _clean_model_output(response.output_text)
        if result:
            return result
    except Exception as exc:
        logger.warning("LLM summary failed for %s: %s", title, exc)

    return None


def summarize_article(content: str, title: str, source: str, rss_summary: str = "") -> dict[str, Any]:
    content = re.sub(r"\s+", " ", (content or "")).strip()

    if len(content) >= MIN_ARTICLE_CHARS:
        llm_summary = _openai_summary(content, title, source)
        if llm_summary:
            return {
                "summary": llm_summary,
                "summary_method": "llm_full_article",
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
