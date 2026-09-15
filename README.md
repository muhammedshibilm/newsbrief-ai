# NewsBrief AI

Personal AI-powered news briefing focused on important Malayalam/general news and technology news.

## V1

RSS feeds -> normalize -> deduplicate -> importance score -> data/news.json

Qwen and Firebase are intentionally not included in V1. They will be added after the collector is verified.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/fetch_news.py
```

The generated feed is written to `data/news.json`.

## GitHub Actions

The workflow supports manual execution and runs every 30 minutes. It fetches configured feeds and commits a changed `data/news.json` to the main branch.

## Data integrity

Every article keeps its original source, URL, title, and publication time. Derived fields are kept separate so later AI-generated titles and summaries do not replace source information.
