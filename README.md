# NewsBrief AI

A personal AI-assisted news briefing system focused on **important Malayalam/general news** and **technology news**.

The project is designed to reduce information overload by collecting news automatically, removing duplicates, analysing importance, classifying topics, and producing a ranked feed that can later be consumed by a mobile app.

## Current Architecture

```text
RSS Feeds
    ↓
Fetch & Normalize
    ↓
Deduplication
    ↓
Mathematical AI Engine
    ├── TF-IDF vectorization
    ├── Cosine similarity
    ├── Topic classification
    ├── Novelty detection
    ├── Importance scoring
    └── Recency/source weighting
    ↓
Ranked News
    ↓
data/news.json
```

## AI / Machine Learning

NewsBrief AI currently uses a lightweight mathematical AI engine implemented in pure Python. It does **not** require a paid AI API or a hosted LLM.

### Techniques used

- **TF-IDF** — represents article text as numerical vectors.
- **Cosine similarity** — measures similarity between articles and helps identify repetitive stories.
- **Prototype-based classification** — estimates whether an article is more relevant to `general` or `tech` topics.
- **Novelty scoring** — estimates how different an article is from the other collected stories.
- **Importance scoring** — combines important-event signals, topic relevance, source quality, and recency.

The implementation is in:

```text
src/ai_engine.py
```

## Project Structure

```text
newsbrief-ai/
├── .github/
│   └── workflows/
│       └── news.yml
├── config/
│   └── sources.json
├── data/
│   └── news.json
├── src/
│   ├── ai_engine.py
│   └── fetch_news.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/fetch_news.py
```

The generated feed is written to:

```text
data/news.json
```

## GitHub Actions

The repository includes a GitHub Actions workflow that:

1. Checks out the repository.
2. Sets up Python.
3. Installs dependencies.
4. Fetches configured RSS feeds.
5. Runs the mathematical AI processing pipeline.
6. Validates `data/news.json`.
7. Commits the updated feed when news changes.

The workflow supports **manual execution** and is scheduled to run every **30 minutes**.

## News Sources

Sources are configured in:

```text
config/sources.json
```

The initial configuration contains Malayalam/general news searches and technology-focused feeds covering areas such as:

- Kerala
- India
- World
- Science
- Artificial Intelligence
- Software Development
- Cybersecurity
- Open Source
- Android

## Data Model

Each article preserves its original source information, including:

- Original title
- Original URL
- Source
- Feed
- Publication time

AI-derived information is stored separately, so the original source data is not overwritten.

Example AI fields include:

```json
{
  "ai_category": "tech",
  "ai_confidence": 0.82,
  "ai_importance": 8,
  "ai_novelty": 0.71,
  "ai": {
    "method": "tfidf_cosine_prototype",
    "version": 1
  }
}
```

## Roadmap

### Phase 1 — Collection

- [x] RSS collection
- [x] Text normalization
- [x] URL/title deduplication
- [x] Automated GitHub Actions execution

### Phase 2 — Mathematical AI

- [x] TF-IDF vectorization
- [x] Cosine similarity
- [x] Topic classification
- [x] Novelty detection
- [x] Importance ranking
- [ ] Learn from personal user feedback
- [ ] Improve classification using a trained ML model

### Phase 3 — Language Intelligence

Future versions may add a local language model for tasks that mathematical NLP cannot handle well:

- AI-generated Malayalam titles
- Malayalam short summaries
- Detailed summaries
- Better semantic classification
- Entity extraction

A local model can be added later without replacing the mathematical ranking system.

### Phase 4 — Mobile Delivery

- [ ] Mobile application
- [ ] General/Malayalam news tab
- [ ] Technology news tab
- [ ] News images
- [ ] Source links
- [ ] Offline/cache support
- [ ] Optional notifications

## Design Goal

The main goal is **not simply to collect more news**. It is to identify the small number of stories that are most useful to the user while keeping the original source available for verification.

The project intentionally starts with transparent mathematical methods before introducing a larger language model. This makes the ranking system explainable, lightweight, and suitable for execution in GitHub Actions without a dedicated server.
