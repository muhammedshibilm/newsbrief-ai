# NewsBrief AI

A personal AI-assisted news briefing system focused on **important Malayalam/general news** and **technology news**.

The project reduces information overload by collecting news automatically, removing duplicates, analysing importance, learning from feedback, and producing a ranked feed for a future mobile app.

## Architecture

```text
RSS Sources
    ↓
Fetch + Normalize
    ↓
Exact Deduplication
    ↓
Mathematical AI Engine
    ├── TF-IDF (1,2)-grams
    ├── Cosine similarity
    ├── Topic classification
    ├── Novelty detection
    ├── Entity/keyword signals
    ├── Importance scoring
    └── Recency + source weighting
    ↓
Personal Learning Layer
    ├── User feedback (like / skip)
    ├── Source preference signals
    └── Category preference signals
    ↓
Personalized Ranking
    ↓
data/news.json
    ↓
Future Mobile App
```

## AI / Machine Learning

NewsBrief AI starts with transparent, lightweight mathematical methods rather than a hosted LLM. This keeps the pipeline inexpensive and suitable for GitHub Actions.

### Current techniques

- **TF-IDF** — converts article text into numerical feature vectors.
- **1,2-gram features** — captures both individual terms and short phrases.
- **Cosine similarity** — measures textual similarity between stories.
- **Prototype classification** — estimates `general` versus `tech` relevance.
- **Novelty scoring** — identifies stories that are less repetitive.
- **Importance scoring** — combines important-event signals, topic relevance, source quality, and recency.
- **Feedback learning** — stores user `like` / `skip` signals and derives conservative source/category preference boosts.

The core implementation is in `src/ai_engine.py`; feedback storage and preference logic are in `src/learning.py`.

## Feedback Learning

Feedback is stored in:

```text
data/feedback.json
```

Each feedback record can contain the article URL, text, category, source, label, and timestamp. The current learning layer deliberately uses small preference boosts instead of allowing a tiny feedback dataset to completely change the ranking.

A future trained classifier can be introduced after enough labelled feedback is collected. When that happens, the TF-IDF vectorizer and classifier should be persisted together so that the feature space remains consistent.

## Project Structure

```text
newsbrief-ai/
├── .github/
│   └── workflows/
│       └── news.yml
├── config/
│   └── sources.json
├── data/
│   ├── news.json
│   └── feedback.json
├── src/
│   ├── ai_engine.py
│   ├── fetch_news.py
│   └── learning.py
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
5. Runs mathematical AI processing.
6. Applies the personal learning layer when feedback exists.
7. Validates `data/news.json`.
8. Commits the updated feed when news changes.

The workflow supports manual execution and is scheduled every **30 minutes**.

## News Sources

Sources are configured in `config/sources.json` and currently cover:

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

Original source information is preserved, while AI-derived fields are stored separately.

Example:

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
- [x] Feedback storage and preference signals
- [ ] Fully trained personal classifier
- [ ] Adaptive source reliability model
- [ ] Better entity extraction
- [ ] Semantic deduplication with embeddings

### Phase 3 — Language Intelligence

Possible future additions:

- AI-generated Malayalam titles
- Malayalam short summaries
- Detailed summaries
- Stronger semantic classification
- Entity extraction
- Optional local LLM through Ollama

The local language model will be an optional language layer; the mathematical ranking system remains the foundation.

### Phase 4 — Mobile Delivery

- [ ] Mobile application
- [ ] General/Malayalam news tab
- [ ] Technology news tab
- [ ] News images
- [ ] Source links
- [ ] Offline/cache support
- [ ] User feedback controls
- [ ] Optional notifications

## Design Goal

The goal is **not to collect more news**. It is to identify a small number of useful stories while preserving the original source for verification.

The project intentionally starts with explainable mathematical NLP and lightweight machine learning before introducing larger language models. This makes the system easier to understand, test, and run without a dedicated 24/7 server.
