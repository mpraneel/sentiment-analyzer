# SentimentIQ

A Flask web application that performs real-time sentiment analysis on free-form text using NLTK's VADER model, persists every result to a database, and surfaces a polished single-page UI with live score breakdowns, keyword extraction, a paginated history browser, and an aggregate statistics dashboard.

---

## Features

- **VADER-based sentiment scoring** — compound score in \[-1, 1\] plus positive / neutral / negative component scores
- **NLP keyword extraction** — tokenisation, stop-word removal, and lemmatisation surface the most meaningful content words
- **Auto-persist** — every analysis is saved to SQLite (dev) or PostgreSQL (prod) via SQLAlchemy
- **REST API** — clean JSON endpoints for programmatic use
- **History browser** — paginated table of all past analyses
- **Statistics dashboard** — aggregate counts, distribution donut chart, and average compound score meter
- **Responsive UI** — works on desktop and mobile
- **Health endpoint** — lightweight liveness + DB readiness probe for deployment platforms

---

## Tech stack

| Layer       | Technology                            |
|-------------|---------------------------------------|
| Web server  | Flask 3                               |
| ORM         | Flask-SQLAlchemy                      |
| NLP / ML    | NLTK 3.9 (VADER, punkt\_tab, WordNet) |
| Database    | SQLite (dev) / PostgreSQL (prod)      |
| WSGI        | Gunicorn                              |
| Frontend    | Vanilla HTML / CSS / JS (no build step) |

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd sentiment_tool
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables (optional)

```bash
cp .env.example .env
# Edit .env if you want to use PostgreSQL or change the port.
# Leave the DB_* variables commented out to use the local SQLite fallback.
```

### 4. Download NLTK data

The app downloads required NLTK packages automatically on first startup. To pre-download manually:

```bash
python nltk_setup.py
```

### 5. Run

```bash
python app.py
# or
python run.py
```

Open `http://localhost:5000` in your browser.

---

## API reference

All endpoints return JSON. Error responses have the shape `{ "error": "<message>" }`.

### `POST /analyze`

Analyse the sentiment of a text snippet.

**Request body**
```json
{ "text": "Your text here" }
```

**Response `201`**
```json
{
  "id":              42,
  "original_text":   "Your text here",
  "processed_text":  "text",
  "sentiment":       "positive",
  "compound_score":  0.6369,
  "detailed_scores": { "pos": 0.694, "neu": 0.306, "neg": 0.0 },
  "keywords":        ["text"],
  "timestamp":       "2025-04-29T12:00:00.000000"
}
```

**Constraints** — `text` must be a non-empty string, maximum 5 000 characters.

---

### `GET /history`

Return recent analyses, newest first.

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `limit`     | int  | 20      | Max records to return (capped at 100) |
| `offset`    | int  | 0       | Pagination offset |

**Response `200`**
```json
{
  "records": [ ...SentimentRecord ],
  "total":   150,
  "limit":   20,
  "offset":  0
}
```

---

### `GET /trends/summary`

Aggregate sentiment statistics across all stored records.

**Response `200`**
```json
{
  "total_analyzed":  150,
  "positive_count":  80,
  "negative_count":  40,
  "neutral_count":   30,
  "avg_compound":    0.1842,
  "last_updated":    "2025-04-29T12:00:00.000000"
}
```

---

### `GET /health`

Liveness / readiness probe.

**Response `200`** (healthy)
```json
{ "status": "ok", "db": "ok", "timestamp": "..." }
```

**Response `503`** (DB unreachable)
```json
{ "status": "degraded", "db": "error", "timestamp": "..." }
```

---

## Production deployment

### Gunicorn

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 4
```

### PostgreSQL

Set the following environment variables before starting:

```
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=5432
DB_NAME=sentimentiq
```

The app automatically switches from SQLite to PostgreSQL when all four `DB_*` variables are present.

---

## Project structure

```
sentiment_tool/
├── app.py                  # Flask app, SQLAlchemy model, all routes
├── sentiment_analyzer.py   # VADER scoring + NLP keyword extraction
├── nltk_setup.py           # NLTK resource download helper
├── run.py                  # Dev-server launcher
├── Procfile                # Gunicorn entrypoint for deployment platforms
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Dev/test dependencies (includes pytest)
├── pytest.ini              # Pytest configuration
├── .env.example
├── .gitignore
├── tests/
│   ├── conftest.py         # Shared fixtures (in-memory SQLite test DB)
│   ├── test_api.py         # Integration tests for all API endpoints
│   └── test_analyzer.py    # Unit tests for sentiment_analyzer
├── templates/
│   └── index.html          # Single-page UI (Analyze / History / Stats)
└── static/
    └── style.css
```

---

## How sentiment scoring works

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based model built for social and expressive text. Unlike bag-of-words approaches it handles:

- **Capitalisation** — "GREAT" scores higher than "great"
- **Punctuation** — "great!!!" scores higher than "great"
- **Negation** — "not great" is handled correctly
- **Degree modifiers** — "extremely good" vs "kind of good"

The raw text is fed directly into VADER so none of these signals are lost. A separate NLP preprocessing pass (lowercase → remove punctuation / digits → tokenise → remove stop-words → lemmatise) is run in parallel purely to extract display keywords.

The compound score is mapped to labels using the thresholds recommended in the original VADER paper:

| Compound score | Label    |
|----------------|----------|
| ≥ 0.05         | Positive |
| ≤ −0.05        | Negative |
| Between        | Neutral  |

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use an in-memory SQLite database and require no external services. The suite covers all API endpoints and the core sentiment analysis logic, including a specific regression test that verifies negation handling (VADER must run on the original text, not preprocessed text).
