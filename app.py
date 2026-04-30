"""
app.py  —  Flask Sentiment Analysis API

Entry point for the application. Defines the Flask app, SQLAlchemy model,
and all API routes. Run via:  python app.py   (dev)  or  gunicorn app:app  (prod)
"""

import datetime
import os

from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

from nltk_setup import download_nltk_resources
from sentiment_analyzer import MAX_TEXT_LENGTH, get_sentiment_analysis

# ---------------------------------------------------------------------------
# Startup — ensure NLTK resources are present before any request is served
# ---------------------------------------------------------------------------
download_nltk_resources()

load_dotenv()  # Load .env for local development (no-op in production)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Database configuration
# Falls back to a local SQLite file when PostgreSQL env vars are absent,
# which makes the project run out-of-the-box for local development.
# ---------------------------------------------------------------------------
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")

if DB_USER and DB_PASSWORD and DB_HOST and DB_NAME:
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    app.logger.info("Using PostgreSQL database.")
else:
    db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "sentiment_data.db")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.logger.warning(
        f"PostgreSQL env vars not set — using local SQLite at {db_path}. "
        "Set DB_USER / DB_PASSWORD / DB_HOST / DB_NAME for PostgreSQL."
    )

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ---------------------------------------------------------------------------
# Database model
# ---------------------------------------------------------------------------

class SentimentRecord(db.Model):
    """Persisted record of a single sentiment analysis request."""

    __tablename__ = "sentiment_records"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(
        db.DateTime, default=datetime.datetime.utcnow, nullable=False, index=True
    )
    original_text = db.Column(db.Text, nullable=False)
    processed_text = db.Column(db.Text, nullable=True)
    sentiment_label = db.Column(db.String(10), nullable=False, index=True)
    compound_score = db.Column(db.Float, nullable=False)
    positive_score = db.Column(db.Float, nullable=True)
    neutral_score = db.Column(db.Float, nullable=True)
    negative_score = db.Column(db.Float, nullable=True)
    keywords = db.Column(db.Text, nullable=True)  # comma-separated keyword list

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "original_text": self.original_text,
            "processed_text": self.processed_text,
            "sentiment": self.sentiment_label,
            "compound_score": round(self.compound_score, 4),
            "detailed_scores": {
                "pos": round(self.positive_score or 0.0, 4),
                "neu": round(self.neutral_score or 0.0, 4),
                "neg": round(self.negative_score or 0.0, 4),
            },
            "keywords": self.keywords.split(",") if self.keywords else [],
        }

    def __repr__(self) -> str:
        return f"<SentimentRecord #{self.id} {self.sentiment_label}>"


with app.app_context():
    db.create_all()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _json_error(message: str, status: int):
    """Return a standardised JSON error response."""
    return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyse the sentiment of a text snippet.

    Request body (JSON):
        { "text": "<string>" }

    Response (JSON):
        {
          "id":              int,
          "original_text":   str,
          "processed_text":  str,
          "sentiment":       "positive" | "negative" | "neutral",
          "compound_score":  float,   // [-1.0, 1.0]
          "detailed_scores": { "pos": float, "neu": float, "neg": float },
          "keywords":        [str, ...],
          "timestamp":       str      // ISO-8601 UTC
        }
    """
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return _json_error("Request body must be JSON with a 'text' field.", 400)

    text = data["text"]

    if not isinstance(text, str):
        return _json_error("'text' must be a string.", 400)

    text = text.strip()

    if not text:
        return _json_error("'text' cannot be empty or whitespace.", 400)

    if len(text) > MAX_TEXT_LENGTH:
        return _json_error(
            f"'text' exceeds the maximum allowed length of {MAX_TEXT_LENGTH} characters.", 413
        )

    # Run analysis
    original, processed, sentiment, compound, scores, keywords = get_sentiment_analysis(text)

    # Persist to database
    record = SentimentRecord(
        original_text=original,
        processed_text=processed,
        sentiment_label=sentiment,
        compound_score=compound,
        positive_score=scores["pos"],
        neutral_score=scores["neu"],
        negative_score=scores["neg"],
        keywords=",".join(keywords),
    )
    try:
        db.session.add(record)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.error(f"DB write failed: {exc}")
        # Analysis succeeded — return the result even if persistence failed
        return jsonify({
            "id": None,
            "original_text": original,
            "processed_text": processed,
            "sentiment": sentiment,
            "compound_score": round(compound, 4),
            "detailed_scores": {
                "pos": round(scores["pos"], 4),
                "neu": round(scores["neu"], 4),
                "neg": round(scores["neg"], 4),
            },
            "keywords": keywords,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "warning": "Result could not be saved to the database.",
        })

    return jsonify(record.to_dict()), 201


@app.route("/history", methods=["GET"])
def history():
    """
    Return the most recent analyses.

    Query params:
        limit  (int, default 20, max 100) — number of records to return
        offset (int, default 0)           — pagination offset

    Response (JSON):
        {
          "records": [ ...SentimentRecord.to_dict() ],
          "total":   int,
          "limit":   int,
          "offset":  int
        }
    """
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        return _json_error("'limit' and 'offset' must be integers.", 400)

    try:
        total = SentimentRecord.query.count()
        records = (
            SentimentRecord.query
            .order_by(SentimentRecord.timestamp.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return jsonify({
            "records": [r.to_dict() for r in records],
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    except Exception as exc:
        app.logger.error(f"History query failed: {exc}")
        return _json_error("Could not retrieve history.", 500)


@app.route("/trends/summary", methods=["GET"])
def trends_summary():
    """
    Return aggregate sentiment counts and average compound score.

    Response (JSON):
        {
          "total_analyzed":    int,
          "positive_count":    int,
          "negative_count":    int,
          "neutral_count":     int,
          "avg_compound":      float,
          "last_updated":      str   // ISO-8601 UTC
        }
    """
    try:
        from sqlalchemy import func

        total = SentimentRecord.query.count()
        positive = SentimentRecord.query.filter_by(sentiment_label="positive").count()
        negative = SentimentRecord.query.filter_by(sentiment_label="negative").count()
        neutral = SentimentRecord.query.filter_by(sentiment_label="neutral").count()
        avg_row = db.session.query(func.avg(SentimentRecord.compound_score)).scalar()
        avg_compound = round(float(avg_row), 4) if avg_row is not None else 0.0

        return jsonify({
            "total_analyzed": total,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "avg_compound": avg_compound,
            "last_updated": datetime.datetime.utcnow().isoformat(),
        })
    except Exception as exc:
        app.logger.error(f"Trends query failed: {exc}")
        return _json_error("Could not retrieve trend data.", 500)


@app.route("/health", methods=["GET"])
def health():
    """
    Lightweight liveness / readiness probe for deployment platforms.

    Response (JSON):
        { "status": "ok", "db": "ok" | "error", "timestamp": str }
    """
    db_status = "ok"
    try:
        db.session.execute(db.text("SELECT 1"))
    except Exception as exc:
        app.logger.error(f"Health check DB ping failed: {exc}")
        db_status = "error"

    status_code = 200 if db_status == "ok" else 503
    return jsonify({
        "status": "ok" if db_status == "ok" else "degraded",
        "db": db_status,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }), status_code


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
