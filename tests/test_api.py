"""
test_api.py  —  integration tests for all Flask API endpoints.

Each test hits the live Flask test client and inspects the JSON response.
The database is reset between tests by the clean_db fixture in conftest.py.
"""

import json
import pytest
from app import SentimentRecord, db


# ============================================================
# Helpers
# ============================================================

def post_analyze(client, payload, content_type="application/json"):
    return client.post(
        "/analyze",
        data=json.dumps(payload),
        content_type=content_type,
    )


def _seed_records(app, count=5):
    """Insert *count* synthetic SentimentRecord rows for history/stats tests."""
    labels = ["positive", "negative", "neutral"]
    with app.app_context():
        for i in range(count):
            r = SentimentRecord(
                original_text=f"Sample text number {i}",
                processed_text=f"sample text number {i}",
                sentiment_label=labels[i % 3],
                compound_score=round(0.3 * (i % 3 - 1), 4),
                positive_score=0.4 if i % 3 == 0 else 0.0,
                neutral_score=0.6,
                negative_score=0.4 if i % 3 == 1 else 0.0,
                keywords=f"sample,text,number,{i}",
            )
            db.session.add(r)
        db.session.commit()


# ============================================================
# GET /
# ============================================================

class TestIndex:
    def test_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_returns_html(self, client):
        res = client.get("/")
        assert b"SentimentIQ" in res.data


# ============================================================
# POST /analyze
# ============================================================

class TestAnalyze:

    def test_positive_text_returns_201(self, client):
        res = post_analyze(client, {"text": "This is absolutely wonderful, I love it!"})
        assert res.status_code == 201

    def test_positive_text_sentiment_label(self, client):
        res = post_analyze(client, {"text": "This is absolutely wonderful, I love it!"})
        data = res.get_json()
        assert data["sentiment"] == "positive"

    def test_negative_text_sentiment_label(self, client):
        res = post_analyze(client, {"text": "This is terrible. I hate it so much."})
        data = res.get_json()
        assert data["sentiment"] == "negative"

    def test_neutral_text_sentiment_label(self, client):
        res = post_analyze(client, {"text": "The item arrived on Tuesday in a box."})
        data = res.get_json()
        assert data["sentiment"] == "neutral"

    def test_response_contains_required_keys(self, client):
        res = post_analyze(client, {"text": "Great product, very happy."})
        data = res.get_json()
        for key in ("id", "original_text", "processed_text", "sentiment",
                    "compound_score", "detailed_scores", "keywords", "timestamp"):
            assert key in data, f"Missing key: {key}"

    def test_detailed_scores_sum_to_one(self, client):
        res = post_analyze(client, {"text": "Pretty decent overall experience."})
        s = res.get_json()["detailed_scores"]
        total = round(s["pos"] + s["neu"] + s["neg"], 4)
        assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected ~1.0"

    def test_compound_score_in_range(self, client):
        res = post_analyze(client, {"text": "Not terrible, not great."})
        compound = res.get_json()["compound_score"]
        assert -1.0 <= compound <= 1.0

    def test_keywords_is_list(self, client):
        res = post_analyze(client, {"text": "Amazing quality and fast delivery."})
        assert isinstance(res.get_json()["keywords"], list)

    def test_negation_handled_correctly(self, client):
        """VADER running on original text must correctly handle 'not good'."""
        res = post_analyze(client, {"text": "This is absolutely not good at all."})
        data = res.get_json()
        assert data["sentiment"] in ("negative", "neutral")
        assert data["compound_score"] < 0.05

    def test_record_is_persisted(self, client, app):
        post_analyze(client, {"text": "Fantastic experience, highly recommend."})
        with app.app_context():
            count = SentimentRecord.query.count()
        assert count == 1

    def test_original_text_preserved(self, client):
        text = "Excellent product! 10/10 would buy again."
        res = post_analyze(client, {"text": text})
        assert res.get_json()["original_text"] == text

    # --- Validation failures ---

    def test_missing_text_field_returns_400(self, client):
        res = post_analyze(client, {"message": "oops"})
        assert res.status_code == 400

    def test_empty_string_returns_400(self, client):
        res = post_analyze(client, {"text": ""})
        assert res.status_code == 400

    def test_whitespace_only_returns_400(self, client):
        res = post_analyze(client, {"text": "   \n\t  "})
        assert res.status_code == 400

    def test_non_json_body_returns_400(self, client):
        res = client.post(
            "/analyze",
            data="text=hello",
            content_type="application/x-www-form-urlencoded",
        )
        assert res.status_code == 400

    def test_text_too_long_returns_413(self, client):
        res = post_analyze(client, {"text": "a" * 5001})
        assert res.status_code == 413

    def test_text_at_exact_limit_accepted(self, client):
        res = post_analyze(client, {"text": "a" * 5000})
        assert res.status_code == 201

    def test_error_response_has_error_key(self, client):
        res = post_analyze(client, {"text": ""})
        assert "error" in res.get_json()


# ============================================================
# GET /history
# ============================================================

class TestHistory:

    def test_empty_db_returns_200(self, client):
        res = client.get("/history")
        assert res.status_code == 200

    def test_empty_db_returns_empty_records(self, client):
        data = client.get("/history").get_json()
        assert data["records"] == []
        assert data["total"] == 0

    def test_records_returned_after_analysis(self, client, app):
        _seed_records(app, count=3)
        data = client.get("/history").get_json()
        assert len(data["records"]) == 3
        assert data["total"] == 3

    def test_default_limit_is_20(self, client, app):
        _seed_records(app, count=25)
        data = client.get("/history").get_json()
        assert len(data["records"]) == 20
        assert data["total"] == 25

    def test_custom_limit_respected(self, client, app):
        _seed_records(app, count=10)
        data = client.get("/history?limit=5").get_json()
        assert len(data["records"]) == 5

    def test_limit_capped_at_100(self, client, app):
        _seed_records(app, count=5)
        data = client.get("/history?limit=999").get_json()
        assert data["limit"] == 100

    def test_offset_pagination(self, client, app):
        _seed_records(app, count=10)
        page1 = client.get("/history?limit=5&offset=0").get_json()
        page2 = client.get("/history?limit=5&offset=5").get_json()
        ids_p1 = {r["id"] for r in page1["records"]}
        ids_p2 = {r["id"] for r in page2["records"]}
        assert ids_p1.isdisjoint(ids_p2), "Pages must not overlap"

    def test_records_newest_first(self, client, app):
        _seed_records(app, count=5)
        records = client.get("/history").get_json()["records"]
        timestamps = [r["timestamp"] for r in records]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_invalid_limit_returns_400(self, client):
        res = client.get("/history?limit=banana")
        assert res.status_code == 400

    def test_record_has_required_keys(self, client, app):
        _seed_records(app, count=1)
        record = client.get("/history?limit=1").get_json()["records"][0]
        for key in ("id", "timestamp", "original_text", "sentiment",
                    "compound_score", "detailed_scores", "keywords"):
            assert key in record, f"Missing key: {key}"


# ============================================================
# GET /trends/summary
# ============================================================

class TestTrendsSummary:

    def test_returns_200(self, client):
        assert client.get("/trends/summary").status_code == 200

    def test_zero_counts_on_empty_db(self, client):
        data = client.get("/trends/summary").get_json()
        assert data["total_analyzed"] == 0
        assert data["positive_count"] == 0
        assert data["negative_count"] == 0
        assert data["neutral_count"] == 0

    def test_counts_correct_after_seeding(self, client, app):
        # Seed: 3 positive, 2 negative, 1 neutral
        with app.app_context():
            for label, score in [
                ("positive", 0.8), ("positive", 0.6), ("positive", 0.5),
                ("negative", -0.7), ("negative", -0.4),
                ("neutral",  0.02),
            ]:
                db.session.add(SentimentRecord(
                    original_text="x", processed_text="x",
                    sentiment_label=label, compound_score=score,
                    positive_score=0.0, neutral_score=0.0, negative_score=0.0,
                ))
            db.session.commit()

        data = client.get("/trends/summary").get_json()
        assert data["total_analyzed"] == 6
        assert data["positive_count"] == 3
        assert data["negative_count"] == 2
        assert data["neutral_count"] == 1

    def test_avg_compound_is_numeric(self, client, app):
        _seed_records(app, count=3)
        data = client.get("/trends/summary").get_json()
        assert isinstance(data["avg_compound"], float)

    def test_response_has_required_keys(self, client):
        data = client.get("/trends/summary").get_json()
        for key in ("total_analyzed", "positive_count", "negative_count",
                    "neutral_count", "avg_compound", "last_updated"):
            assert key in data


# ============================================================
# GET /health
# ============================================================

class TestHealth:

    def test_returns_200_when_db_ok(self, client):
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"
        assert data["db"] == "ok"

    def test_has_timestamp(self, client):
        data = client.get("/health").get_json()
        assert "timestamp" in data
