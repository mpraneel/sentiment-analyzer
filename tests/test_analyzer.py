"""
test_analyzer.py  —  unit tests for sentiment_analyzer.py

These tests are isolated from Flask entirely — they call the analysis
functions directly and verify their output contracts.
"""

import pytest
from sentiment_analyzer import (
    preprocess_text,
    extract_keywords,
    get_sentiment_analysis,
    MAX_TEXT_LENGTH,
)


# ============================================================
# preprocess_text
# ============================================================

class TestPreprocessText:

    def test_returns_string(self):
        assert isinstance(preprocess_text("Hello world"), str)

    def test_lowercases_input(self):
        result = preprocess_text("HELLO WORLD")
        assert result == result.lower()

    def test_removes_punctuation(self):
        result = preprocess_text("Hello, world! How's it going?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_removes_digits(self):
        result = preprocess_text("I bought 3 items for $10 each")
        assert "3" not in result
        assert "10" not in result

    def test_removes_stopwords(self):
        # "the", "is", "a" are stopwords
        result = preprocess_text("The cat is a mammal")
        tokens = result.split()
        for stopword in ("the", "is", "a"):
            assert stopword not in tokens

    def test_empty_string_returns_empty(self):
        assert preprocess_text("") == ""

    def test_all_stopwords_returns_empty(self):
        # A sentence composed entirely of stopwords
        result = preprocess_text("the a an and is are was were")
        assert result == ""

    def test_lemmatisation_applied(self):
        # "running" should lemmatise to "running" or "run"
        result = preprocess_text("The cats are running quickly")
        assert "cat" in result or "cats" in result   # lemmatiser may vary
        assert "run" in result or "running" in result


# ============================================================
# extract_keywords
# ============================================================

class TestExtractKeywords:

    def test_returns_list(self):
        assert isinstance(extract_keywords("Amazing quality product"), list)

    def test_no_duplicates(self):
        kws = extract_keywords("good good good product product")
        assert len(kws) == len(set(kws))

    def test_max_15_keywords(self):
        long_text = " ".join([f"word{i}" for i in range(50)])
        assert len(extract_keywords(long_text)) <= 15

    def test_empty_input_returns_empty_list(self):
        assert extract_keywords("") == []

    def test_stopword_only_input_returns_empty_list(self):
        assert extract_keywords("the and or but") == []


# ============================================================
# get_sentiment_analysis
# ============================================================

class TestGetSentimentAnalysis:

    def test_returns_six_tuple(self):
        result = get_sentiment_analysis("Great product!")
        assert len(result) == 6

    def test_original_text_preserved(self):
        text = "  Hello World!  "
        original, *_ = get_sentiment_analysis(text)
        assert original == text.strip()

    def test_positive_label(self):
        _, _, label, _, _, _ = get_sentiment_analysis(
            "This is absolutely wonderful, I love it!"
        )
        assert label == "positive"

    def test_negative_label(self):
        _, _, label, _, _, _ = get_sentiment_analysis(
            "Terrible experience. I hate this product so much."
        )
        assert label == "negative"

    def test_neutral_label(self):
        _, _, label, _, _, _ = get_sentiment_analysis(
            "The package arrived on Tuesday."
        )
        assert label == "neutral"

    def test_compound_score_in_valid_range(self):
        _, _, _, compound, _, _ = get_sentiment_analysis("It was okay I guess")
        assert -1.0 <= compound <= 1.0

    def test_detailed_scores_keys_present(self):
        _, _, _, _, scores, _ = get_sentiment_analysis("Nice.")
        assert set(scores.keys()) >= {"pos", "neu", "neg", "compound"}

    def test_detailed_scores_sum_to_one(self):
        _, _, _, _, scores, _ = get_sentiment_analysis("Fantastic!")
        total = round(scores["pos"] + scores["neu"] + scores["neg"], 4)
        assert abs(total - 1.0) < 0.01

    def test_keywords_is_list(self):
        _, _, _, _, _, keywords = get_sentiment_analysis("Amazing quality!")
        assert isinstance(keywords, list)

    def test_negation_reduces_score(self):
        """
        VADER runs on the original text, so 'not good' should score lower
        than 'good'. This test verifies that VADER is NOT being called on
        preprocessed text (which would strip 'not').
        """
        _, _, _, compound_positive, _, _ = get_sentiment_analysis("This is good.")
        _, _, _, compound_negated,  _, _ = get_sentiment_analysis("This is not good.")
        assert compound_negated < compound_positive, (
            "Negation should reduce the compound score. "
            "Check that VADER is running on the original text, not preprocessed text."
        )

    def test_capitalisation_amplifies_score(self):
        """VADER should score 'GREAT' higher than 'great'."""
        _, _, _, score_lower, _, _ = get_sentiment_analysis("great product")
        _, _, _, score_upper, _, _ = get_sentiment_analysis("GREAT product")
        assert score_upper >= score_lower

    def test_max_text_length_constant_is_positive(self):
        assert MAX_TEXT_LENGTH > 0

    def test_very_short_text(self):
        result = get_sentiment_analysis("OK")
        assert len(result) == 6

    def test_unicode_text(self):
        result = get_sentiment_analysis("Magnifique! Très bien.")
        assert len(result) == 6

    def test_text_with_only_punctuation(self):
        # Should not raise — VADER handles this gracefully
        _, _, label, compound, _, _ = get_sentiment_analysis("!!! ??? ...")
        assert label in ("positive", "negative", "neutral")

    def test_sentiment_label_matches_compound_positive(self):
        _, _, label, compound, _, _ = get_sentiment_analysis(
            "I absolutely love this, it is the best thing ever!"
        )
        if compound >= 0.05:
            assert label == "positive"

    def test_sentiment_label_matches_compound_negative(self):
        _, _, label, compound, _, _ = get_sentiment_analysis(
            "Worst purchase of my life. Total disaster."
        )
        if compound <= -0.05:
            assert label == "negative"
