"""
sentiment_analyzer.py

Handles text preprocessing and VADER-based sentiment scoring.

Important design note:
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    designed for raw, social-media-style text. It relies on punctuation,
    capitalization, and negation words ("not", "never") for its scoring.
    Preprocessing that strips those signals (stopword removal, lemmatization)
    is counterproductive for VADER.

    This module therefore runs VADER on the ORIGINAL text and uses the
    preprocessing pipeline separately to extract content keywords for display.
"""

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Module-level singletons — initialised once for efficiency
# ---------------------------------------------------------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    print(
        "[sentiment_analyzer] Stopwords not found. "
        "Run nltk_setup.py or call download_nltk_resources() first."
    )
    stop_words = set()

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Maximum input length accepted by the API (characters)
MAX_TEXT_LENGTH = 5000


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Returns a cleaned, lemmatised string of content keywords.

    This is used for keyword display, NOT for VADER scoring.
    Steps: lowercase → strip punctuation → strip digits →
           tokenise → remove stopwords → lemmatise → rejoin.
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\d+", "", text)        # remove numbers
    tokens = word_tokenize(text)
    content_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(content_tokens)


def extract_keywords(text: str) -> list[str]:
    """
    Returns the top content keywords from *text* (up to 15), sorted by length
    so the most distinctive words appear first.
    """
    processed = preprocess_text(text)
    if not processed:
        return []
    tokens = processed.split()
    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    unique = [t for t in tokens if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]
    # Sort longer (more specific) words first, cap at 15
    return sorted(unique, key=len, reverse=True)[:15]


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def get_sentiment_analysis(text_input: str) -> tuple[str, str, str, float, dict, list[str]]:
    """
    Analyses the sentiment of *text_input*.

    VADER is run on the original (unmodified) text so that capitalisation,
    punctuation, and negation words contribute their full signal.
    Preprocessing is used only to extract display keywords.

    Returns
    -------
    original_text   : str   — the input as received
    processed_text  : str   — stopword-free lemmatised keywords (for display)
    sentiment_label : str   — "positive" | "negative" | "neutral"
    compound_score  : float — VADER compound score in [-1, 1]
    all_scores      : dict  — full VADER scores {"neg", "neu", "pos", "compound"}
    keywords        : list  — top content keywords extracted from the text
    """
    original_text = str(text_input).strip()

    # Run VADER on the original text for accurate scoring
    scores = analyzer.polarity_scores(original_text)
    compound_score = scores["compound"]

    # Label thresholds recommended by the VADER paper
    if compound_score >= 0.05:
        sentiment_label = "positive"
    elif compound_score <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Extract keywords separately for display purposes
    processed_text = preprocess_text(original_text)
    keywords = extract_keywords(original_text)

    return original_text, processed_text, sentiment_label, compound_score, scores, keywords


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from nltk_setup import download_nltk_resources
    download_nltk_resources()

    samples = [
        "This is a wonderfully fantastic product! I absolutely LOVE it.",
        "This is terrible. I hate it so much — worst purchase ever.",
        "The item arrived on Tuesday. It is a box.",
        "Not bad at all! Could be better, but I'm not unhappy with it.",
    ]

    for sample in samples:
        orig, processed, label, compound, all_scores, kw = get_sentiment_analysis(sample)
        print(f"\nText     : {orig}")
        print(f"Sentiment: {label} (compound={compound:.4f})")
        print(f"Scores   : pos={all_scores['pos']:.3f}  neu={all_scores['neu']:.3f}  neg={all_scores['neg']:.3f}")
        print(f"Keywords : {kw}")
