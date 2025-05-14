import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize components once when the module is loaded for efficiency
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("[sentiment_analyzer.py] Stopwords not found. Make sure to run nltk_setup.py or download them.")
    # You might want to call nltk_setup.download_nltk_resources() here as a fallback,
    # or ensure app.py calls it first. For simplicity, we assume it's handled.
    stop_words = set() # Default to empty if not found to avoid crashing

lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    text = str(text).lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1
    ]
    return " ".join(processed_tokens)

def get_sentiment_analysis(text_input):
    """
    Analyzes sentiment of the input text.
    Returns: original_text, processed_text, sentiment_label, compound_score, all_vader_scores
    """
    original_text = str(text_input) # Ensure it's a string
    processed = preprocess_text(original_text)

    if not processed: # Handle empty string after preprocessing
        return original_text, processed, "neutral", 0.0, {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    scores = analyzer.polarity_scores(processed)
    compound_score = scores['compound']

    if compound_score >= 0.05:
        sentiment_label = "positive"
    elif compound_score <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    return original_text, processed, sentiment_label, compound_score, scores

if __name__ == '__main__':
    # Example usage:
    # Ensure nltk_setup.py has been run or resources are available
    # from nltk_setup import download_nltk_resources
    # download_nltk_resources() # If you want to ensure downloads when running this file directly

    test_feedback = "This is a wonderfully fantastic product! I love it."
    original, processed, label, compound, all_scores = get_sentiment_analysis(test_feedback)
    print(f"Original: {original}")
    print(f"Processed: {processed}")
    print(f"Sentiment: {label} (Compound: {compound:.4f})")
    print(f"All Scores: {all_scores}")

    test_feedback_neg = "This is terrible, I hate it so much."
    original, processed, label, compound, all_scores = get_sentiment_analysis(test_feedback_neg)
    print(f"\nOriginal: {original}")
    print(f"Processed: {processed}")
    print(f"Sentiment: {label} (Compound: {compound:.4f})")
    print(f"All Scores: {all_scores}")