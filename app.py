# app.py
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import datetime
import os
from dotenv import load_dotenv

# Import from your custom modules
from nltk_setup import download_nltk_resources
from sentiment_analyzer import get_sentiment_analysis

# --- Initial Setup ---
# Ensure NLTK resources are available on startup
# For production (like Elastic Beanstalk), .ebextensions or similar is preferred for setup.
# This call here is a good practice for local dev or simpler deployments.
download_nltk_resources()

load_dotenv() # Load environment variables from .env file for local dev

app = Flask(__name__)

# --- Database Configuration ---
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME')

if DB_USER and DB_PASSWORD and DB_HOST and DB_NAME:
    app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'sentiment_data.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    print(f"WARNING: Using local SQLite database at {db_path}. Set DB environment variables for PostgreSQL.")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Model ---
class SentimentRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    processed_text = db.Column(db.Text, nullable=True)
    sentiment_label = db.Column(db.String(10), nullable=False)
    compound_score = db.Column(db.Float, nullable=False)
    positive_score = db.Column(db.Float, nullable=True)
    neutral_score = db.Column(db.Float, nullable=True)
    negative_score = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<SentimentRecord {self.id} - {self.sentiment_label}>"

with app.app_context():
    db.create_all()

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_feedback_route():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    feedback_text = data['text']
    if not feedback_text.strip(): # Check if the text is not just whitespace
        return jsonify({'error': 'Text cannot be empty or just whitespace'}), 400

    # Use the imported sentiment analysis function
    original, processed, sentiment, compound, all_scores = get_sentiment_analysis(feedback_text)

    # Store in database
    try:
        record = SentimentRecord(
            original_text=original,
            processed_text=processed,
            sentiment_label=sentiment,
            compound_score=compound,
            positive_score=all_scores['pos'],
            neutral_score=all_scores['neu'],
            negative_score=all_scores['neg']
        )
        db.session.add(record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Database error: {e}")
        # Consider whether to notify the user or just log
        # return jsonify({'error': 'Could not store analysis result due to a server issue.'}), 500


    return jsonify({
        'original_text': original,
        'processed_text': processed,
        'sentiment': sentiment,
        'compound_score': compound,
        'detailed_scores': all_scores,
        'timestamp': datetime.datetime.utcnow().isoformat() # Use current time for response
    })

@app.route('/trends/summary', methods=['GET'])
def trends_summary():
    try:
        total_records = SentimentRecord.query.count()
        positive_count = SentimentRecord.query.filter_by(sentiment_label='positive').count()
        negative_count = SentimentRecord.query.filter_by(sentiment_label='negative').count()
        neutral_count = SentimentRecord.query.filter_by(sentiment_label='neutral').count()

        return jsonify({
            'total_analyzed': total_records,
            'positive_feedbacks': positive_count,
            'negative_feedbacks': negative_count,
            'neutral_feedbacks': neutral_count,
            'last_updated': datetime.datetime.utcnow().isoformat()
        })
    except Exception as e:
        app.logger.error(f"Trend summary error: {e}")
        return jsonify({'error': 'Could not retrieve trend data'}), 500

if __name__ == '__main__':
    # For production, use a WSGI server like Gunicorn (specified in Procfile)
    # The host='0.0.0.0' is important for Docker/Gunicorn to bind correctly
    port = int(os.environ.get("PORT", 5000)) # PORT env var is common for deployment platforms
    app.run(debug=os.environ.get("FLASK_DEBUG", "False").lower() == "true", host='0.0.0.0', port=port)