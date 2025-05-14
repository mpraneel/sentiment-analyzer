from flask import Blueprint, request, render_template
from .sentiment import analyze_sentiment
from models.database import save_feedback

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        feedback = request.form['feedback']
        sentiment = analyze_sentiment(feedback)
        save_feedback(feedback, sentiment)
    return render_template('index.html', sentiment=sentiment)
