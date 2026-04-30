"""
run.py  —  convenience dev-server launcher

Usage:
    python run.py

For production use Gunicorn:
    gunicorn app:app
"""

from app import app

if __name__ == "__main__":
    app.run(debug=True)
