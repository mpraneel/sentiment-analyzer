"""
conftest.py  —  shared pytest fixtures

The app fixture spins up a Flask test instance backed by an in-memory SQLite
database so tests are fully self-contained and never touch the real database.
"""

import pytest
from app import app as flask_app, db as _db, SentimentRecord


@pytest.fixture(scope="session")
def app():
    """
    Application fixture — session-scoped so Flask is only initialised once.
    Uses an in-memory SQLite DB that is created fresh for the test run.
    """
    flask_app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        }
    )
    with flask_app.app_context():
        _db.create_all()
        yield flask_app
        _db.drop_all()


@pytest.fixture(scope="session")
def client(app):
    """Flask test client — reused across all tests in the session."""
    return app.test_client()


@pytest.fixture(autouse=True)
def clean_db(app):
    """
    Wipe all SentimentRecord rows before every test so each test
    starts from a known-empty state.
    """
    with app.app_context():
        _db.session.query(SentimentRecord).delete()
        _db.session.commit()
        yield
