"""
app/database.py
----------------------------------------------------------------------
Defines and creates the PostgreSQL schema for the Psychological
Assessment System, and provides a small helper function for getting
a connection.

Tables:
  users              -- one row per registered person
  survey_responses   -- one row per completed survey, storing the raw
                        text answers as JSON so we always know exactly
                        what the user actually selected
  assessment_results -- one row per completed survey, storing what the
                        Random Forest predicted for that submission

Why split survey_responses and assessment_results into two tables
instead of one?
  - survey_responses is raw INPUT data (what the user said)
  - assessment_results is derived OUTPUT data (what the model said)
  Keeping them separate means that if we ever retrain the model on
  improved logic, we can re-run predictions on old responses without
  losing or overwriting the original input data.

CONNECTION:
  The database connection string is read from the DATABASE_URL
  environment variable. On Render, this is automatically provided
  once you connect a PostgreSQL database to your web service.
  For local development, set DATABASE_URL in a .env file or your
  terminal environment (see documentation/SETUP_GUIDE.md).
----------------------------------------------------------------------
"""

import os
import psycopg2
import psycopg2.extras


def get_database_url():
    """
    Reads the PostgreSQL connection string from the DATABASE_URL
    environment variable. Render provides this automatically when a
    PostgreSQL database is linked to a web service.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "On Render, link your PostgreSQL database to this web service. "
            "Locally, set DATABASE_URL to your PostgreSQL connection string."
        )
    return url


def get_connection():
    """
    Returns a new PostgreSQL connection.
    cursor_factory=RealDictCursor lets us access columns by name,
    e.g. row["username"] instead of row[1], which mirrors the same
    row["column"] access pattern used throughout the rest of the app.
    """
    conn = psycopg2.connect(
        get_database_url(),
        cursor_factory=psycopg2.extras.RealDictCursor
    )
    return conn


def init_db():
    """
    Creates all tables if they do not already exist.
    Safe to call every time the app starts -- it will never
    overwrite existing data.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            SERIAL PRIMARY KEY,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',
            created_at    TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS survey_responses (
            id           SERIAL PRIMARY KEY,
            user_id      INTEGER NOT NULL REFERENCES users (id),
            answers_json TEXT NOT NULL,
            submitted_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS assessment_results (
            id                  SERIAL PRIMARY KEY,
            response_id         INTEGER NOT NULL REFERENCES survey_responses (id),
            user_id             INTEGER NOT NULL REFERENCES users (id),
            predicted_condition TEXT NOT NULL,
            confidence_percent  REAL NOT NULL,
            probability_json    TEXT NOT NULL,
            created_at          TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialised (PostgreSQL).")


if __name__ == "__main__":
    init_db()
