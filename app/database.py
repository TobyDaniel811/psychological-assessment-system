"""
app/database.py
----------------------------------------------------------------------
Defines and creates the SQLite schema for the Psychological Assessment
System, and provides small helper functions for getting a connection.

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
----------------------------------------------------------------------
"""

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "database.db")


def get_connection():
    """
    Returns a new SQLite connection.
    row_factory = sqlite3.Row lets us access columns by name,
    e.g. row["username"] instead of row[1], which is far less
    error-prone when tables grow.

    timeout=10 tells SQLite to wait up to 10 seconds for a lock to
    clear instead of immediately raising "database is locked". This
    matters especially on Windows, where Flask's debug reloader can
    briefly run two processes at once.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
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
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',   -- 'user' or 'admin'
            created_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS survey_responses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            answers_json TEXT NOT NULL,   -- the 10 raw text answers, stored as JSON
            submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS assessment_results (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            response_id         INTEGER NOT NULL,
            user_id             INTEGER NOT NULL,
            predicted_condition TEXT NOT NULL,
            confidence_percent  REAL NOT NULL,
            probability_json    TEXT NOT NULL,   -- full breakdown, stored as JSON
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (response_id) REFERENCES survey_responses (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialised at: {DB_PATH}")


if __name__ == "__main__":
    init_db()
