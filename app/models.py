"""
app/models.py
----------------------------------------------------------------------
User-related database operations and the Flask-Login User class.

Passwords are NEVER stored as plain text. We use Werkzeug's
generate_password_hash() / check_password_hash(), which applies a
salted hash (PBKDF2 by default). Even if someone got direct access
to the database, they could not read anyone's actual password.
----------------------------------------------------------------------
"""

import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app.database import get_connection


class User(UserMixin):
    """
    Wraps a row from the `users` table so Flask-Login can manage
    sessions for it. Flask-Login requires a get_id() method, which
    UserMixin provides automatically using self.id.
    """
    def __init__(self, id, username, email, password_hash, role):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role

    @staticmethod
    def get_by_id(user_id):
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row is None:
            return None
        return User(row["id"], row["username"], row["email"],
                     row["password_hash"], row["role"])

    @staticmethod
    def get_by_username(username):
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row is None:
            return None
        return User(row["id"], row["username"], row["email"],
                     row["password_hash"], row["role"])

    @staticmethod
    def create(username, email, password, role="user"):
        """
        Hashes the password and inserts a new user row.
        Raises psycopg2.errors.UniqueViolation if username/email
        already exists (the UNIQUE constraints in the schema enforce
        this).
        """
        password_hash = generate_password_hash(password)
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password_hash, role) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (username, email, password_hash, role)
        )
        new_id = cur.fetchone()["id"]
        conn.commit()
        cur.close()
        conn.close()
        return new_id

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
