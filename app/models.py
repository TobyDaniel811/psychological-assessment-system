"""
app/models.py
----------------------------------------------------------------------
User-related database operations and the Flask-Login User class.

Passwords are NEVER stored as plain text. We use Werkzeug's
generate_password_hash() / check_password_hash(), which applies a
salted hash (PBKDF2 by default). Even if someone got direct access
to database.db, they could not read anyone's actual password.
----------------------------------------------------------------------
"""

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
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return User(row["id"], row["username"], row["email"],
                     row["password_hash"], row["role"])

    @staticmethod
    def get_by_username(username):
        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return User(row["id"], row["username"], row["email"],
                     row["password_hash"], row["role"])

    @staticmethod
    def create(username, email, password, role="user"):
        """
        Hashes the password and inserts a new user row.
        Raises sqlite3.IntegrityError if username/email already exists
        (the UNIQUE constraints in the schema enforce this).
        """
        password_hash = generate_password_hash(password)
        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, role) "
            "VALUES (?, ?, ?, ?)",
            (username, email, password_hash, role)
        )
        conn.commit()
        new_id = cur.lastrowid
        conn.close()
        return new_id

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
