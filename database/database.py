"""Secure, parameterized SQLite access for users and emotion records."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from .models import EmotionRecord, User


SUPPORTED_EMOTIONS = frozenset(
    {"Bored", "Confident", "Confused", "Curious", "Frustrated"}
)
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class DatabaseManager:
    """Manage the application's SQLite schema and parameterized queries."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        """Initialize the manager and create missing tables automatically."""

        default_path = Path(__file__).resolve().parent / "emotion_support.db"
        self.database_path = Path(database_path or default_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_database()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection and safely commit or roll back changes."""

        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize_database(self) -> None:
        """Apply the idempotent schema bundled with this package."""

        schema_path = Path(__file__).with_name("schema.sql")
        with self.connection() as connection:
            connection.executescript(schema_path.read_text(encoding="utf-8"))

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password with scrypt and a unique cryptographic salt."""

        if len(password) < 8:
            raise ValueError("Password must contain at least 8 characters.")
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
        )
        return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"

    @staticmethod
    def verify_password(password: str, encoded_hash: str) -> bool:
        """Verify a password in constant time and reject malformed hashes."""

        try:
            algorithm, n, r, p, salt_hex, digest_hex = encoded_hash.split("$")
            if algorithm != "scrypt":
                return False
            candidate = hashlib.scrypt(
                password.encode("utf-8"),
                salt=bytes.fromhex(salt_hex),
                n=int(n),
                r=int(r),
                p=int(p),
                dklen=len(bytes.fromhex(digest_hex)),
            )
            return hmac.compare_digest(candidate.hex(), digest_hex)
        except (TypeError, ValueError):
            return False

    def register_user(
        self, name: str, email: str, password: str, role: str = "student"
    ) -> User:
        """Validate and create a user, raising ValueError for invalid input."""

        clean_name = name.strip()
        clean_email = email.strip().lower()
        if not 2 <= len(clean_name) <= 100:
            raise ValueError("Name must contain between 2 and 100 characters.")
        if not EMAIL_PATTERN.fullmatch(clean_email):
            raise ValueError("Enter a valid email address.")
        if role not in {"student", "admin"}:
            raise ValueError("Role must be either 'student' or 'admin'.")

        try:
            with self.connection() as connection:
                cursor = connection.execute(
                    "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                    (clean_name, clean_email, self.hash_password(password), role),
                )
                row = connection.execute(
                    "SELECT user_id, name, email, role, created_at FROM users WHERE user_id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
        except sqlite3.IntegrityError as error:
            if "email" in str(error).lower() or "unique" in str(error).lower():
                raise ValueError("An account with this email already exists.") from error
            raise ValueError("The user could not be registered.") from error
        return self._row_to_user(row)

    def authenticate_user(self, email: str, password: str) -> User | None:
        """Return a safe user object when credentials match, otherwise None."""

        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip(),)
            ).fetchone()
        if row is None or not self.verify_password(password, row["password"]):
            return None
        return self._row_to_user(row)

    @staticmethod
    def _hash_auth_token(token: str) -> str:
        """Return a stable hash for browser-persisted auth tokens."""

        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def issue_auth_token(self, user_id: int) -> str:
        """Create a browser-safe token that can restore the current login after refresh."""

        token = secrets.token_urlsafe(32)
        token_hash = self._hash_auth_token(token)
        with self.connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO auth_tokens (token_hash, user_id) VALUES (?, ?)",
                (token_hash, user_id),
            )
        return token

    def get_user_by_auth_token(self, token: str) -> User | None:
        """Resolve a browser token to a user without exposing account data."""

        token_hash = self._hash_auth_token(token.strip())
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT user_id
                FROM auth_tokens
                WHERE token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE auth_tokens
                SET last_used_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE token_hash = ?
                """,
                (token_hash,),
            )
        return self.get_user(int(row["user_id"]))

    def revoke_auth_token(self, token: str) -> None:
        """Delete a browser token during sign-out or invalid-session cleanup."""

        token_hash = self._hash_auth_token(token.strip())
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM auth_tokens WHERE token_hash = ?",
                (token_hash,),
            )

    def get_user(self, user_id: int) -> User | None:
        """Find a user by primary key without returning password material."""

        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT
                    user_id,
                    name,
                    email,
                    role,
                    created_at
                FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def add_emotion_record(
        self,
        *,
        user_id: int,
        input_text: str,
        field: str,
        predicted_emotion: str,
        secondary_emotion: str | None,
        confidence_score: float,
        model_used: str,
        ai_response: str,
        response_type: str = "personalized_guidance",
        emotion_scores: Mapping[str, float] | None = None,
    ) -> EmotionRecord:
        """Persist one validated emotion result and return its stored form."""

        if predicted_emotion not in SUPPORTED_EMOTIONS:
            raise ValueError("Unsupported primary emotion.")
        if secondary_emotion is not None and secondary_emotion not in SUPPORTED_EMOTIONS:
            raise ValueError("Unsupported secondary emotion.")
        if not 0.0 <= float(confidence_score) <= 1.0:
            raise ValueError("Confidence score must be between 0 and 1.")
        if not input_text.strip():
            raise ValueError("Input text cannot be empty.")

        scores = {str(key): float(value) for key, value in (emotion_scores or {}).items()}
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO emotion_records (
                    user_id, input_text, field, predicted_emotion, secondary_emotion,
                    confidence_score, model_used, ai_response, response_type, emotion_scores
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id, input_text.strip(), field.strip() or "General",
                    predicted_emotion, secondary_emotion, float(confidence_score),
                    model_used.strip(), ai_response, response_type, json.dumps(scores),
                ),
            )
            row = connection.execute(
                "SELECT * FROM emotion_records WHERE record_id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return self._row_to_record(row)

    def get_emotion_history(
        self,
        user_id: int,
        *,
        search: str = "",
        emotion: str | None = None,
        field: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EmotionRecord]:
        """Return filtered prediction history ordered from newest to oldest."""

        if limit < 1 or limit > 1000 or offset < 0:
            raise ValueError("Invalid pagination values.")
        clauses = ["user_id = ?"]
        parameters: list[object] = [user_id]
        if search.strip():
            clauses.append("(input_text LIKE ? ESCAPE '\\' OR ai_response LIKE ? ESCAPE '\\')")
            escaped = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            parameters.extend([f"%{escaped}%", f"%{escaped}%"])
        if emotion:
            if emotion not in SUPPORTED_EMOTIONS:
                raise ValueError("Unsupported emotion filter.")
            clauses.append("predicted_emotion = ?")
            parameters.append(emotion)
        if field:
            clauses.append("field = ?")
            parameters.append(field)
        parameters.extend([limit, offset])
        query = f"""
            SELECT * FROM emotion_records
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp DESC, record_id DESC
            LIMIT ? OFFSET ?
        """
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._row_to_record(row) for row in rows]

    def mark_csv_logged(self, record_ids: list[int], logged: bool = True) -> int:
        """Update CSV export status and return the number of changed records."""

        if not record_ids:
            return 0
        placeholders = ",".join("?" for _ in record_ids)
        with self.connection() as connection:
            cursor = connection.execute(
                f"UPDATE emotion_records SET csv_logged = ? WHERE record_id IN ({placeholders})",
                [int(logged), *record_ids],
            )
        return cursor.rowcount

    def delete_emotion_record(self, record_id: int, user_id: int) -> bool:
        """Delete a record only when it belongs to the requesting user."""

        with self.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM emotion_records WHERE record_id = ? AND user_id = ?",
                (record_id, user_id),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        """Convert a SQLite row into a User object."""

        return User(
            user_id=row["user_id"],
            name=row["name"],
            email=row["email"],
            role=row["role"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EmotionRecord:
        """Convert a SQLite row into a typed EmotionRecord object."""

        try:
            scores = json.loads(row["emotion_scores"])
        except (TypeError, json.JSONDecodeError):
            scores = {}
        return EmotionRecord(
            record_id=row["record_id"], user_id=row["user_id"],
            input_text=row["input_text"], field=row["field"],
            predicted_emotion=row["predicted_emotion"],
            secondary_emotion=row["secondary_emotion"],
            confidence_score=float(row["confidence_score"]),
            model_used=row["model_used"], ai_response=row["ai_response"],
            response_type=row["response_type"], emotion_scores=scores,
            timestamp=row["timestamp"], csv_logged=bool(row["csv_logged"]),
        )
    def update_profile(
        self,
        user_id: int,
        name: str,
    ) -> None:
        """Update user's profile."""

        clean_name = name.strip()
        if not 2 <= len(clean_name) <= 100:
            raise ValueError("Name must contain between 2 and 100 characters.")

        with self.connection() as connection:
            connection.execute(
                """
                UPDATE users
                SET name = ?
                WHERE user_id = ?
                """,
                (
                    clean_name,
                    user_id,
                ),
            )
