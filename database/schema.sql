-- SQLite schema for authentication and emotion-prediction history.
-- Password contains a one-way scrypt hash; plaintext passwords are never stored.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL CHECK (length(trim(name)) BETWEEN 2 AND 100),
    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'student' CHECK (role IN ('student', 'admin')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_used_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id
    ON auth_tokens(user_id);

CREATE TABLE IF NOT EXISTS emotion_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    input_text TEXT NOT NULL CHECK (length(trim(input_text)) > 0),
    field TEXT NOT NULL DEFAULT 'General',
    predicted_emotion TEXT NOT NULL CHECK (
        predicted_emotion IN ('Bored', 'Confident', 'Confused', 'Curious', 'Frustrated')
    ),
    secondary_emotion TEXT CHECK (
        secondary_emotion IS NULL OR
        secondary_emotion IN ('Bored', 'Confident', 'Confused', 'Curious', 'Frustrated')
    ),
    confidence_score REAL NOT NULL CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    model_used TEXT NOT NULL,
    ai_response TEXT NOT NULL DEFAULT '',
    response_type TEXT NOT NULL DEFAULT 'personalized_guidance',
    emotion_scores TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    csv_logged INTEGER NOT NULL DEFAULT 0 CHECK (csv_logged IN (0, 1)),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_emotion_records_user_time
    ON emotion_records(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_emotion_records_emotion
    ON emotion_records(predicted_emotion);
CREATE INDEX IF NOT EXISTS idx_emotion_records_field
    ON emotion_records(field);

