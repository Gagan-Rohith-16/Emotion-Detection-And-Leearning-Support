"""Typed value objects returned by the SQLite data-access layer."""

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class User:
    user_id: int
    name: str
    email: str
    role: str
    created_at: str


@dataclass(frozen=True, slots=True)
class EmotionRecord:
    """Represent one stored emotion prediction and its generated guidance."""

    record_id: int
    user_id: int
    input_text: str
    field: str
    predicted_emotion: str
    secondary_emotion: str | None
    confidence_score: float
    model_used: str
    ai_response: str
    response_type: str
    emotion_scores: dict[str, float]
    timestamp: str
    csv_logged: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly mapping for tables and CSV export."""

        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "input_text": self.input_text,
            "field": self.field,
            "predicted_emotion": self.predicted_emotion,
            "secondary_emotion": self.secondary_emotion,
            "confidence_score": self.confidence_score,
            "model_used": self.model_used,
            "ai_response": self.ai_response,
            "response_type": self.response_type,
            "emotion_scores": self.emotion_scores,
            "timestamp": self.timestamp,
            "csv_logged": self.csv_logged,
        }

