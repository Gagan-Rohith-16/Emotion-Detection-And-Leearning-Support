"""Explainable phrase and keyword enhancement for learning emotions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from models.bilstm import EMOTION_LABELS
from preprocessing import TextPreprocessor


# Multi-word phrases receive more weight because they express clearer intent.
EMOTION_PATTERNS: dict[str, dict[str, float]] = {
    "Bored": {
        "bored": 1.0, "boring": 1.0, "not interested": 1.2, "lost interest": 1.3,
        "same thing": 0.7, "repetitive": 1.0, "dull": 0.9, "tired of": 1.1,
        "cannot focus": 0.8, "falling asleep": 1.2, "yawning": 0.8,
    },
    "Confident": {
        "confident": 1.1, "i understand": 1.2, "makes sense": 1.1, "i can do": 1.1,
        "got it": 1.2, "easy for me": 1.0, "ready": 0.7, "sure": 0.7,
        "success": 0.8, "mastered": 1.3, "comfortable with": 1.0,
    },
    "Confused": {
        "confused": 1.2, "do not understand": 1.3, "cannot understand": 1.3,
        "does not make sense": 1.3, "unclear": 1.0, "lost": 0.9, "stuck": 0.9,
        "what does": 0.7, "how does": 0.6, "why does": 0.6, "mixed up": 1.1,
        "question": 0.5, "cannot follow": 1.2,
    },
    "Curious": {
        "curious": 1.2, "want to learn": 1.1, "want to know": 1.1,
        "interested in": 1.0, "wonder": 0.9, "explore": 0.8, "discover": 0.8,
        "tell me more": 1.0, "fascinating": 1.1, "how can": 0.7, "why is": 0.7,
        "excited to learn": 1.3, "idea": 0.5,
    },
    "Frustrated": {
        "frustrated": 1.3, "annoyed": 1.0, "angry": 1.1, "giving up": 1.3,
        "give up": 1.2, "hate this": 1.3, "impossible": 1.1, "too difficult": 1.1,
        "nothing works": 1.3, "tried everything": 1.2, "again and again": 0.9,
        "waste of time": 1.3, "fed up": 1.3, "crying": 0.9,
    },
}

NEGATION_WORDS = frozenset({"not", "never", "no", "hardly", "barely", "without"})
INTENSIFIERS = frozenset({"very", "really", "extremely", "so", "totally", "completely"})


@dataclass(frozen=True, slots=True)
class RuleEvidence:
    """Describe one matched rule and its signed contribution."""

    emotion: str
    phrase: str
    contribution: float
    negated: bool


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Contain normalized rule scores and human-readable matching evidence."""

    scores: dict[str, float]
    evidence: tuple[RuleEvidence, ...]
    matched: bool
    model_name: str = "Keyword Rules"


class KeywordRuleEngine:
    """Score explicit emotional language while handling nearby negation."""

    def __init__(self, preprocessor: TextPreprocessor | None = None) -> None:
        """Create the rule engine with the shared normalization pipeline."""

        self.preprocessor = preprocessor or TextPreprocessor()

    def analyze(self, text: str) -> RuleResult:
        """Return normalized rule scores and evidence for a student description."""

        normalized = self.preprocessor.preprocess(text).normalized_text.lower()
        raw_scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
        evidence: list[RuleEvidence] = []
        for emotion, patterns in EMOTION_PATTERNS.items():
            for phrase, base_weight in patterns.items():
                for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized):
                    context = normalized[max(0, match.start() - 35):match.start()]
                    context_tokens = re.findall(r"[a-z]+", context)[-3:]
                    negated = any(token in NEGATION_WORDS for token in context_tokens)
                    intensified = any(token in INTENSIFIERS for token in context_tokens)
                    contribution = base_weight * (1.25 if intensified else 1.0)
                    # Negation suppresses rather than reverses a cue; reversal is too brittle.
                    contribution *= 0.15 if negated else 1.0
                    raw_scores[emotion] += contribution
                    evidence.append(
                        RuleEvidence(emotion, phrase, round(contribution, 4), negated)
                    )

        normalized_scores = self._normalize(raw_scores)
        ordered_evidence = tuple(
            sorted(evidence, key=lambda item: item.contribution, reverse=True)
        )
        return RuleResult(
            scores=normalized_scores,
            evidence=ordered_evidence,
            matched=bool(evidence),
        )

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        """Scale rule weights into a probability-like distribution."""

        total = sum(max(0.0, value) for value in scores.values())
        if total <= 0.0:
            return {emotion: 0.0 for emotion in EMOTION_LABELS}
        return {
            emotion: round(max(0.0, scores.get(emotion, 0.0)) / total, 6)
            for emotion in EMOTION_LABELS
        }

