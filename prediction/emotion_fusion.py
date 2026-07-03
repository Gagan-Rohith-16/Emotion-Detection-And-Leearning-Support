"""Confidence-aware fusion of BiLSTM, BERT, and keyword-rule scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from models.bilstm import EMOTION_LABELS

from .keyword_rules import RuleResult


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Define ensemble weights and mixed-emotion decision thresholds."""

    bilstm_weight: float = 0.35
    bert_weight: float = 0.45
    rule_weight: float = 0.20
    mixed_min_score: float = 0.20
    mixed_max_gap: float = 0.22
    confidence_floor: float = 0.05

    def validate(self) -> None:
        """Raise ValueError when weights or thresholds are out of range."""

        weights = (self.bilstm_weight, self.bert_weight, self.rule_weight)
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("Fusion weights must be non-negative with a positive sum.")
        if any(
            not 0.0 <= value <= 1.0
            for value in (self.mixed_min_score, self.mixed_max_gap, self.confidence_floor)
        ):
            raise ValueError("Fusion thresholds must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Represent the final explainable emotion decision from all available signals."""

    primary_emotion: str
    secondary_emotion: str | None
    confidence: float
    secondary_confidence: float
    scores: dict[str, float]
    is_mixed: bool
    model_used: str
    model_agreement: float
    component_weights: dict[str, float]
    rule_evidence: tuple[Any, ...] = field(default_factory=tuple)


class EmotionFusionEngine:
    """Fuse normalized emotion distributions with confidence-aware weights."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        """Create an ensemble engine with validated configuration."""

        self.config = config or FusionConfig()
        self.config.validate()

    def fuse(
        self,
        *,
        bilstm: Any | Mapping[str, float] | None = None,
        bert: Any | Mapping[str, float] | None = None,
        rules: RuleResult | Mapping[str, float] | None = None,
    ) -> FusionResult:
        """Combine any available model signals into one ranked emotion result."""

        components: list[tuple[str, dict[str, float], float]] = []
        if bilstm is not None:
            scores = self._extract_scores(bilstm)
            components.append(
                ("BiLSTM", scores, self.config.bilstm_weight * self._reliability(scores))
            )
        if bert is not None:
            scores = self._extract_scores(bert)
            components.append(
                ("BERT", scores, self.config.bert_weight * self._reliability(scores))
            )
        if rules is not None:
            scores = self._extract_scores(rules)
            if sum(scores.values()) > 0.0:
                components.append(("Keyword Rules", scores, self.config.rule_weight))
        if not components:
            raise ValueError("At least one usable emotion signal is required for fusion.")

        total_weight = sum(weight for _, _, weight in components)
        if total_weight <= 0.0:
            raise ValueError("Available emotion signals have no usable confidence.")
        effective_weights = {
            name: weight / total_weight for name, _, weight in components
        }
        fused = {emotion: 0.0 for emotion in EMOTION_LABELS}
        for name, scores, _ in components:
            for emotion in EMOTION_LABELS:
                fused[emotion] += effective_weights[name] * scores[emotion]
        fused = self._normalize(fused)

        ranking = sorted(EMOTION_LABELS, key=fused.get, reverse=True)
        primary, secondary = ranking[0], ranking[1]
        primary_score, secondary_score = fused[primary], fused[secondary]
        is_mixed = (
            secondary_score >= self.config.mixed_min_score
            and primary_score - secondary_score <= self.config.mixed_max_gap
        )
        agreement = self._calculate_agreement([scores for _, scores, _ in components])
        evidence = rules.evidence if isinstance(rules, RuleResult) else ()
        return FusionResult(
            primary_emotion=primary,
            secondary_emotion=secondary if is_mixed else None,
            confidence=round(primary_score, 6),
            secondary_confidence=round(secondary_score, 6),
            scores={emotion: round(fused[emotion], 6) for emotion in EMOTION_LABELS},
            is_mixed=is_mixed,
            model_used=" + ".join(name for name, _, _ in components),
            model_agreement=agreement,
            component_weights={
                name: round(weight, 6) for name, weight in effective_weights.items()
            },
            rule_evidence=evidence,
        )

    def _extract_scores(self, prediction: Any | Mapping[str, float]) -> dict[str, float]:
        """Read, validate, and normalize scores from mappings or prediction objects."""

        source = prediction if isinstance(prediction, Mapping) else getattr(prediction, "scores", None)
        if not isinstance(source, Mapping):
            raise TypeError("Prediction signals must expose an emotion score mapping.")
        unknown = set(source).difference(EMOTION_LABELS)
        if unknown:
            raise ValueError(f"Unsupported emotions in score mapping: {', '.join(sorted(unknown))}")
        values: dict[str, float] = {}
        for emotion in EMOTION_LABELS:
            value = float(source.get(emotion, 0.0))
            if value < 0.0 or value != value or value == float("inf"):
                raise ValueError("Emotion scores must be finite and non-negative.")
            values[emotion] = value
        if sum(values.values()) <= 0.0:
            return values
        return self._normalize(values)

    def _reliability(self, scores: dict[str, float]) -> float:
        """Estimate reliability from the margin between the top two probabilities."""

        ranked = sorted(scores.values(), reverse=True)
        if not ranked or ranked[0] <= 0.0:
            return self.config.confidence_floor
        margin = ranked[0] - ranked[1]
        # Top confidence dominates while margin rewards decisive distributions.
        reliability = 0.7 * ranked[0] + 0.3 * margin
        return max(self.config.confidence_floor, min(1.0, reliability))

    @staticmethod
    def _normalize(scores: Mapping[str, float]) -> dict[str, float]:
        """Normalize non-negative emotion values to sum to exactly one."""

        total = sum(max(0.0, float(scores.get(emotion, 0.0))) for emotion in EMOTION_LABELS)
        if total <= 0.0:
            return {emotion: 0.0 for emotion in EMOTION_LABELS}
        return {
            emotion: max(0.0, float(scores.get(emotion, 0.0))) / total
            for emotion in EMOTION_LABELS
        }

    @staticmethod
    def _calculate_agreement(components: list[dict[str, float]]) -> float:
        """Return the proportion of component pairs sharing the same top emotion."""

        if len(components) <= 1:
            return 1.0
        winners = [max(EMOTION_LABELS, key=scores.get) for scores in components]
        matches = 0
        pairs = 0
        for left in range(len(winners)):
            for right in range(left + 1, len(winners)):
                pairs += 1
                matches += int(winners[left] == winners[right])
        return round(matches / pairs, 6)
