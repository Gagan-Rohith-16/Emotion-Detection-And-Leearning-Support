"""Model-aware preprocessing for student learning-problem descriptions.

BERT performs best with natural sentence structure, while a custom BiLSTM often
benefits from more regularized tokens. This module produces both representations
from one validated input without discarding emotionally useful cues.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass


# Negation words are intentionally retained because they can reverse sentiment.
NEGATIONS = frozenset({"no", "nor", "not", "never", "none", "neither", "cannot"})

# A compact stopword set avoids a runtime corpus download and remains reproducible.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
        "for", "from", "has", "have", "he", "her", "hers", "him", "his", "i",
        "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours",
        "she", "that", "the", "their", "theirs", "them", "they", "this", "those",
        "to", "was", "we", "were", "what", "when", "where", "which", "who", "will",
        "with", "you", "your", "yours",
    }
)

# Expanding frequent contractions makes negation explicit for both model families.
CONTRACTIONS = {
    "can't": "cannot", "cant": "cannot", "won't": "will not", "wont": "will not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "i'm": "i am",
    "i've": "i have", "i'll": "i will", "i'd": "i would", "it's": "it is",
    "that's": "that is", "there's": "there is", "they're": "they are",
    "we're": "we are", "you're": "you are", "i’ll": "i will", "i’m": "i am",
    "don’t": "do not", "can’t": "cannot", "won’t": "will not",
}

# Emoji aliases preserve strong affective signals that punctuation cleanup would lose.
EMOJI_ALIASES = {
    "😕": " confused ", "😟": " worried ", "🤔": " thinking curious ",
    "😤": " frustrated ", "😡": " angry frustrated ", "😭": " crying frustrated ",
    "😢": " sad ", "😴": " bored tired ", "🥱": " bored yawning ",
    "😊": " happy confident ", "😎": " confident ", "💪": " confident ",
    "💡": " curious idea ", "✨": " excited curious ", "❓": " question confused ",
    "✅": " success confident ", "❤️": " positive ", "❤": " positive ",
}

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|<url>|<email>|\d+(?:\.\d+)?", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?])\1{2,}")
REPEATED_CHARACTER_PATTERN = re.compile(r"([a-zA-Z])\1{3,}")


@dataclass(frozen=True, slots=True)
class PreprocessedText:
    """Contain traceable text variants for downstream models and analytics."""

    original_text: str
    normalized_text: str
    bert_text: str
    bilstm_text: str
    tokens: tuple[str, ...]
    word_count: int
    character_count: int

    def is_empty(self) -> bool:
        """Return whether normalization produced no usable language tokens."""

        return not self.tokens


class TextPreprocessor:
    """Create consistent BERT, BiLSTM, and keyword-rule input variants."""

    def __init__(
        self,
        *,
        min_characters: int = 3,
        max_characters: int = 5000,
        remove_stopwords_for_bilstm: bool = False,
    ) -> None:
        """Configure input limits and optional BiLSTM stopword removal."""

        if min_characters < 1 or max_characters < min_characters:
            raise ValueError("Text length limits are invalid.")
        self.min_characters = min_characters
        self.max_characters = max_characters
        self.remove_stopwords_for_bilstm = remove_stopwords_for_bilstm

    def preprocess(self, text: str) -> PreprocessedText:
        """Validate one input and produce model-specific text representations."""

        original = self._validate_input(text)
        normalized = self._normalize_shared(original)
        bert_text = self._prepare_for_bert(normalized)
        all_tokens = tuple(token.lower() for token in TOKEN_PATTERN.findall(normalized))
        bilstm_tokens = self._prepare_bilstm_tokens(all_tokens)
        if not bilstm_tokens:
            raise ValueError("Text must contain at least one word or number.")
        return PreprocessedText(
            original_text=original,
            normalized_text=normalized,
            bert_text=bert_text,
            bilstm_text=" ".join(bilstm_tokens),
            tokens=bilstm_tokens,
            word_count=len(all_tokens),
            character_count=len(original),
        )

    def preprocess_batch(self, texts: list[str] | tuple[str, ...]) -> list[PreprocessedText]:
        """Preprocess an ordered collection while preserving input order."""

        if not isinstance(texts, (list, tuple)):
            raise TypeError("Batch input must be a list or tuple of strings.")
        return [self.preprocess(text) for text in texts]

    def _validate_input(self, text: str) -> str:
        """Reject non-string, blank, undersized, and oversized user input."""

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        cleaned = SPACE_PATTERN.sub(" ", text.replace("\x00", " ")).strip()
        if len(cleaned) < self.min_characters:
            raise ValueError(f"Please enter at least {self.min_characters} characters.")
        if len(cleaned) > self.max_characters:
            raise ValueError(f"Input cannot exceed {self.max_characters} characters.")
        return cleaned

    @staticmethod
    def _normalize_shared(text: str) -> str:
        """Normalize Unicode, markup, links, contractions, emojis, and spacing."""

        normalized = unicodedata.normalize("NFKC", html.unescape(text))
        normalized = URL_PATTERN.sub(" <url> ", normalized)
        normalized = EMAIL_PATTERN.sub(" <email> ", normalized)
        for emoji, alias in EMOJI_ALIASES.items():
            normalized = normalized.replace(emoji, alias)

        # Replace longer contraction keys first to avoid partial substitutions.
        for contraction in sorted(CONTRACTIONS, key=len, reverse=True):
            replacement = CONTRACTIONS[contraction]
            normalized = re.sub(
                rf"(?<!\w){re.escape(contraction)}(?!\w)",
                replacement,
                normalized,
                flags=re.IGNORECASE,
            )
        normalized = REPEATED_PUNCTUATION_PATTERN.sub(r"\1\1", normalized)
        # "soooo" becomes "soo": emphasis remains without exploding vocabulary.
        normalized = REPEATED_CHARACTER_PATTERN.sub(r"\1\1", normalized)
        return SPACE_PATTERN.sub(" ", normalized).strip()

    @staticmethod
    def _prepare_for_bert(text: str) -> str:
        """Retain natural casing and punctuation while normalizing whitespace."""

        # Control characters can upset tokenizers and are never useful model input.
        printable = "".join(character for character in text if character.isprintable())
        return SPACE_PATTERN.sub(" ", printable).strip()

    def _prepare_bilstm_tokens(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        """Optionally remove low-information stopwords while preserving negation."""

        if not self.remove_stopwords_for_bilstm:
            return tokens
        return tuple(
            token for token in tokens if token not in STOPWORDS or token in NEGATIONS
        )


def preprocess_text(text: str) -> PreprocessedText:
    """Convenience wrapper using the platform's default preprocessing policy."""

    return TextPreprocessor().preprocess(text)
