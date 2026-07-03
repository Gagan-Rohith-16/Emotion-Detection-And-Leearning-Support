"""Trainable BiLSTM emotion classifier with persistent inference artifacts.

TensorFlow is imported only when the model is built, trained, or loaded. This
keeps lightweight application tasks usable before optional ML dependencies are
installed and produces a clear setup message instead of an import-time crash.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing import TextPreprocessor


EMOTION_LABELS: tuple[str, ...] = (
    "Bored",
    "Confident",
    "Confused",
    "Curious",
    "Frustrated",
)


class MLDependencyError(RuntimeError):
    """Indicate that an optional machine-learning dependency is unavailable."""


class ModelNotReadyError(RuntimeError):
    """Indicate that trained artifacts do not yet exist or are incomplete."""


@dataclass(frozen=True, slots=True)
class BiLSTMConfig:
    """Define reproducible architecture, tokenization, and training parameters."""

    vocabulary_size: int = 20_000
    sequence_length: int = 120
    embedding_dimension: int = 128
    lstm_units: int = 64
    dense_units: int = 64
    dropout_rate: float = 0.35
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 20
    validation_split: float = 0.2
    random_seed: int = 42

    def validate(self) -> None:
        """Raise ValueError when a configuration cannot form a valid model."""

        if self.vocabulary_size < 100 or self.sequence_length < 4:
            raise ValueError("Vocabulary and sequence sizes are too small.")
        if min(self.embedding_dimension, self.lstm_units, self.dense_units) < 1:
            raise ValueError("Layer dimensions must be positive.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("Dropout rate must be in the range [0, 1).")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("Training parameters must be positive.")
        if not 0.0 < self.validation_split < 0.5:
            raise ValueError("Validation split must be between 0 and 0.5.")


@dataclass(frozen=True, slots=True)
class BiLSTMPrediction:
    """Represent ranked BiLSTM probabilities for one student description."""

    primary_emotion: str
    secondary_emotion: str
    confidence: float
    secondary_confidence: float
    scores: dict[str, float]
    is_mixed: bool
    model_name: str = "BiLSTM"


class BiLSTMEmotionClassifier:
    """Build, train, persist, load, and serve the BiLSTM emotion model."""

    MODEL_FILENAME = "emotion_bilstm.keras"
    TOKENIZER_FILENAME = "tokenizer.json"
    METADATA_FILENAME = "metadata.json"

    def __init__(
        self,
        artifact_directory: str | Path | None = None,
        *,
        config: BiLSTMConfig | None = None,
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        """Configure artifact paths without loading heavyweight dependencies."""

        default_directory = Path(__file__).resolve().parent / "artifacts" / "bilstm"
        self.artifact_directory = Path(artifact_directory or default_directory).resolve()
        self.config = config or BiLSTMConfig()
        self.config.validate()
        self.preprocessor = preprocessor or TextPreprocessor()
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def model_path(self) -> Path:
        """Return the expected Keras model artifact location."""

        return self.artifact_directory / self.MODEL_FILENAME

    @property
    def tokenizer_path(self) -> Path:
        """Return the expected tokenizer artifact location."""

        return self.artifact_directory / self.TOKENIZER_FILENAME

    @property
    def metadata_path(self) -> Path:
        """Return the expected model metadata location."""

        return self.artifact_directory / self.METADATA_FILENAME

    def is_ready(self) -> bool:
        """Return whether all required inference artifacts are present."""

        return all(
            path.is_file()
            for path in (self.model_path, self.tokenizer_path, self.metadata_path)
        )

    @staticmethod
    def _tensorflow() -> Any:
        """Load TensorFlow lazily and provide an actionable dependency error."""

        try:
            # TensorFlow logging is reduced before import to keep app logs readable.
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise MLDependencyError(
                "TensorFlow is required for BiLSTM training and inference. "
                "Install the project dependencies with: pip install -r requirements.txt"
            ) from error
        return tf

    def _set_random_seeds(self, tf: Any) -> None:
        """Seed Python, NumPy, and TensorFlow for reproducible experiments."""

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        tf.keras.utils.set_random_seed(self.config.random_seed)

    def build_model(self) -> Any:
        """Build and compile the bidirectional LSTM architecture."""

        tf = self._tensorflow()
        self._set_random_seeds(tf)
        inputs = tf.keras.Input(shape=(self.config.sequence_length,), name="token_ids")
        values = tf.keras.layers.Embedding(
            input_dim=self.config.vocabulary_size,
            output_dim=self.config.embedding_dimension,
            mask_zero=True,
            name="token_embedding",
        )(inputs)
        values = tf.keras.layers.SpatialDropout1D(self.config.dropout_rate)(values)
        values = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.config.lstm_units, return_sequences=True),
            name="context_bilstm",
        )(values)
        values = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(max(16, self.config.lstm_units // 2)),
            name="summary_bilstm",
        )(values)
        values = tf.keras.layers.Dense(
            self.config.dense_units, activation="relu", name="emotion_features"
        )(values)
        values = tf.keras.layers.Dropout(self.config.dropout_rate)(values)
        outputs = tf.keras.layers.Dense(
            len(EMOTION_LABELS), activation="softmax", name="emotion_probabilities"
        )(values)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="emotion_bilstm")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self._model = model
        return model

    def _create_tokenizer(self, tf: Any) -> Any:
        """Create the capped tokenizer used consistently in training and serving."""

        return tf.keras.preprocessing.text.Tokenizer(
            num_words=self.config.vocabulary_size,
            oov_token="<OOV>",
            filters='"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
            lower=True,
        )

    def _encode_texts(self, texts: list[str], tf: Any) -> np.ndarray:
        """Convert preprocessed strings into padded integer sequences."""

        if self._tokenizer is None:
            raise ModelNotReadyError("The BiLSTM tokenizer has not been initialized.")
        sequences = self._tokenizer.texts_to_sequences(texts)
        return np.asarray(
            tf.keras.utils.pad_sequences(
                sequences,
                maxlen=self.config.sequence_length,
                padding="post",
                truncating="post",
            ),
            dtype=np.int32,
        )

    def train(
        self,
        texts: list[str],
        labels: list[str],
        *,
        class_weight: dict[int, float] | None = None,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """Train on labeled text, save best artifacts, and return metric history."""

        if len(texts) != len(labels) or len(texts) < 10:
            raise ValueError("Training requires at least 10 aligned texts and labels.")
        unknown_labels = sorted(set(labels).difference(EMOTION_LABELS))
        if unknown_labels:
            raise ValueError(f"Unsupported training labels: {', '.join(unknown_labels)}")

        tf = self._tensorflow()
        self._set_random_seeds(tf)
        prepared = [self.preprocessor.preprocess(text).bilstm_text for text in texts]
        targets = np.asarray([EMOTION_LABELS.index(label) for label in labels], dtype=np.int32)
        self._tokenizer = self._create_tokenizer(tf)
        self._tokenizer.fit_on_texts(prepared)
        features = self._encode_texts(prepared, tf)
        model = self.build_model()
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
            ),
        ]
        history = model.fit(
            features,
            targets,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            class_weight=class_weight,
            callbacks=callbacks,
            shuffle=True,
            verbose=verbose,
        )
        self.save()
        return {key: [float(value) for value in values] for key, values in history.history.items()}

    def save(self) -> None:
        """Atomically persist the model, tokenizer, labels, and configuration."""

        if self._model is None or self._tokenizer is None:
            raise ModelNotReadyError("Train or load the BiLSTM before saving it.")
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        self._model.save(self.model_path)
        self.tokenizer_path.write_text(self._tokenizer.to_json(), encoding="utf-8")
        metadata = {
            "format_version": 1,
            "model_type": "BiLSTM",
            "labels": list(EMOTION_LABELS),
            "config": asdict(self.config),
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def load(self) -> None:
        """Load and validate persisted artifacts for inference."""

        if not self.is_ready():
            raise ModelNotReadyError(
                f"BiLSTM artifacts are missing from {self.artifact_directory}. Train the model first."
            )
        tf = self._tensorflow()
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("labels", ())) != EMOTION_LABELS:
            raise ModelNotReadyError("BiLSTM label metadata is incompatible.")
        saved_config = BiLSTMConfig(**metadata["config"])
        saved_config.validate()
        self.config = saved_config
        self._tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            self.tokenizer_path.read_text(encoding="utf-8")
        )
        self._model = tf.keras.models.load_model(self.model_path, compile=False)

    def predict(self, text: str, *, mixed_threshold: float = 0.18) -> BiLSTMPrediction:
        """Predict ranked emotions and identify a plausible mixed-emotion result."""

        if not 0.0 <= mixed_threshold <= 1.0:
            raise ValueError("Mixed-emotion threshold must be between 0 and 1.")
        if self._model is None or self._tokenizer is None:
            self.load()
        tf = self._tensorflow()
        prepared = self.preprocessor.preprocess(text).bilstm_text
        features = self._encode_texts([prepared], tf)
        probabilities = np.asarray(self._model.predict(features, verbose=0)[0], dtype=float)
        if probabilities.shape != (len(EMOTION_LABELS),) or not np.isfinite(probabilities).all():
            raise RuntimeError("BiLSTM returned an invalid probability vector.")
        ranking = np.argsort(probabilities)[::-1]
        primary_index, secondary_index = int(ranking[0]), int(ranking[1])
        primary_score = float(probabilities[primary_index])
        secondary_score = float(probabilities[secondary_index])
        scores = {
            label: round(float(probabilities[index]), 6)
            for index, label in enumerate(EMOTION_LABELS)
        }
        return BiLSTMPrediction(
            primary_emotion=EMOTION_LABELS[primary_index],
            secondary_emotion=EMOTION_LABELS[secondary_index],
            confidence=primary_score,
            secondary_confidence=secondary_score,
            scores=scores,
            is_mixed=secondary_score >= mixed_threshold,
        )
