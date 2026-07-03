"""Hugging Face BERT fine-tuning and inference for learning emotions.

The implementation keeps PyTorch and Transformers imports lazy so the Streamlit
shell and non-ML utilities remain usable in lightweight environments.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing import TextPreprocessor

from .bilstm import EMOTION_LABELS, MLDependencyError, ModelNotReadyError


@dataclass(frozen=True, slots=True)
class BERTConfig:
    """Define the pretrained checkpoint and reproducible fine-tuning settings."""

    base_model_name: str = "bert-base-uncased"
    max_length: int = 160
    batch_size: int = 8
    epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    validation_split: float = 0.2
    random_seed: int = 42

    def validate(self) -> None:
        """Raise ValueError when fine-tuning parameters are unsafe or invalid."""

        if not self.base_model_name.strip():
            raise ValueError("A Hugging Face base model name is required.")
        if not 16 <= self.max_length <= 512:
            raise ValueError("BERT maximum length must be between 16 and 512.")
        if self.batch_size < 1 or self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("Batch size, epochs, and learning rate must be positive.")
        if not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("Weight decay must be between 0 and 1.")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("Warmup ratio must be in the range [0, 1).")
        if not 0.0 < self.validation_split < 0.5:
            raise ValueError("Validation split must be between 0 and 0.5.")


@dataclass(frozen=True, slots=True)
class BERTPrediction:
    """Represent ranked probabilities from the fine-tuned BERT classifier."""

    primary_emotion: str
    secondary_emotion: str
    confidence: float
    secondary_confidence: float
    scores: dict[str, float]
    is_mixed: bool
    model_name: str = "BERT"


class _EncodedTextDataset:
    """Minimal PyTorch-compatible dataset without a module-level torch import."""

    def __init__(self, encodings: dict[str, Any], labels: list[int]) -> None:
        """Store tokenizer tensors and their aligned integer labels."""

        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of encoded examples."""

        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one encoded example in the format expected by BERT."""

        item = {key: value[index] for key, value in self.encodings.items()}
        # Tokenizer tensors already establish the PyTorch dependency and dtype.
        item["labels"] = self.labels[index]
        return item


class BERTEmotionClassifier:
    """Fine-tune, persist, load, and serve a five-class BERT model."""

    METADATA_FILENAME = "emotion_metadata.json"

    def __init__(
        self,
        artifact_directory: str | Path | None = None,
        *,
        config: BERTConfig | None = None,
        preprocessor: TextPreprocessor | None = None,
        device: str | None = None,
    ) -> None:
        """Configure BERT while deferring all heavyweight model loading."""

        default_directory = Path(__file__).resolve().parent / "artifacts" / "bert"
        self.artifact_directory = Path(artifact_directory or default_directory).resolve()
        self.config = config or BERTConfig()
        self.config.validate()
        self.preprocessor = preprocessor or TextPreprocessor()
        self.requested_device = device
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: Any | None = None

    @property
    def metadata_path(self) -> Path:
        """Return the BERT label and training-configuration metadata path."""

        return self.artifact_directory / self.METADATA_FILENAME

    def is_ready(self) -> bool:
        """Return whether a complete Hugging Face artifact directory exists."""

        required = (
            self.metadata_path,
            self.artifact_directory / "config.json",
            self.artifact_directory / "tokenizer_config.json",
        )
        has_weights = any(
            (self.artifact_directory / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        return has_weights and all(path.is_file() for path in required)

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any, Any]:
        """Load torch and required Transformers objects with a clear error."""

        try:
            import torch  # pylint: disable=import-outside-toplevel
            from transformers import (  # pylint: disable=import-outside-toplevel
                AutoModelForSequenceClassification,
                AutoTokenizer,
                get_linear_schedule_with_warmup,
            )
        except ImportError as error:
            raise MLDependencyError(
                "PyTorch and Hugging Face Transformers are required for BERT. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from error
        return torch, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    def _resolve_device(self, torch: Any) -> Any:
        """Select an explicitly requested device or the best available default."""

        if self.requested_device:
            requested = torch.device(self.requested_device)
            if requested.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is not available.")
            return requested
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_random_seeds(self, torch: Any) -> None:
        """Seed Python, NumPy, and PyTorch for reproducible fine-tuning."""

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def initialize_pretrained(self) -> None:
        """Download/load the configured base BERT and attach five output labels."""

        torch, tokenizer_class, model_class, _ = self._dependencies()
        self._set_random_seeds(torch)
        id_to_label = {index: label for index, label in enumerate(EMOTION_LABELS)}
        label_to_id = {label: index for index, label in id_to_label.items()}
        self._tokenizer = tokenizer_class.from_pretrained(self.config.base_model_name)
        self._model = model_class.from_pretrained(
            self.config.base_model_name,
            num_labels=len(EMOTION_LABELS),
            id2label=id_to_label,
            label2id=label_to_id,
            problem_type="single_label_classification",
        )
        self._device = self._resolve_device(torch)
        self._model.to(self._device)

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        """Tokenize natural BERT text with truncation and fixed padding."""

        if self._tokenizer is None:
            raise ModelNotReadyError("The BERT tokenizer is not initialized.")
        return self._tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def fine_tune(
        self,
        texts: list[str],
        labels: list[str],
        *,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Fine-tune BERT with a deterministic split and return epoch metrics."""

        if len(texts) != len(labels) or len(texts) < 20:
            raise ValueError("BERT fine-tuning requires at least 20 aligned examples.")
        unknown_labels = sorted(set(labels).difference(EMOTION_LABELS))
        if unknown_labels:
            raise ValueError(f"Unsupported training labels: {', '.join(unknown_labels)}")
        torch, _, _, scheduler_factory = self._dependencies()
        if self._model is None or self._tokenizer is None:
            self.initialize_pretrained()
        self._set_random_seeds(torch)

        prepared = [self.preprocessor.preprocess(text).bert_text for text in texts]
        targets = [EMOTION_LABELS.index(label) for label in labels]
        indices = list(range(len(prepared)))
        random.Random(self.config.random_seed).shuffle(indices)
        validation_size = max(1, round(len(indices) * self.config.validation_split))
        validation_indices, training_indices = indices[:validation_size], indices[validation_size:]
        train_texts = [prepared[index] for index in training_indices]
        train_labels = [targets[index] for index in training_indices]
        validation_texts = [prepared[index] for index in validation_indices]
        validation_labels = [targets[index] for index in validation_indices]

        train_dataset = _EncodedTextDataset(self._encode(train_texts), train_labels)
        validation_dataset = _EncodedTextDataset(self._encode(validation_texts), validation_labels)
        collate = self._collate_batch(torch)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate
        )
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = len(train_loader) * self.config.epochs
        scheduler = scheduler_factory(
            optimizer,
            num_warmup_steps=round(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )
        history: dict[str, list[float]] = {
            "train_loss": [], "validation_loss": [], "validation_accuracy": []
        }
        best_validation_loss = math.inf
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, torch)
            validation_loss, validation_accuracy = self._evaluate(validation_loader, torch)
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            history["validation_accuracy"].append(validation_accuracy)
            if verbose:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} - loss: {train_loss:.4f} "
                    f"- val_loss: {validation_loss:.4f} - val_accuracy: {validation_accuracy:.4f}"
                )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self.save()
        self.load()
        return history

    @staticmethod
    def _collate_batch(torch: Any) -> Any:
        """Create a collator that converts scalar labels into long tensors."""

        def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
            """Stack encoded examples into one device-independent mini-batch."""

            keys = ("input_ids", "attention_mask", "token_type_ids")
            batch = {
                key: torch.stack([example[key] for example in examples])
                for key in keys if key in examples[0]
            }
            batch["labels"] = torch.tensor(
                [example["labels"] for example in examples], dtype=torch.long
            )
            return batch

        return collate

    def _train_epoch(self, loader: Any, optimizer: Any, scheduler: Any, torch: Any) -> float:
        """Run one optimization epoch and return mean training loss."""

        self._model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(self._device) for key, value in batch.items()}
            output = self._model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(output.loss.detach().cpu())
        return total_loss / max(1, len(loader))

    def _evaluate(self, loader: Any, torch: Any) -> tuple[float, float]:
        """Evaluate without gradients and return mean loss and accuracy."""

        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(self._device) for key, value in batch.items()}
                output = self._model(**batch)
                total_loss += float(output.loss.detach().cpu())
                predictions = output.logits.argmax(dim=-1)
                correct += int((predictions == batch["labels"]).sum().detach().cpu())
                total += int(batch["labels"].shape[0])
        return total_loss / max(1, len(loader)), correct / max(1, total)

    def save(self) -> None:
        """Persist the Hugging Face model, tokenizer, labels, and configuration."""

        if self._model is None or self._tokenizer is None:
            raise ModelNotReadyError("Initialize or load BERT before saving it.")
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(self.artifact_directory, safe_serialization=True)
        self._tokenizer.save_pretrained(self.artifact_directory)
        metadata = {
            "format_version": 1,
            "model_type": "BERT",
            "labels": list(EMOTION_LABELS),
            "config": asdict(self.config),
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def load(self) -> None:
        """Load a locally fine-tuned BERT without contacting external services."""

        if not self.is_ready():
            raise ModelNotReadyError(
                f"BERT artifacts are missing from {self.artifact_directory}. Fine-tune the model first."
            )
        torch, tokenizer_class, model_class, _ = self._dependencies()
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("labels", ())) != EMOTION_LABELS:
            raise ModelNotReadyError("BERT label metadata is incompatible.")
        saved_config = BERTConfig(**metadata["config"])
        saved_config.validate()
        self.config = saved_config
        self._tokenizer = tokenizer_class.from_pretrained(
            self.artifact_directory, local_files_only=True
        )
        self._model = model_class.from_pretrained(
            self.artifact_directory, local_files_only=True
        )
        self._device = self._resolve_device(torch)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, text: str, *, mixed_threshold: float = 0.18) -> BERTPrediction:
        """Return ranked BERT emotion probabilities for one learning description."""

        if not 0.0 <= mixed_threshold <= 1.0:
            raise ValueError("Mixed-emotion threshold must be between 0 and 1.")
        if self._model is None or self._tokenizer is None:
            self.load()
        torch, _, _, _ = self._dependencies()
        prepared = self.preprocessor.preprocess(text).bert_text
        encoded = {
            key: value.to(self._device) for key, value in self._encode([prepared]).items()
        }
        self._model.eval()
        with torch.inference_mode():
            logits = self._model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        if probabilities.shape != (len(EMOTION_LABELS),) or not np.isfinite(probabilities).all():
            raise RuntimeError("BERT returned an invalid probability vector.")
        ranking = np.argsort(probabilities)[::-1]
        primary_index, secondary_index = int(ranking[0]), int(ranking[1])
        primary_score = float(probabilities[primary_index])
        secondary_score = float(probabilities[secondary_index])
        return BERTPrediction(
            primary_emotion=EMOTION_LABELS[primary_index],
            secondary_emotion=EMOTION_LABELS[secondary_index],
            confidence=primary_score,
            secondary_confidence=secondary_score,
            scores={
                label: round(float(probabilities[index]), 6)
                for index, label in enumerate(EMOTION_LABELS)
            },
            is_mixed=secondary_score >= mixed_threshold,
        )
