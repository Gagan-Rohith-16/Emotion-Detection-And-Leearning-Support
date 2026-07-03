import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.bert import BERTEmotionClassifier

# Load training data
# Load training data
train_df = pd.read_csv(PROJECT_ROOT / "dataset" / "processed" / "train.csv")

# Create a balanced subset
samples_per_class = 4000

balanced_parts = []

for emotion in train_df["emotion"].unique():
    emotion_df = train_df[train_df["emotion"] == emotion]

    if len(emotion_df) > samples_per_class:
        emotion_df = emotion_df.sample(
            n=samples_per_class,
            random_state=42
        )

    balanced_parts.append(emotion_df)

train_df = (
    pd.concat(balanced_parts)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

print("=" * 60)
print("Balanced Dataset")
print(train_df["emotion"].value_counts())
print("Training Samples:", len(train_df))
print("=" * 60)

texts = train_df["text"].tolist()
labels = train_df["emotion"].tolist()

print("=" * 60)
print("Training Samples :", len(texts))
print("Emotion Distribution")
print(train_df["emotion"].value_counts())
print("=" * 60)

classifier = BERTEmotionClassifier()

history = classifier.fine_tune(
    texts=texts,
    labels=labels,
)

print("\nTraining Finished Successfully!")

print("\nArtifacts saved to:")
print(classifier.artifact_directory)