import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from models.bilstm import (
    BiLSTMEmotionClassifier,
    EMOTION_LABELS,
)

# Read training dataset
train_df = pd.read_csv(PROJECT_ROOT / "dataset" / "processed" / "train.csv")

texts = train_df["text"].tolist()
labels = train_df["emotion"].tolist()

# Compute class weights
unique_labels = np.array(sorted(set(labels)))

weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_labels,
    y=np.array(labels),
)

label_to_index = {
    label: EMOTION_LABELS.index(label)
    for label in unique_labels
}

class_weight = {
    label_to_index[label]: float(weight)
    for label, weight in zip(unique_labels, weights)
}


print("Class Weights")
print(class_weight)
print()

classifier = BiLSTMEmotionClassifier()

history = classifier.train(
    texts=texts,
    labels=labels,
    class_weight=class_weight,
)

print("\nTraining Finished Successfully!")

print("\nArtifacts saved to:")
print(classifier.artifact_directory)