from pathlib import Path
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

processed_file = DATASET_DIR / "processed" / "emotion_dataset.csv"
bored_file = DATASET_DIR / "raw" / "bored_dataset.csv"

# Load datasets
emotion_df = pd.read_csv(processed_file)
bored_df = pd.read_csv(bored_file)

print("Original Dataset")
print(emotion_df["emotion"].value_counts())
print()

print("Bored Dataset")
print(bored_df["emotion"].value_counts())
print()

# Merge
merged_df = pd.concat([emotion_df, bored_df], ignore_index=True)

# Shuffle
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_file = DATASET_DIR / "processed" / "emotion_dataset.csv"
merged_df.to_csv(output_file, index=False)

print("Merged Dataset")
print(merged_df["emotion"].value_counts())
print()

print(f"Saved merged dataset to:\n{output_file}")