from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Get the project paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

INPUT_FILE = DATASET_DIR / "processed" / "emotion_dataset.csv"
OUTPUT_DIR = DATASET_DIR / "processed"

# Read processed dataset
df = pd.read_csv(INPUT_FILE)

print("Original Dataset")
print(df["emotion"].value_counts())
print()

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Train (80%) / Temp (20%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["emotion"],
    random_state=42
)

# Validation (10%) / Test (10%)
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["emotion"],
    random_state=42
)



OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
validation_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
print("Dataset Split Completed Successfully!\n")

print("Train")
print(train_df["emotion"].value_counts())
print()

print("Validation")
print(validation_df["emotion"].value_counts())
print()

print("Test")
print(test_df["emotion"].value_counts())

print("\nFiles Saved:")
print("train.csv")
print("validation.csv")
print("test.csv")