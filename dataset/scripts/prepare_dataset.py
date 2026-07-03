import os
import pandas as pd

RAW_FILE = "../raw/go_emotions_dataset.csv"
OUTPUT_FILE = "../processed/emotion_dataset.csv"

df = pd.read_csv(RAW_FILE)

emotion_map = {
    "Confused": ["confusion"],
    "Curious": ["curiosity"],
    "Frustrated": [
        "anger",
        "annoyance",
        "disappointment",
        "disapproval",
    ],
    "Confident": [
        "approval",
        "optimism",
        "pride",
        "admiration",
    ],
}

rows = []

for _, row in df.iterrows():

    text = row["text"]

    assigned = False

    for emotion, labels in emotion_map.items():

        for label in labels:

            if row[label] == 1:

                rows.append(
                    {
                        "text": text,
                        "emotion": emotion,
                    }
                )

                assigned = True
                break

        if assigned:
            break

processed = pd.DataFrame(rows)

processed.to_csv(OUTPUT_FILE, index=False)

print(processed["emotion"].value_counts())

print()

print("Saved:", OUTPUT_FILE)