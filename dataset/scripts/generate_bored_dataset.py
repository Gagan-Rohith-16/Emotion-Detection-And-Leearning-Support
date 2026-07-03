import pandas as pd
import random

subjects = [
    "mathematics",
    "computer science",
    "physics",
    "chemistry",
    "biology",
    "history",
    "english",
]

activities = [
    "reading the chapter",
    "watching the lecture",
    "solving problems",
    "doing assignments",
    "revising notes",
    "preparing for exams",
]

feelings = [
    "I'm getting bored",
    "I can't stay interested",
    "I keep losing focus",
    "This feels repetitive",
    "I'm zoning out",
    "I don't feel motivated",
    "Nothing feels exciting anymore",
]

templates = [
    "{f} while {a} in {s}.",
    "{a} in {s} is becoming very boring.",
    "{f}. {a} doesn't keep my attention.",
    "I keep getting distracted while {a} in {s}.",
    "I have no interest in {a} anymore.",
    "{s} feels repetitive and I cannot stay focused.",
    "I'm tired of {a} every day.",
]

rows = []

for _ in range(3000):
    rows.append({
        "text": random.choice(templates).format(
            f=random.choice(feelings),
            a=random.choice(activities),
            s=random.choice(subjects),
        ),
        "emotion": "Bored"
    })

df = pd.DataFrame(rows)

from pathlib import Path

# Create DataFrame
df = pd.DataFrame(rows)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent

# Create dataset/raw if it doesn't exist
RAW_DIR = SCRIPT_DIR.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Output file
output_file = RAW_DIR / "bored_dataset.csv"

# Save CSV
df.to_csv(output_file, index=False)

print(df.head())
print()
print(f"Generated {len(df)} samples")
print(f"Saved to: {output_file}")