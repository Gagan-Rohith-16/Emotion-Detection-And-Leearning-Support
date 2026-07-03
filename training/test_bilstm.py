import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.bilstm import BiLSTMEmotionClassifier

classifier = BiLSTMEmotionClassifier()

samples = [
    "I don't understand recursion even after studying for hours.",
    "I'm excited to learn machine learning.",
    "This chapter is so boring that I keep losing focus.",
    "I finally solved my programming problem and feel great!",
    "I want to know more about artificial intelligence.",
]

for text in samples:
    result = classifier.predict(text)

    print("=" * 70)
    print("Input:", text)
    print("Primary Emotion :", result.primary_emotion)
    print("Secondary Emotion:", result.secondary_emotion)
    print("Confidence:", round(result.confidence, 3))
    print("Mixed Emotion:", result.is_mixed)
    print()