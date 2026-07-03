from models.bert import BERTEmotionClassifier

classifier = BERTEmotionClassifier()

prediction = classifier.predict(
    "I am excited to learn machine learning."
)

print("Primary:", prediction.primary_emotion)
print("Confidence:", prediction.confidence)
print("Scores:")
print(prediction.scores)