from prediction.emotion_pipeline import EmotionPredictionPipeline

pipeline = EmotionPredictionPipeline()

result = pipeline.predict(
    "I don't understand recursion even after studying for hours."
)

print(result)