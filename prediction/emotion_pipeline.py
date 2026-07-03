from models.bilstm import BiLSTMEmotionClassifier
from models.bert import BERTEmotionClassifier
from prediction.keyword_rules import KeywordRuleEngine
from prediction.emotion_fusion import EmotionFusionEngine


class EmotionPredictionPipeline:

    def __init__(self):

        self.bilstm = BiLSTMEmotionClassifier()
        self.bert = BERTEmotionClassifier()
        self.rules = KeywordRuleEngine()
        self.fusion = EmotionFusionEngine()
        self.bert_available = self.bert.is_ready()

    def predict(self, text: str):

        bilstm_result = self.bilstm.predict(text)

        bert_result = None

        if self.bert_available:
            try:
                bert_result = self.bert.predict(text)
            except Exception:
                # Keep the app functional in deployments where BERT weights are absent.
                bert_result = None

        rule_result = self.rules.analyze(text)

        fused = self.fusion.fuse(
            bilstm=bilstm_result,
            bert=bert_result,
            rules=rule_result,
        )

        return {
            "primary_emotion": fused.primary_emotion,
            "secondary_emotion": fused.secondary_emotion,
            "confidence": fused.confidence,
            "scores": fused.scores,
            "mixed": fused.is_mixed,
            "agreement": fused.model_agreement,
            "rule_evidence": fused.rule_evidence,
        }