"""Shared pytest configuration and fixtures for DeepMisconception test suite."""
import pytest
from app.services.comprehension_engine import ComprehensionEngine
from app.ml.multimodal_transformer import MultimodalPredictionResult
from app.database import Base, engine


class MockIntegrationClassifier:
    """Mock trained classifier used for integration tests of report synthesis."""

    def predict_session_bundle(self, bundle):
        answers = bundle.explanation.answers if bundle.explanation else []
        labels = [a.get("answer_label") for a in answers]

        incorrect_count = sum(1 for l in labels if l == "Incorrect")
        partial_count = sum(1 for l in labels if l == "Partially Correct")

        if incorrect_count >= 2:
            pred_class = "Possible Misconception"
            probs = {"Understood": 0.05, "Partial Understanding": 0.15, "Possible Misconception": 0.80}
            conf = 0.80
        elif incorrect_count == 1 or partial_count >= 1:
            pred_class = "Partial Understanding"
            probs = {"Understood": 0.20, "Partial Understanding": 0.70, "Possible Misconception": 0.10}
            conf = 0.70
        else:
            pred_class = "Understood"
            probs = {"Understood": 0.85, "Partial Understanding": 0.10, "Possible Misconception": 0.05}
            conf = 0.85

        return MultimodalPredictionResult(
            model_status="READY",
            model_version="mock-trained-v1.0",
            predicted_class=pred_class,
            class_probabilities=probs,
            confidence=conf,
            modality_availability={
                "code": bool(bundle.code and bundle.code.code),
                "explanation": bool(bundle.explanation and len(bundle.explanation.answers) > 0),
                "process": bool(bundle.process and len(bundle.process.events) > 0),
            },
            supporting_feature_scores={},
            needs_instructor_review=False,
        )


@pytest.fixture(autouse=True)
def configure_test_classifier(request):
    """Enable mock classifier for integration tests of report synthesis, while resetting for dedicated ML unit tests."""
    Base.metadata.create_all(bind=engine)
    # When testing the un-mocked architecture or model-not-trained states, ensure no override is active
    if "test_transformer_final_classifier" in request.node.nodeid or "test_multimodal_methodology" in request.node.nodeid:
        ComprehensionEngine.reset_classifier_override()
        yield
        ComprehensionEngine.reset_classifier_override()
    else:
        ComprehensionEngine.set_classifier_override(MockIntegrationClassifier())
        yield
        ComprehensionEngine.reset_classifier_override()
