"""Unit tests for Phase 8 Machine Learning Pipeline, Transformer Fusion & Ablation."""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.ml.feature_extractor import (
    ASTFeatureExtractor,
    ExplanationFeatureExtractor,
    ProcessFeatureExtractor,
    MultimodalFeatureExtractor,
)
from app.ml.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    DecisionTreeBaseline,
    MLPBaseline,
)
from app.ml.multimodal_transformer import MultimodalCrossAttentionFusion
from app.ml.ablation import AblationRunner
from app.ml.dataset_generator import BenchmarkDatasetGenerator

client = TestClient(app)


def test_feature_extractors_dimensions():
    """Verify feature extractor dimensions and modality ablation masking."""
    code = "def loop_sum(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
    explanation = "The loop adds i to the accumulator total until n is reached."
    proc = {"paste_ratio": 0.1, "burstiness_ratio": 0.4, "coding_duration_seconds": 120, "run_count": 3}

    ast_vec = ASTFeatureExtractor.extract(code)
    assert ast_vec.shape == (16,)
    assert ast_vec.dtype == np.float32
    assert ast_vec[13] == 0.0  # not recursive
    assert ast_vec[14] == 0.0  # no syntax error

    exp_vec = ExplanationFeatureExtractor.extract(explanation, code)
    assert exp_vec.shape == (16,)
    assert exp_vec.dtype == np.float32

    proc_vec = ProcessFeatureExtractor.extract(proc)
    assert proc_vec.shape == (8,)
    assert proc_vec[0] == pytest.approx(0.1, 0.01)

    full_vec = MultimodalFeatureExtractor.extract(code, explanation, proc)
    assert full_vec.shape == (40,)

    # Test masking
    code_only = MultimodalFeatureExtractor.extract(code, explanation, proc, modality_mask="code_only")
    assert np.all(code_only[16:] == 0.0)
    assert np.any(code_only[:16] > 0.0)

    exp_only = MultimodalFeatureExtractor.extract(code, explanation, proc, modality_mask="explanation_only")
    assert np.all(exp_only[:16] == 0.0)
    assert np.all(exp_only[32:] == 0.0)
    assert np.any(exp_only[16:32] > 0.0)


def test_dataset_generator():
    """Verify reproducible synthetic dataset generator."""
    X, y, records = BenchmarkDatasetGenerator.generate(n_samples=60, seed=42)
    assert X.shape == (60, 40)
    assert len(y) == 60
    assert len(records) == 60
    # Balanced classes
    assert list(np.bincount(y)) == [20, 20, 20]


def test_classical_baselines_training_and_evaluation():
    """Verify classical Scikit-Learn baseline classifiers."""
    X, y, _ = BenchmarkDatasetGenerator.generate(n_samples=90, seed=42)
    X_train, X_test = X[:60], X[60:]
    y_train, y_test = y[:60], y[60:]

    for ModelClass in [LogisticRegressionBaseline, RandomForestBaseline, DecisionTreeBaseline, MLPBaseline]:
        clf = ModelClass()
        clf.fit(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 3


def test_multimodal_cross_attention_transformer():
    """Verify PyTorch/NumPy cross-attention neural network architecture."""
    X, y, _ = BenchmarkDatasetGenerator.generate(n_samples=90, seed=42)
    X_train, X_test = X[:60], X[60:]
    y_train, y_test = y[:60], y[60:]

    model = MultimodalCrossAttentionFusion(seed=42)

    # Test single forward pass
    x_code = X_test[:, :16]
    x_viva = X_test[:, 16:32]
    x_proc = X_test[:, 32:40]
    probs, attn = model.forward(x_code, x_viva, x_proc)

    assert probs.shape == (30, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    assert attn.shape == (30, 32)

    # Test training
    model.fit(X_train, y_train, epochs=25, lr=0.03)
    eval_metrics = model.evaluate(X_test, y_test)
    assert eval_metrics["accuracy"] >= 0.50
    assert eval_metrics["macro_f1"] >= 0.50


def test_ablation_study_runner():
    """Verify ablation study runs across all 5 modality configurations."""
    ablation = AblationRunner.run_ablation(n_samples=90, seed=42)
    assert ablation["n_samples"] == 90
    assert len(ablation["results"]) == 5

    configs = [r["config_key"] for r in ablation["results"]]
    assert "code_only" in configs
    assert "explanation_only" in configs
    assert "process_only" in configs
    assert "code_plus_explanation" in configs
    assert "full_multimodal" in configs


def test_ml_api_endpoints():
    """Verify /api/ml/ablation and /api/ml/predict/{session_id} endpoints."""
    # Ablation endpoint
    ablation_res = client.get("/api/ml/ablation?samples=60")
    assert ablation_res.status_code == 200
    data = ablation_res.json()
    assert len(data["results"]) == 5

    # Predict endpoint
    sess_res = client.post("/api/sessions", json={"student_id": "STU-ML-01", "student_name": "ML Tester", "title": "ML Test"})
    s_id = sess_res.json()["id"]

    pred_res = client.post(f"/api/ml/predict/{s_id}")
    assert pred_res.status_code == 200
    pred = pred_res.json()
    assert "predicted_label" in pred
    assert "confidence" in pred
    assert "class_probabilities" in pred
    assert pred["feature_dimensions"]["total_dim"] == 40
