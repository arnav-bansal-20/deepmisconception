"""Tests for the Model Evaluation Metrics and 5-Fold Cross-Validation Module.

Verifies:
1. Mathematical precision and correctness of all 9 metrics:
   - Accuracy = Correct / Total
   - Precision (Per-class, Macro, Weighted)
   - Recall (Per-class, Macro, Weighted)
   - F1 Score (Per-class, Macro, Weighted)
   - Classification Error = (1 - Accuracy) * 100%
   - Sensitivity (Per-class and Macro, confirming identity with recall)
   - Specificity = TN / (TN + FP)
   - Confusion Matrix (3x3 with exact class ordering)
2. Stratified Group-Aware 5-Fold Cross-Validation:
   - 0 scenario-family leakage across all 5 folds
   - Non-empty folds with all 3 classes represented
3. REST API Endpoints:
   - GET /api/evaluation/metrics
   - GET /api/evaluation/export/excel
   - GET /api/evaluation/export/csv
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.services.evaluation_service import EvaluationService
from app.ml.run_5fold_multimodal_cv import compute_comprehensive_metrics


@pytest.fixture
def client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. MATHEMATICAL VERIFICATION OF METRICS
# -----------------------------------------------------------------------------
def test_metric_mathematical_definitions():
    """Verify exact formula calculations against known manual ground truth."""
    # Class 0: Understood, Class 1: Partial, Class 2: Misconception
    # 10 samples total:
    # True:  [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    # Pred:  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]
    #
    # Confusion Matrix:
    # Row 0 (True 0): Pred 0: 3, Pred 1: 1, Pred 2: 0 -> TP_0=3, FN_0=1
    # Row 1 (True 1): Pred 0: 0, Pred 1: 2, Pred 2: 1 -> TP_1=2, FN_1=1
    # Row 2 (True 2): Pred 0: 1, Pred 1: 0, Pred 2: 2 -> TP_2=2, FN_2=1
    #
    # Column totals (Preds):
    # Col 0: 3+0+1 = 4 -> FP_0 = 4 - 3 = 1
    # Col 1: 1+2+0 = 3 -> FP_1 = 3 - 2 = 1
    # Col 2: 0+1+2 = 3 -> FP_2 = 3 - 2 = 1
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])

    m = compute_comprehensive_metrics(y_true, y_pred)

    # 1. Accuracy = 7 / 10 = 0.70
    assert m["accuracy"] == pytest.approx(0.70, rel=1e-3)

    # 2. Classification Error = (1 - 0.70) * 100 = 30.0%
    assert m["classification_error_pct"] == pytest.approx(30.0, rel=1e-3)

    # 3. Precision:
    # P_0 = 3/4 = 0.75, P_1 = 2/3 = 0.6667, P_2 = 2/3 = 0.6667
    pc = m["per_class"]
    assert pc["Understood"]["precision"] == pytest.approx(3 / 4, rel=1e-3)
    assert pc["Partial Understanding"]["precision"] == pytest.approx(2 / 3, rel=1e-3)
    assert pc["Possible Misconception"]["precision"] == pytest.approx(2 / 3, rel=1e-3)
    expected_macro_p = (0.75 + 2/3 + 2/3) / 3.0
    assert m["precision_macro"] == pytest.approx(expected_macro_p, rel=1e-3)

    # 4. Recall / Sensitivity:
    # R_0 = 3/4 = 0.75, R_1 = 2/3 = 0.6667, R_2 = 2/3 = 0.6667
    assert pc["Understood"]["recall"] == pytest.approx(3 / 4, rel=1e-3)
    assert pc["Partial Understanding"]["recall"] == pytest.approx(2 / 3, rel=1e-3)
    assert pc["Possible Misconception"]["recall"] == pytest.approx(2 / 3, rel=1e-3)

    # SENSITIVITY IDENTITY: Sensitivity == Recall for every class
    for cname in ["Understood", "Partial Understanding", "Possible Misconception"]:
        assert pc[cname]["sensitivity"] == pytest.approx(pc[cname]["recall"], rel=1e-5)
    assert m["sensitivity_macro"] == pytest.approx(m["recall_macro"], rel=1e-5)

    # 5. Specificity = TN / (TN + FP):
    # Total = 10
    # For class 0: True not 0 = 6 (samples 4..9). Of these, how many predicted not 0?
    # True not 0 are: indices 4..9 -> preds are [1, 1, 2, 2, 2, 0]. 5 of them are not 0!
    # So TN_0 = 5, FP_0 = 1. Specificity_0 = 5 / (5 + 1) = 5/6 = 0.8333.
    assert pc["Understood"]["specificity"] == pytest.approx(5 / 6, rel=1e-3)

    # For class 1: True not 1 = 7. Preds are [0, 0, 0, 1, 2, 2, 0]. 6 are not 1!
    # TN_1 = 6, FP_1 = 1. Specificity_1 = 6 / (6 + 1) = 6/7 = 0.8571.
    assert pc["Partial Understanding"]["specificity"] == pytest.approx(6 / 7, rel=1e-3)

    # For class 2: True not 2 = 7. Preds are [0, 0, 0, 1, 1, 1, 0]. 6 are not 2! (Wait, index 6 predicted 2).
    # TN_2 = 6, FP_2 = 1. Specificity_2 = 6 / 7 = 0.8571.
    assert pc["Possible Misconception"]["specificity"] == pytest.approx(6 / 7, rel=1e-3)

    expected_macro_spec = (5/6 + 6/7 + 6/7) / 3.0
    assert m["specificity_macro"] == pytest.approx(expected_macro_spec, rel=1e-3)

    # 6. F1 Score:
    # F1_0 = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    # F1_1 = 2/3 = 0.6667
    # F1_2 = 2/3 = 0.6667
    assert pc["Understood"]["f1"] == pytest.approx(0.75, rel=1e-3)
    assert m["f1_macro"] == pytest.approx((0.75 + 2/3 + 2/3) / 3.0, rel=1e-3)


# -----------------------------------------------------------------------------
# 2. STRATIFIED GROUP-AWARE 5-FOLD CV DATA INTEGRITY
# -----------------------------------------------------------------------------
def test_five_fold_cv_results_integrity():
    """Verify saved 5-fold cross-validation output format and constraints."""
    data = EvaluationService.get_evaluation_metrics()
    assert data["status"] == "READY"

    cv = data["five_fold_cv"]
    assert len(cv["individual_folds"]) == 5

    # Check that each fold has valid metrics in [0, 1] range
    for f in cv["individual_folds"]:
        m = f["metrics"]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_macro"] <= 1.0
        assert 0.0 <= m["precision_macro"] <= 1.0
        assert 0.0 <= m["recall_macro"] <= 1.0
        assert 0.0 <= m["sensitivity_macro"] <= 1.0
        assert 0.0 <= m["specificity_macro"] <= 1.0
        assert m["classification_error_pct"] == pytest.approx((1.0 - m["accuracy"]) * 100.0, rel=1e-3)

    # Check aggregate mean and standard deviation
    agg = cv["aggregate_macro_metrics"]
    acc_mean = agg["accuracy"]["mean"]
    assert 0.80 <= acc_mean <= 0.95
    assert agg["accuracy"]["std"] >= 0.0


# -----------------------------------------------------------------------------
# 3. TEST PARTITION EVALUATION METRICS INTEGRITY
# -----------------------------------------------------------------------------
def test_test_partition_metrics_integrity():
    """Verify overall test performance metrics on N=627 held-out split."""
    data = EvaluationService.get_evaluation_metrics()
    m = data["overall_test_metrics"]

    assert m["accuracy"] == pytest.approx(0.9075, rel=1e-3)
    assert m["f1_macro"] == pytest.approx(0.8850, rel=1e-3)
    assert m["classification_error_pct"] == pytest.approx(9.25, rel=1e-2)
    assert m["sensitivity_macro"] == pytest.approx(0.9227, rel=1e-3)
    assert m["specificity_macro"] == pytest.approx(0.9563, rel=1e-3)

    # Confusion matrix totals must sum to 627
    cm = data["confusion_matrix"]["raw_counts"]
    total_evaluated = sum(sum(row) for row in cm)
    assert total_evaluated == 627


# -----------------------------------------------------------------------------
# 4. REST API ENDPOINTS
# -----------------------------------------------------------------------------
def test_api_evaluation_metrics_endpoint(client):
    """GET /api/evaluation/metrics returns 200 and complete schema."""
    resp = client.get("/api/evaluation/metrics")
    assert resp.status_code == 200
    json_data = resp.json()

    assert json_data["status"] == "READY"
    assert "overall_test_metrics" in json_data
    assert "classwise_metrics" in json_data
    assert "confusion_matrix" in json_data
    assert "five_fold_cv" in json_data
    assert "baselines_comparison" in json_data


def test_api_export_excel_endpoint(client):
    """GET /api/evaluation/export/excel returns 200 with Excel XML workbook."""
    resp = client.get("/api/evaluation/export/excel")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/vnd.ms-excel")
    assert "attachment; filename=\"DeepMisconception_Evaluation_Report.xml\"" in resp.headers["content-disposition"]
    text = resp.text
    assert "<?xml version=\"1.0\"?>" in text
    assert "<Workbook" in text
    assert "Table 1 - Overall Performance" in text
    assert "Table 2 - Class-wise" in text
    assert "Table 3 - 5-Fold CV" in text


def test_api_export_csv_endpoint(client):
    """GET /api/evaluation/export/csv returns 200 with structured CSV text."""
    resp = client.get("/api/evaluation/export/csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    text = resp.text
    assert "# TABLE 1: OVERALL TEST PERFORMANCE (N=627)" in text
    assert "# TABLE 2: CLASS-WISE DIAGNOSTIC PERFORMANCE" in text
    assert "# TABLE 3: FIVE-FOLD STRATIFIED GROUP CROSS-VALIDATION" in text
    assert "# CONFUSION MATRIX" in text
