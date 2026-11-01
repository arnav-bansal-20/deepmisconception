"""Inter-rater agreement and scientific evaluation metrics service.

Computes:
1. Cohen's Kappa (kappa) between raters (AI vs Instructor or Instructor vs Instructor).
2. Observed percentage agreement (P_o).
3. Confusion matrix across comprehension and viva categories.
4. Per-class precision, recall, F1, and macro-F1.
5. Rate of uncertain / flagged predictions.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sqlalchemy.orm import Session

from app.models.comprehension_report import ComprehensionReport
from app.models.instructor_review import InstructorReview
from app.models.answer_assessment import AnswerAssessment
from app.models.session import CodingSession
from app.ml.dataset_generator import BenchmarkDatasetGenerator
from app.ml.baselines import RandomForestBaseline


COMPREHENSION_CLASSES = ["Understood", "Partial Understanding", "Possible Misconception"]
VIVA_CLASSES = ["Correct", "Partially Correct", "Incorrect"]


def compute_cohens_kappa(rater_a: List[str], rater_b: List[str], labels: List[str]) -> Tuple[float, str]:
    """Compute Cohen's Kappa between two categorical raters.
    
    Formula:
        kappa = (P_o - P_e) / (1 - P_e)
    where:
        P_o = observed proportionate agreement
        P_e = hypothetical probability of chance agreement
    """
    if not rater_a or not rater_b or len(rater_a) != len(rater_b):
        return 0.0, "Insufficient Data"

    n = len(rater_a)
    if n == 0:
        return 0.0, "Insufficient Data"

    # Count observed agreements
    agreements = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    p_o = agreements / n

    # Compute marginal distributions for chance agreement P_e
    p_e = 0.0
    for label in labels:
        count_a = sum(1 for a in rater_a if a == label)
        count_b = sum(1 for b in rater_b if b == label)
        p_e += (count_a / n) * (count_b / n)

    if 1.0 - p_e < 1e-9:
        # If perfect marginal agreement or denominator is zero
        kappa = 1.0 if p_o >= 0.999 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    # Landis & Koch (1977) interpretation
    if kappa < 0:
        interp = "Poor (Disagreement)"
    elif kappa <= 0.20:
        interp = "Slight Agreement"
    elif kappa <= 0.40:
        interp = "Fair Agreement"
    elif kappa <= 0.60:
        interp = "Moderate Agreement"
    elif kappa <= 0.80:
        interp = "Substantial Agreement"
    else:
        interp = "Almost Perfect Agreement"

    return float(round(kappa, 4)), interp


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> Dict[str, Any]:
    """Calculate confusion matrix, per-class precision/recall/F1, and macro-F1."""
    label_to_idx = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1

    total_samples = len(y_true)
    correct_count = sum(cm[i][i] for i in range(num_classes))
    accuracy = correct_count / total_samples if total_samples > 0 else 0.0

    per_class = {}
    f1_scores = []
    
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(cm[i]),
        }
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,
        },
    }


class InterRaterEvaluationService:
    """Evaluates concordance between automated AI diagnoses and human instructor annotations."""

    def __init__(self, db: Session):
        self.db = db

    def evaluate_comprehension_inter_rater(
        self,
        classroom_id: Optional[str] = None,
        teacher_id: Optional[str] = None,
        include_benchmark: bool = True,
    ) -> Dict[str, Any]:
        """Compute Cohen's Kappa, agreement rate, and confusion matrix.
        
        Uses actual instructor reviews from the database where available,
        and optionally supplements with the synthetic gold-standard benchmark
        so educators can evaluate full statistical metrics immediately.
        """
        # 1. Fetch real session reviews from database
        query = (
            self.db.query(ComprehensionReport, InstructorReview)
            .join(CodingSession, CodingSession.id == ComprehensionReport.session_id)
            .join(InstructorReview, InstructorReview.session_id == ComprehensionReport.session_id)
        )
        if classroom_id:
            query = query.filter(CodingSession.classroom_id == classroom_id)
        if teacher_id:
            query = query.filter(CodingSession.teacher_id == teacher_id)

        pairs = query.all()
        db_ai_labels = []
        db_human_labels = []

        for report, review in pairs:
            if review.assigned_label in COMPREHENSION_CLASSES and report.overall_label in COMPREHENSION_CLASSES:
                db_ai_labels.append(report.overall_label)
                db_human_labels.append(review.assigned_label)

        # 2. If benchmark is requested or real reviews are sparse (< 10)
        benchmark_ai_labels = []
        benchmark_human_labels = []
        
        if include_benchmark or len(db_ai_labels) < 5:
            # Generate reproducible benchmark cohort
            X, y, records = BenchmarkDatasetGenerator.generate(n_samples=120, seed=42)
            # Train Random Forest baseline on 60% of data, evaluate on remaining 40%
            split = int(0.6 * len(X))
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]
            test_records = records[split:]

            rf = RandomForestBaseline().fit(X_train, y_train)
            test_preds = rf.predict(X_test)

            for i, pred_idx in enumerate(test_preds):
                true_label = test_records[i]["label"]
                ai_label = COMPREHENSION_CLASSES[pred_idx]
                benchmark_human_labels.append(true_label)
                benchmark_ai_labels.append(ai_label)

        # Combine datasets
        final_human = db_human_labels + benchmark_human_labels
        final_ai = db_ai_labels + benchmark_ai_labels

        kappa, interpretation = compute_cohens_kappa(
            rater_a=final_ai,
            rater_b=final_human,
            labels=COMPREHENSION_CLASSES,
        )

        metrics = compute_classification_metrics(
            y_true=final_human,
            y_pred=final_ai,
            labels=COMPREHENSION_CLASSES,
        )

        # Compute flagged for review rate across DB sessions
        total_evals = self.db.query(AnswerAssessment).count()
        flagged_evals = (
            self.db.query(AnswerAssessment)
            .filter(AnswerAssessment.needs_instructor_review == True)
            .count()
        )
        flagged_rate = round((flagged_evals / total_evals) * 100, 2) if total_evals > 0 else 0.0

        p_o = round(metrics["accuracy"] * 100, 2)

        return {
            "cohen_kappa": kappa,
            "kappa_interpretation": interpretation,
            "percentage_agreement": p_o,
            "total_comparisons": len(final_human),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_metrics": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "flagged_uncertain_rate": flagged_rate,
            "source_breakdown": {
                "database_reviews": len(db_human_labels),
                "benchmark_samples": len(benchmark_human_labels),
            },
            "classes": COMPREHENSION_CLASSES,
        }
