"""Classical machine learning baselines for comprehension classification."""
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

LABELS = ["Understood", "Partial Understanding", "Possible Misconception"]
LABEL_MAP = {lbl: idx for idx, lbl in enumerate(LABELS)}
REV_LABEL_MAP = {idx: lbl for idx, lbl in enumerate(LABELS)}


class BaseComprehensionClassifier:
    """Base wrapper providing uniform training and evaluation APIs."""

    def __init__(self, model_name: str, model_instance: Any):
        self.model_name = model_name
        self.model = model_instance

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseComprehensionClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return np.ones((len(X), len(LABELS))) / len(LABELS)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        acc = float(accuracy_score(y, preds))
        macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
        prec = float(precision_score(y, preds, average="macro", zero_division=0))
        rec = float(recall_score(y, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y, preds, labels=[0, 1, 2]).tolist()

        return {
            "model_name": self.model_name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "confusion_matrix": cm,
        }


class LogisticRegressionBaseline(BaseComprehensionClassifier):
    def __init__(self, random_state: int = 42):
        model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
        super().__init__("Logistic Regression", model)


class RandomForestBaseline(BaseComprehensionClassifier):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=8, random_state=random_state, class_weight="balanced")
        super().__init__("Random Forest", model)


class DecisionTreeBaseline(BaseComprehensionClassifier):
    def __init__(self, max_depth: int = 6, random_state: int = 42):
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight="balanced")
        super().__init__("Decision Tree", model)


class MLPBaseline(BaseComprehensionClassifier):
    def __init__(self, hidden_layer_sizes: tuple = (64, 32), random_state: int = 42):
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=500, random_state=random_state, early_stopping=True)
        super().__init__("Multilayer Perceptron (MLP)", model)
