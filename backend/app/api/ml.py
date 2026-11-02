"""Machine Learning and Ablation API routes."""
from typing import Dict, Any, Optional
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.models.process_event import ProcessEvent
from app.ml.feature_extractor import MultimodalFeatureExtractor
from app.ml.multimodal_transformer import MultimodalCrossAttentionFusion
from app.ml.ablation import AblationRunner
from app.services.process_telemetry import ProcessTelemetryService

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Pre-trained / cached default model instance
_model_instance = MultimodalCrossAttentionFusion(seed=42)


@router.get("/status")
def get_ml_status() -> Dict[str, Any]:
    """Expose explicit ML model availability, device acceleration, and training status."""
    from app.ml.codebert_embedder import CodeBERTEmbedder, get_optimal_device
    from app.ml.sentence_transformer_embedder import SentenceTransformerEmbedder
    from app.ml.multimodal_transformer import TORCH_AVAILABLE

    codebert_stat = CodeBERTEmbedder.get_status()
    st_stat = SentenceTransformerEmbedder.get_status()
    device = get_optimal_device()

    return {
        "status": "healthy",
        "pytorch_available": TORCH_AVAILABLE,
        "compute_device": device,
        "models": {
            "codebert": {
                "name": codebert_stat["model_name"],
                "is_available": codebert_stat["is_available"],
                "status": codebert_stat["status"],
                "device": codebert_stat["device"],
                "dimension": codebert_stat["dimension"],
                "fallback_feature": "AST structural representation" if not codebert_stat["is_available"] else None,
            },
            "sentence_transformer": {
                "name": st_stat["model_name"],
                "is_available": st_stat["is_available"],
                "status": st_stat["status"],
                "device": st_stat["device"],
                "dimension": st_stat["dimension"],
                "fallback_feature": "Rubric semantic score" if not st_stat["is_available"] else None,
            },
            "multimodal_transformer": {
                "name": _model_instance.model_name,
                "framework": _model_instance.model_framework,
                "is_available": TORCH_AVAILABLE,
                "is_trained": _model_instance.is_trained,
                "device": str(_model_instance.device),
                "num_classes": _model_instance.num_classes,
                "class_labels": ["Understood", "Partial Understanding", "Possible Misconception"],
            },
        },
    }


@router.get("/ablation")
def get_ablation_benchmarks(
    samples: int = Query(180, ge=30, le=600, description="Number of synthetic student samples"),
    seed: int = Query(42, description="Random seed for reproducibility"),
) -> Dict[str, Any]:
    """Execute systematic multimodal ablation study across Code, Explanation, and Process."""
    return AblationRunner.run_ablation(n_samples=samples, seed=seed)


@router.post("/predict/{session_id}")
def predict_session_comprehension(
    session_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Run multimodal cross-attention neural inference on live session data."""
    session = db.query(CodingSession).filter(CodingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")

    latest_sub = (
        db.query(CodeSubmission)
        .filter(CodeSubmission.session_id == session_id)
        .order_by(CodeSubmission.created_at.desc())
        .first()
    )
    assessments = (
        db.query(AnswerAssessment)
        .filter(AnswerAssessment.session_id == session_id)
        .all()
    )
    proc_events = (
        db.query(ProcessEvent)
        .filter(ProcessEvent.session_id == session_id)
        .all()
    )

    code = latest_sub.code if latest_sub else ""
    ast_sum = latest_sub.ast_summary if latest_sub else None
    explanation = " ".join(a.student_answer for a in assessments) if assessments else ""
    proc_features = ProcessTelemetryService.compute_features(session_id, proc_events)

    feat = MultimodalFeatureExtractor.extract(code, explanation, proc_features, ast_sum)
    X = np.expand_dims(feat, axis=0)

    probs = _model_instance.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    labels = ["Understood", "Partial Understanding", "Possible Misconception"]

    return {
        "session_id": session_id,
        "predicted_label": labels[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "class_probabilities": {
            labels[i]: round(float(probs[i]), 4) for i in range(len(labels))
        },
        "feature_dimensions": {
            "ast_features_dim": 16,
            "explanation_features_dim": 16,
            "process_features_dim": 8,
            "total_dim": 40,
        },
    }
