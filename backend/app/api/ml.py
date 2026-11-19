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
from app.ml.multimodal_transformer import get_multimodal_transformer, LABELS
from app.ml.evaluation_pipeline import ABLATION_CONFIGURATIONS
from app.ml.multimodal_inputs import MultimodalStudentInputBundle
from app.services.process_telemetry import ProcessTelemetryService

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


@router.get("/status")
def get_ml_status() -> Dict[str, Any]:
    """Expose explicit ML model availability, device acceleration, and training status."""
    from app.ml.codebert_embedder import CodeBERTEmbedder, get_optimal_device
    from app.ml.sentence_transformer_embedder import SentenceTransformerEmbedder
    from app.ml.multimodal_transformer import TORCH_AVAILABLE

    codebert_stat = CodeBERTEmbedder.get_status()
    st_stat = SentenceTransformerEmbedder.get_status()
    device = get_optimal_device()
    model = get_multimodal_transformer()

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
                "name": model.model_name,
                "framework": model.model_framework,
                "is_available": TORCH_AVAILABLE,
                "is_trained": model.is_trained,
                "device": str(model.device),
                "num_classes": model.num_classes,
                "class_labels": LABELS,
            },
        },
    }


@router.get("/ablation")
def get_ablation_benchmarks() -> Dict[str, Any]:
    """Return the 9 systematic ablation configurations prepared for empirical benchmarking."""
    return {
        "status": "CONFIGURED",
        "total_configurations": len(ABLATION_CONFIGURATIONS),
        "configurations": ABLATION_CONFIGURATIONS,
        "note": "Ablation matrix is configured and ready to execute once the multimodal dataset is collected and models are trained.",
    }


@router.post("/predict/{session_id}")
def predict_session_comprehension(
    session_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Run multimodal cross-attention neural inference on live session data adhering to Section 15 schema."""
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
    questions = (
        db.query(VivaQuestion)
        .filter(VivaQuestion.session_id == session_id)
        .all()
    )
    proc_events = (
        db.query(ProcessEvent)
        .filter(ProcessEvent.session_id == session_id)
        .all()
    )

    proc_features = ProcessTelemetryService.compute_features(session_id, proc_events)

    input_bundle = MultimodalStudentInputBundle.from_session_data(
        session_id=session_id,
        events=proc_events,
        telemetry_features=proc_features,
        code=latest_sub.code if latest_sub else "",
        ast_summary=latest_sub.ast_summary if latest_sub else {},
        exec_output=latest_sub.execution_output if latest_sub else "",
        err_output=latest_sub.error_output if latest_sub else "",
        exit_code=0 if (latest_sub and latest_sub.execution_output and not latest_sub.error_output) else 1,
        is_valid=latest_sub.is_valid if latest_sub else True,
        viva_assessments=assessments,
        viva_questions=questions,
    )

    classifier = get_multimodal_transformer()
    pred_res = classifier.predict_session_bundle(input_bundle)

    return {
        "session_id": session_id,
        "predicted_class": pred_res.predicted_class,
        "class_probabilities": pred_res.class_probabilities,
        "model_status": pred_res.model_status,
        "model_version": pred_res.model_version,
        "modality_availability": pred_res.modality_availability,
        "supporting_feature_scores": pred_res.supporting_feature_scores,
        "needs_instructor_review": pred_res.needs_instructor_review,
        "confidence": pred_res.confidence,
    }
