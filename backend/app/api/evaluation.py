"""API router for model performance evaluation, metrics, and report downloads."""
from fastapi import APIRouter, Response, HTTPException, status
from fastapi.responses import PlainTextResponse
from typing import Dict, Any

from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["Model Evaluation"])


@router.get("/metrics")
def get_evaluation_metrics() -> Dict[str, Any]:
    """Retrieve the full empirical evaluation suite:
    - Nine evaluation metrics (Accuracy, Precision, Recall, F1, Error, Sensitivity, Specificity, Confusion Matrix, 5-Fold CV)
    - Overall test performance (Table 1)
    - Class-wise diagnostic breakdown (Table 2)
    - Stratified group-aware 5-fold cross-validation (Table 3)
    - Confusion matrix with raw counts and normalized percentages
    - Baseline model comparisons
    """
    return EvaluationService.get_evaluation_metrics()


@router.get("/export/excel")
def export_excel_report():
    """Download comprehensive evaluation report as a multi-sheet Excel XML spreadsheet."""
    content = EvaluationService.generate_excel_report()
    if content == b"MODEL_NOT_TRAINED":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trained model checkpoint is unavailable for evaluation export."
        )

    filename = "DeepMisconception_Evaluation_Report.xml"
    return Response(
        content=content,
        media_type="application/vnd.ms-excel",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/export/csv", response_class=PlainTextResponse)
def export_csv_summary():
    """Download evaluation tables in structured CSV format."""
    csv_text = EvaluationService.generate_csv_summary()
    filename = "DeepMisconception_Evaluation_Metrics.csv"
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
