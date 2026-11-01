"""Code analysis and execution API routes."""
from fastapi import APIRouter, HTTPException, status
from app.schemas.ast_analysis import ASTAnalysisRequest, ASTAnalysisResult
from app.schemas.submission import CodeRunRequest, CodeRunResponse
from app.services.ast_analyzer import ASTAnalyzer
from app.services.safe_executor import SafeExecutor

router = APIRouter(prefix="/code", tags=["Code Analysis & Execution"])


@router.post("/analyze", response_model=ASTAnalysisResult)
def analyze_code(payload: ASTAnalysisRequest):
    """Perform concrete Python AST parsing and concept identification."""
    analyzer = ASTAnalyzer(payload.code)
    return analyzer.analyze()


@router.post("/run", response_model=CodeRunResponse)
def run_code(payload: CodeRunRequest):
    """Execute Python code within the isolated subprocess sandbox."""
    executor = SafeExecutor(timeout_seconds=payload.timeout_seconds)
    return executor.execute(payload.code)
