"""Code submission and persistence routes."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.schemas.submission import CodeSubmissionCreate, CodeSubmissionResponse
from app.services.ast_analyzer import ASTAnalyzer
from app.services.safe_executor import SafeExecutor

router = APIRouter(prefix="/submissions", tags=["Submissions"])


@router.post("", response_model=CodeSubmissionResponse, status_code=status.HTTP_201_CREATED)
def submit_code(sub_in: CodeSubmissionCreate, db: Session = Depends(get_db)):
    """Save a snapshot of student code, analyze AST, and execute safely."""
    session = db.query(CodingSession).filter(CodingSession.id == sub_in.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")

    # 1. Run AST analysis
    analyzer = ASTAnalyzer(sub_in.code)
    ast_result = analyzer.analyze()

    # 2. Run Safe Execution if syntax is valid
    exec_output = None
    err_output = None
    exec_time_str = None
    
    if ast_result.is_syntax_valid:
        executor = SafeExecutor()
        run_res = executor.execute(sub_in.code)
        exec_output = run_res.stdout
        err_output = run_res.stderr if not run_res.success else None
        exec_time_str = f"{run_res.execution_time_ms:.1f}ms"
    else:
        err_output = ast_result.syntax_error_details.get("diagnostic") if ast_result.syntax_error_details else "Syntax error"

    submission = CodeSubmission(
        session_id=sub_in.session_id,
        code=sub_in.code,
        ast_summary=ast_result.model_dump(),
        execution_output=exec_output,
        error_output=err_output,
        is_valid=ast_result.is_syntax_valid,
        execution_time_ms=exec_time_str
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    # Auto-generate viva questions if none exist yet for this session
    from app.models.viva_question import VivaQuestion
    from app.services.question_generator import QuestionGenerator
    existing_q_count = db.query(VivaQuestion).filter(VivaQuestion.session_id == session.id).count()
    if existing_q_count == 0 and ast_result.is_syntax_valid:
        try:
            q_gen = QuestionGenerator()
            qs = q_gen.generate_questions(
                code=submission.code,
                ast_result=ast_result,
                execution_output=exec_output,
                count=3
            )
            for q in qs:
                db.add(VivaQuestion(
                    session_id=session.id,
                    submission_id=submission.id,
                    question_text=q.question_text,
                    focal_code=q.focal_code,
                    concept=q.concept,
                    difficulty=q.difficulty,
                    expected_reasoning=q.expected_reasoning,
                    rubric=q.rubric,
                    misconception_indicators=q.misconception_indicators,
                ))
            db.commit()
        except Exception as e:
            # Question generation error should not fail submission save
            pass

    return submission


@router.get("/session/{session_id}", response_model=List[CodeSubmissionResponse])
def get_session_submissions(session_id: str, db: Session = Depends(get_db)):
    """Retrieve all submissions for a session in chronological order."""
    submissions = (
        db.query(CodeSubmission)
        .filter(CodeSubmission.session_id == session_id)
        .order_by(CodeSubmission.created_at.desc())
        .all()
    )
    return submissions
