"""Viva questions generation and dynamic answer assessment routes."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.schemas.viva import (
    VivaGenerateRequest,
    VivaQuestionResponse,
    VivaAnswerSubmit,
    AnswerAssessmentResponse,
)
from app.schemas.ast_analysis import ASTAnalysisResult
from app.services.ast_analyzer import ASTAnalyzer
from app.services.question_generator import QuestionGenerator
from app.services.answer_evaluator import AnswerEvaluator

router = APIRouter(prefix="/viva", tags=["Viva Questions"])


@router.post("/generate", response_model=List[VivaQuestionResponse], status_code=status.HTTP_201_CREATED)
def generate_viva_questions(req: VivaGenerateRequest, db: Session = Depends(get_db)):
    """Dynamically generate code-grounded viva questions using the LLM engine."""
    session = db.query(CodingSession).filter(CodingSession.id == req.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")

    # Find relevant submission
    if req.submission_id:
        submission = db.query(CodeSubmission).filter(CodeSubmission.id == req.submission_id).first()
    else:
        submission = (
            db.query(CodeSubmission)
            .filter(CodeSubmission.session_id == req.session_id)
            .order_by(CodeSubmission.created_at.desc())
            .first()
        )

    if not submission:
        raise HTTPException(status_code=400, detail="Cannot generate viva questions: No code submitted yet for this session.")

    # Reconstruct or compute AST result
    if submission.ast_summary and submission.ast_summary.get("is_syntax_valid"):
        ast_result = ASTAnalysisResult(**submission.ast_summary)
    else:
        analyzer = ASTAnalyzer(submission.code)
        ast_result = analyzer.analyze()

    # Clean up previous unanswered questions for this session to ensure clean regeneration
    answered_q_ids = [
        row[0] for row in db.query(AnswerAssessment.question_id).filter(AnswerAssessment.session_id == session.id).all()
    ]
    unanswered_qs = (
        db.query(VivaQuestion)
        .filter(VivaQuestion.session_id == session.id)
        .filter(~VivaQuestion.id.in_(answered_q_ids))
        .all()
    )
    for old_q in unanswered_qs:
        db.delete(old_q)
    db.flush()

    generator = QuestionGenerator()
    generated = generator.generate_questions(
        code=submission.code,
        ast_result=ast_result,
        execution_output=submission.execution_output,
        count=req.count
    )

    created_questions = []
    for q in generated:
        viva_q = VivaQuestion(
            session_id=session.id,
            submission_id=submission.id,
            question_text=q.question_text,
            focal_code=q.focal_code,
            concept=q.concept,
            difficulty=q.difficulty,
            expected_reasoning=q.expected_reasoning,
            rubric=q.rubric,
            misconception_indicators=q.misconception_indicators,
        )
        db.add(viva_q)
        created_questions.append(viva_q)

    db.commit()

    # Refresh and return
    response = []
    for q in created_questions:
        db.refresh(q)
        response.append(
            VivaQuestionResponse(
                id=q.id,
                session_id=q.session_id,
                submission_id=q.submission_id,
                question_text=q.question_text,
                focal_code=q.focal_code,
                concept=q.concept,
                difficulty=q.difficulty,
                expected_reasoning=q.expected_reasoning,
                rubric=q.rubric,
                misconception_indicators=q.misconception_indicators,
                created_at=q.created_at,
                has_answer=False,
                assessment=None
            )
        )

    return response


@router.post("/evaluate", response_model=AnswerAssessmentResponse, status_code=status.HTTP_201_CREATED)
def evaluate_student_answer(answer_in: VivaAnswerSubmit, db: Session = Depends(get_db)):
    """Dynamically evaluate a student's viva explanation against code reality and rubrics."""
    question = db.query(VivaQuestion).filter(VivaQuestion.id == answer_in.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Viva question not found")

    # Fetch code submission context
    submission = None
    if question.submission_id:
        submission = db.query(CodeSubmission).filter(CodeSubmission.id == question.submission_id).first()

    code_context = submission.code if submission else (question.focal_code or "")
    exec_output = submission.execution_output if submission else ""
    ast_summary = submission.ast_summary if submission else None

    evaluator = AnswerEvaluator()
    eval_res = evaluator.evaluate_answer(
        student_answer=answer_in.student_answer,
        question_text=question.question_text,
        focal_code=question.focal_code,
        concept=question.concept,
        expected_reasoning=question.expected_reasoning,
        rubric=question.rubric,
        submitted_code=code_context,
        execution_output=exec_output,
        ast_summary=ast_summary,
    )

    # Check if assessment already exists for this question (update if exists)
    existing_assessment = db.query(AnswerAssessment).filter(AnswerAssessment.question_id == question.id).first()
    if existing_assessment:
        existing_assessment.student_answer = answer_in.student_answer
        existing_assessment.answer_label = eval_res.answer_label
        existing_assessment.expected_reasoning = eval_res.expected_reasoning
        existing_assessment.concepts_correct = eval_res.concepts_correct
        existing_assessment.concepts_missed = eval_res.concepts_missed
        existing_assessment.contradictions = eval_res.contradictions
        existing_assessment.supporting_evidence = eval_res.supporting_evidence
        existing_assessment.evaluation_explanation = eval_res.evaluation_explanation
        existing_assessment.uncertainty_score = eval_res.uncertainty_score
        existing_assessment.needs_instructor_review = eval_res.needs_instructor_review
        db.commit()
        db.refresh(existing_assessment)
        return existing_assessment

    assessment = AnswerAssessment(
        question_id=question.id,
        session_id=question.session_id,
        student_answer=answer_in.student_answer,
        answer_label=eval_res.answer_label,
        expected_reasoning=eval_res.expected_reasoning,
        concepts_correct=eval_res.concepts_correct,
        concepts_missed=eval_res.concepts_missed,
        contradictions=eval_res.contradictions,
        supporting_evidence=eval_res.supporting_evidence,
        evaluation_explanation=eval_res.evaluation_explanation,
        uncertainty_score=eval_res.uncertainty_score,
        needs_instructor_review=eval_res.needs_instructor_review,
        instructor_status="pending"
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    # Check if student has answered the viva questions for this session; if so, transition session status to 'completed' and auto-synthesize report
    session_obj = db.query(CodingSession).filter(CodingSession.id == question.session_id).first()
    if session_obj:
        total_qs = db.query(func.count(VivaQuestion.id)).filter(VivaQuestion.session_id == question.session_id).scalar() or 0
        answered_qs = db.query(func.count(AnswerAssessment.id)).filter(AnswerAssessment.session_id == question.session_id).scalar() or 0
        if total_qs > 0 and answered_qs >= total_qs:
            if session_obj.status == "active":
                session_obj.status = "completed"
                db.add(session_obj)
                db.commit()
            try:
                from app.services.comprehension_engine import ComprehensionEngine
                ComprehensionEngine.synthesize_report(question.session_id, db)
            except Exception as e:
                logger.warning(f"Failed auto-synthesizing report upon viva completion: {e}")

    return assessment


@router.get("/session/{session_id}", response_model=List[VivaQuestionResponse])
def get_session_viva_questions(session_id: str, db: Session = Depends(get_db)):
    """Retrieve all viva questions and assessments for a session."""
    session = db.query(CodingSession).filter(CodingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Coding session not found")

    questions = (
        db.query(VivaQuestion)
        .filter(VivaQuestion.session_id == session_id)
        .order_by(VivaQuestion.created_at.asc())
        .all()
    )
    
    response = []
    for q in questions:
        has_ans = q.assessment is not None
        assessment_dict = None
        if has_ans and q.assessment:
            assessment_dict = AnswerAssessmentResponse.model_validate(q.assessment)
            
        q_resp = VivaQuestionResponse(
            id=q.id,
            session_id=q.session_id,
            submission_id=q.submission_id,
            question_text=q.question_text,
            focal_code=q.focal_code,
            concept=q.concept,
            difficulty=q.difficulty,
            expected_reasoning=q.expected_reasoning,
            rubric=q.rubric,
            misconception_indicators=q.misconception_indicators,
            created_at=q.created_at,
            has_answer=has_ans,
            assessment=assessment_dict
        )
        response.append(q_resp)

    return response
