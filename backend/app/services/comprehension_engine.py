"""ComprehensionEngine for multimodal evidence fusion and student misconception classification.

Ethical, multi-tiered evidence aggregation prioritizing code-grounded explanations and
verified program behavior. Coding-process telemetry is treated strictly as descriptive
context—never as an automated penalty or accusation of misconduct.
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import logging
from sqlalchemy.orm import Session
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.models.process_event import ProcessEvent
from app.models.comprehension_report import ComprehensionReport
from app.services.process_telemetry import ProcessTelemetryService

logger = logging.getLogger(__name__)


class ComprehensionEngine:
    """Fuses code structure, sandbox runtime traces, viva reasoning, and telemetry into a diagnosis."""

    # Weights: prioritize verbal conceptual reasoning and physical execution reality
    WEIGHT_AST = 0.15
    WEIGHT_EXEC = 0.25
    WEIGHT_VIVA = 0.45
    WEIGHT_PROC = 0.15

    @classmethod
    def evaluate_modalities(
        cls,
        latest_sub: Optional[CodeSubmission],
        assessments: List[AnswerAssessment],
        proc_features: Any,
    ) -> Tuple[Dict[str, float], float]:
        """Compute individual dimension scores and composite multimodal score.
        
        Zero punitive deductions for pasting or brief edit sessions. Process telemetry
        provides supportive context on problem-solving rhythm.
        """
        # 1. AST Structural Score (Evaluates syntactic validity & algorithmic constructs)
        if latest_sub:
            if latest_sub.is_valid:
                ast_sum = latest_sub.ast_summary or {}
                complexity = ast_sum.get("cyclomatic_complexity", 1)
                ast_score = 0.75
                if complexity >= 2:
                    ast_score += 0.10
                if ast_sum.get("loops") or ast_sum.get("functions"):
                    ast_score += 0.15
                ast_score = min(1.0, ast_score)
            else:
                # Syntax errors represent an active editing state, not a confirmed misconception
                ast_score = 0.40
        else:
            ast_score = 0.50

        # 2. Execution Fidelity Score (Measures sandbox runtime correctness)
        if latest_sub and latest_sub.execution_output:
            if latest_sub.error_output:
                exec_score = 0.45  # Runtime exception during active debugging
            else:
                exec_score = 1.0 if len(latest_sub.execution_output.strip()) > 0 else 0.85
        elif latest_sub and latest_sub.error_output:
            exec_score = 0.40
        else:
            exec_score = 0.60

        # 3. Viva Reasoning Score (Primary determinant of conceptual comprehension)
        if assessments:
            viva_scores = []
            for a in assessments:
                if a.answer_label == "Correct":
                    viva_scores.append(1.0)
                elif a.answer_label == "Partially Correct":
                    viva_scores.append(0.65)
                elif a.answer_label == "Insufficient Evidence":
                    viva_scores.append(0.40)
                else:  # Incorrect
                    viva_scores.append(0.15)
            viva_score = sum(viva_scores) / len(viva_scores)
        else:
            # When viva answers are not yet submitted, score is neutral/unverified
            viva_score = 0.50

        # 4. Process Rhythm Score (Neutral & Constructive - NO paste penalties)
        # Process telemetry reflects problem-solving habits (e.g. Iterative vs. Block Importer)
        # Pasting code is common in software engineering and is NEVER penalized automatically.
        recovery_rate = getattr(proc_features, "error_recovery_rate", 1.0)
        run_count = getattr(proc_features, "run_count", 0)
        
        proc_score = 0.85  # Neutral baseline
        if recovery_rate >= 0.80:
            proc_score = min(1.0, proc_score + 0.10)
        if run_count >= 2:
            proc_score = min(1.0, proc_score + 0.05)
        proc_score = max(0.50, proc_score)

        scores = {
            "ast_structural_score": round(ast_score, 2),
            "execution_fidelity_score": round(exec_score, 2),
            "viva_reasoning_score": round(viva_score, 2),
            "process_rhythm_score": round(proc_score, 2),
            "process_authenticity_score": round(proc_score, 2),  # Backward compatibility alias
        }

        composite = (
            cls.WEIGHT_AST * ast_score
            + cls.WEIGHT_EXEC * exec_score
            + cls.WEIGHT_VIVA * viva_score
            + cls.WEIGHT_PROC * proc_score
        )
        return scores, round(composite, 3)

    @classmethod
    def synthesize_report(cls, session_id: str, db: Session) -> ComprehensionReport:
        """Synthesize overall comprehension report fusing all multimodal signals transparently."""
        session = db.query(CodingSession).filter(CodingSession.id == session_id).first()
        if not session:
            raise ValueError(f"Coding session '{session_id}' not found")

        latest_sub = (
            db.query(CodeSubmission)
            .filter(CodeSubmission.session_id == session_id)
            .order_by(CodeSubmission.created_at.desc())
            .first()
        )
        questions = (
            db.query(VivaQuestion)
            .filter(VivaQuestion.session_id == session_id)
            .order_by(VivaQuestion.created_at.asc())
            .all()
        )
        assessments = (
            db.query(AnswerAssessment)
            .filter(AnswerAssessment.session_id == session_id)
            .all()
        )
        proc_events = (
            db.query(ProcessEvent)
            .filter(ProcessEvent.session_id == session_id)
            .order_by(ProcessEvent.timestamp.asc())
            .all()
        )

        proc_features = ProcessTelemetryService.compute_features(session_id, proc_events)
        scores, composite_score = cls.evaluate_modalities(latest_sub, assessments, proc_features)

        # Collect diagnosed misconceptions and contradictions
        detected_contradictions: List[str] = []
        detected_misconceptions: List[str] = []
        for a in assessments:
            if a.contradictions:
                for c in a.contradictions:
                    if c and c not in detected_contradictions:
                        detected_contradictions.append(c)
                        if c not in detected_misconceptions:
                            detected_misconceptions.append(c)
            if a.concepts_missed and a.answer_label in ("Incorrect", "Partially Correct"):
                for m in a.concepts_missed:
                    item = f"Missing: {m}"
                    if item not in detected_misconceptions:
                        detected_misconceptions.append(item)

        # Contradiction and review flags
        needs_review = False
        review_reasons = []

        # 1. Missing or insufficient viva evidence
        valid_answers = [
            a for a in assessments 
            if a.answer_label not in ("Insufficient Evidence",) 
            and a.student_answer and a.student_answer.strip()
        ]
        if not assessments or len(valid_answers) == 0:
            needs_review = True
            review_reasons.append("Insufficient viva evidence: Conceptual understanding has not been verbally verified or answers are empty.")

        # 2. Sophistication vs. Reasoning Gap (Complex code but student cannot explain it)
        if scores["ast_structural_score"] >= 0.85 and scores["viva_reasoning_score"] < 0.45:
            needs_review = True
            review_reasons.append("High code structural sophistication with low conceptual reasoning (potential borrowed or AI-generated solution).")

        # 3. Explicit review requests from evaluator uncertainty
        if any(a.needs_instructor_review for a in assessments):
            needs_review = True
            review_reasons.append("One or more viva answers triggered evaluator uncertainty or evasion flags.")

        # Count answer labels
        incorrect_vivas = sum(1 for a in assessments if a.answer_label == "Incorrect")
        correct_vivas = sum(1 for a in assessments if a.answer_label == "Correct")
        insufficient_vivas = sum(
            1 for a in assessments 
            if a.answer_label == "Insufficient Evidence" or not (a.student_answer and a.student_answer.strip())
        )

        # Classification Logic:
        # Rule 1: Missing or Insufficient Evidence -> Explicit Pending Status
        # Never prematurely assign a substantive classification without valid viva answers
        if len(assessments) == 0 or len(valid_answers) == 0:
            overall_label = "Pending / Insufficient Evidence"
            confidence = 0.35
            needs_review = True

        # Rule 2: Possible Misconception requires repeated errors or verified contradictions
        # A single incorrect answer alone or a syntax/execution crash NEVER determines a misconception.
        elif (
            incorrect_vivas >= 2
            or len(detected_contradictions) >= 2
            or (incorrect_vivas == 1 and len(detected_contradictions) >= 1)
        ):
            overall_label = "Possible Misconception"
            confidence = round(max(0.70, min(0.95, 0.60 + 0.15 * incorrect_vivas)), 2)

        # Rule 3: Understood requires solid composite score AND zero incorrect viva answers
        # A student who pastes code but explains it correctly is NOT penalized and can receive 'Understood'.
        elif composite_score >= 0.72 and incorrect_vivas == 0 and len(detected_contradictions) == 0 and len(valid_answers) > 0:
            overall_label = "Understood"
            confidence = round(max(0.75, min(0.98, composite_score)), 2)

        # Rule 4: Partial Understanding (Default balanced state)
        # Captures single error, partially correct answers, or fragile boundary knowledge
        else:
            overall_label = "Partial Understanding"
            confidence = round(max(0.60, min(0.85, composite_score + 0.05)), 2)

        # Generate Evidence Summary Narrative (descriptive, not punitive)
        summary_parts = []
        if overall_label == "Understood":
            summary_parts.append(
                f"Candidate exhibited robust conceptual comprehension (Composite Index: {int(composite_score*100)}%). "
                f"Code structure adheres to programmatic invariants with verified execution."
            )
            if assessments:
                summary_parts.append(
                    f"Viva explanations accurately justified loop boundaries, conditional branches, and state accumulation."
                )
        elif overall_label == "Possible Misconception":
            summary_parts.append(
                f"Definite conceptual gaps identified (Composite Index: {int(composite_score*100)}%). "
                f"The candidate's explanations contradict the operational reality of the executed code."
            )
            if detected_misconceptions:
                summary_parts.append(f"Specific identified flaws: {', '.join(detected_misconceptions[:3])}.")
        elif overall_label == "Pending / Insufficient Evidence":
            summary_parts.append(
                f"Candidate evaluation is pending sufficient viva explanation evidence (Composite Index: {int(composite_score*100)}%). "
                "Syntactic code snapshot is present, but verbal/conceptual justification has not been completed or is insufficient."
            )
        else:
            summary_parts.append(
                f"Candidate demonstrated partial comprehension (Composite Index: {int(composite_score*100)}%). "
                f"Foundational code structures are present, but nuanced edge-case handling or state transformations remain incomplete."
            )

        if proc_features.process_profile != "Uninitialized Session":
            summary_parts.append(
                f"Development telemetry reflected a '{proc_features.process_profile}' workflow ({proc_features.run_count} test runs)."
            )

        evidence_summary = " ".join(summary_parts)

        # Recommended pedagogical interventions
        interventions = []
        misconception_str = " ".join(detected_misconceptions).lower()
        if "loop-boundary" in misconception_str or "range" in misconception_str:
            interventions.append("Practice tracing iteration boundaries with manual trace tables on small inputs.")
        if "recursion" in misconception_str or "base case" in misconception_str:
            interventions.append("Draw recursive call-tree diagrams to visualize base case termination and stack return unwinding.")
        if proc_features.run_count == 0:
            interventions.append("Adopt an incremental test-driven workflow by running sandbox tests frequently after small code edits.")
        if not interventions:
            interventions.append("Advance to higher-complexity invariant testing and edge-case boundary verification exercises.")

        # Persist or update report
        existing_report = (
            db.query(ComprehensionReport)
            .filter(ComprehensionReport.session_id == session_id)
            .first()
        )

        rubric_summary = {
            "ast_weight": cls.WEIGHT_AST,
            "exec_weight": cls.WEIGHT_EXEC,
            "viva_weight": cls.WEIGHT_VIVA,
            "proc_weight": cls.WEIGHT_PROC,
            "total_questions": len(questions),
            "answered_questions": len(assessments),
            "correct_answers": correct_vivas,
            "incorrect_answers": incorrect_vivas,
            "insufficient_answers": insufficient_vivas,
        }

        evidence_breakdown = {
            "detected_misconceptions": detected_misconceptions,
            "detected_contradictions": detected_contradictions,
            "review_flags": review_reasons,
            "process_profile": proc_features.process_profile,
            "paste_ratio": round(proc_features.paste_ratio, 3),
            "error_recovery_rate": round(proc_features.error_recovery_rate, 3),
        }

        if existing_report:
            existing_report.overall_label = overall_label
            existing_report.confidence_score = confidence
            existing_report.evidence_summary = evidence_summary
            existing_report.detected_misconceptions = detected_misconceptions
            existing_report.evidence_breakdown = evidence_breakdown
            existing_report.multimodal_scores = scores
            existing_report.rubric_summary = rubric_summary
            existing_report.recommended_interventions = interventions
            existing_report.needs_instructor_review = needs_review
            db.commit()
            db.refresh(existing_report)
            return existing_report

        report = ComprehensionReport(
            session_id=session_id,
            overall_label=overall_label,
            confidence_score=confidence,
            evidence_summary=evidence_summary,
            detected_misconceptions=detected_misconceptions,
            evidence_breakdown=evidence_breakdown,
            multimodal_scores=scores,
            rubric_summary=rubric_summary,
            recommended_interventions=interventions,
            needs_instructor_review=needs_review,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        return report
