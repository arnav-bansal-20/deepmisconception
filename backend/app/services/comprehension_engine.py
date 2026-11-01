"""ComprehensionEngine for multimodal evidence fusion and student misconception classification."""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.models.process_event import ProcessEvent
from app.models.comprehension_report import ComprehensionReport
from app.services.process_telemetry import ProcessTelemetryService


class ComprehensionEngine:
    """Fuses code structure, sandbox runtime traces, viva reasoning, and telemetry into a diagnosis."""

    WEIGHT_AST = 0.20
    WEIGHT_EXEC = 0.25
    WEIGHT_VIVA = 0.40
    WEIGHT_PROC = 0.15

    @classmethod
    def evaluate_modalities(
        cls,
        latest_sub: Optional[CodeSubmission],
        assessments: List[AnswerAssessment],
        proc_features: Any,
    ) -> Tuple[Dict[str, float], float]:
        """Compute individual dimension scores and composite multimodal score."""
        # 1. AST Structural Score
        if latest_sub:
            if latest_sub.is_valid:
                ast_sum = latest_sub.ast_summary or {}
                complexity = ast_sum.get("cyclomatic_complexity", 1)
                ast_score = 0.70
                if complexity >= 2:
                    ast_score += 0.15
                if ast_sum.get("loops") or ast_sum.get("functions"):
                    ast_score += 0.15
                ast_score = min(1.0, ast_score)
            else:
                ast_score = 0.20  # Syntax error
        else:
            ast_score = 0.50

        # 2. Execution Fidelity Score
        if latest_sub and latest_sub.execution_output:
            if latest_sub.error_output:
                exec_score = 0.30
            else:
                exec_score = 1.0 if len(latest_sub.execution_output.strip()) > 0 else 0.85
        elif latest_sub and latest_sub.error_output:
            exec_score = 0.20
        else:
            exec_score = 0.60

        # 3. Viva Reasoning Score
        if assessments:
            viva_scores = []
            for a in assessments:
                if a.answer_label == "Correct":
                    viva_scores.append(1.0)
                elif a.answer_label == "Partially Correct":
                    viva_scores.append(0.65)
                elif a.answer_label == "Insufficient Evidence":
                    viva_scores.append(0.30)
                else:  # Incorrect
                    viva_scores.append(0.0)
            viva_score = sum(viva_scores) / len(viva_scores)
        else:
            viva_score = 0.50

        # 4. Process Authenticity Score
        paste_ratio = getattr(proc_features, "paste_ratio", 0.0)
        recovery_rate = getattr(proc_features, "error_recovery_rate", 1.0)
        proc_score = 1.0
        if paste_ratio > 0.70:
            proc_score -= 0.35
        elif paste_ratio > 0.40:
            proc_score -= 0.15
        if recovery_rate >= 0.80:
            proc_score = min(1.0, proc_score + 0.10)
        proc_score = max(0.20, proc_score)

        scores = {
            "ast_structural_score": round(ast_score, 2),
            "execution_fidelity_score": round(exec_score, 2),
            "viva_reasoning_score": round(viva_score, 2),
            "process_authenticity_score": round(proc_score, 2),
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
        """Synthesize overall comprehension report fusing all multimodal signals."""
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

        # Collect diagnosed misconceptions
        detected_misconceptions: List[str] = []
        for a in assessments:
            if a.contradictions:
                for c in a.contradictions:
                    if c and c not in detected_misconceptions:
                        detected_misconceptions.append(c)
            if a.concepts_missed and a.answer_label in ("Incorrect", "Partially Correct"):
                for m in a.concepts_missed:
                    item = f"Missing: {m}"
                    if item not in detected_misconceptions:
                        detected_misconceptions.append(item)

        # Contradiction and review flags
        needs_review = False
        review_reasons = []

        # Conflict 1: High AST/Code + Low Viva (potential copypaste without understanding)
        if scores["ast_structural_score"] >= 0.85 and scores["viva_reasoning_score"] < 0.40:
            needs_review = True
            review_reasons.append("High code structural sophistication with low viva conceptual reasoning.")

        # Conflict 2: High paste ratio with incorrect viva explanations
        if proc_features.paste_ratio > 0.60 and scores["viva_reasoning_score"] < 0.50:
            needs_review = True
            review_reasons.append("High paste block volume coupled with ambiguous explanation.")

        # Conflict 3: Any assessment explicitly marked needs_instructor_review
        if any(a.needs_instructor_review for a in assessments):
            needs_review = True
            review_reasons.append("One or more viva answers triggered uncertainty flags.")

        # Final Classification Decision
        incorrect_vivas = sum(1 for a in assessments if a.answer_label == "Incorrect")
        if incorrect_vivas >= 2 or len(detected_misconceptions) >= 3 or composite_score < 0.45:
            overall_label = "Possible Misconception"
            confidence = round(max(0.65, min(0.95, 1.0 - (composite_score / 2.0))), 2)
        elif composite_score >= 0.75 and incorrect_vivas == 0:
            overall_label = "Understood"
            confidence = round(max(0.75, min(0.98, composite_score)), 2)
        else:
            overall_label = "Partial Understanding"
            confidence = round(max(0.60, min(0.88, composite_score + 0.1)), 2)

        # Generate Evidence Summary Narrative
        summary_parts = []
        if overall_label == "Understood":
            summary_parts.append(
                f"Candidate exhibited robust conceptual comprehension (Composite Index: {int(composite_score*100)}%). "
                f"Code structure adheres to programmatic invariants with clean sandbox execution."
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
        else:
            summary_parts.append(
                f"Candidate demonstrated partial comprehension (Composite Index: {int(composite_score*100)}%). "
                f"Foundational code structures are present, but nuanced edge-case handling or state transformations remain incomplete."
            )

        if proc_features.process_profile != "Uninitialized Session":
            summary_parts.append(
                f"Development telemetry reflected a '{proc_features.process_profile}' workflow (paste fraction: {int(proc_features.paste_ratio*100)}%, {proc_features.run_count} test runs)."
            )

        evidence_summary = " ".join(summary_parts)

        # Recommended pedagogical interventions
        interventions = []
        if "loop-boundary" in " ".join(detected_misconceptions).lower():
            interventions.append("Practice tracing iteration boundaries with manual trace tables on small inputs.")
        if "recursion" in " ".join(detected_misconceptions).lower():
            interventions.append("Draw recursive call-tree diagrams to visualize base case termination and stack return unwinding.")
        if proc_features.run_count == 0:
            interventions.append("Adopt an incremental test-driven workflow by running sandbox tests frequently after small code edits.")
        if not interventions:
            interventions.append("Advance to higher-complexity invariant testing and edge-case boundary verification exercises.")

        # Rubric summary
        rubric_summary = {
            "questions_total": len(questions),
            "questions_evaluated": len(assessments),
            "correct_count": sum(1 for a in assessments if a.answer_label == "Correct"),
            "partially_correct_count": sum(1 for a in assessments if a.answer_label == "Partially Correct"),
            "incorrect_count": sum(1 for a in assessments if a.answer_label == "Incorrect"),
            "insufficient_evidence_count": sum(1 for a in assessments if a.answer_label == "Insufficient Evidence"),
            "composite_score": composite_score,
            "review_reasons": review_reasons,
        }

        evidence_breakdown = {
            "scores": scores,
            "composite_score": composite_score,
            "review_flags": review_reasons,
            "ast_complexity": latest_sub.ast_summary.get("cyclomatic_complexity", 1) if (latest_sub and latest_sub.ast_summary) else 1,
            "execution_status": "Clean (Exit 0)" if (latest_sub and not latest_sub.error_output) else "Execution Errors",
            "telemetry_profile": proc_features.process_profile,
        }

        # Check if an existing report for this session already exists; update it or create new
        report = (
            db.query(ComprehensionReport)
            .filter(ComprehensionReport.session_id == session_id)
            .order_by(ComprehensionReport.created_at.desc())
            .first()
        )
        if not report:
            report = ComprehensionReport(
                session_id=session_id,
                overall_label=overall_label,
                confidence_score=confidence,
                evidence_summary=evidence_summary,
                evidence_breakdown=evidence_breakdown,
                detected_misconceptions=detected_misconceptions,
                process_features=proc_features.model_dump(),
                rubric_summary=rubric_summary,
                multimodal_scores=scores,
                recommended_interventions=interventions,
                needs_instructor_review=needs_review,
            )
            db.add(report)
        else:
            report.overall_label = overall_label
            report.confidence_score = confidence
            report.evidence_summary = evidence_summary
            report.evidence_breakdown = evidence_breakdown
            report.detected_misconceptions = detected_misconceptions
            report.process_features = proc_features.model_dump()
            report.rubric_summary = rubric_summary
            report.multimodal_scores = scores
            report.recommended_interventions = interventions
            report.needs_instructor_review = needs_review

        # Update session status
        session.status = "completed"

        db.commit()
        db.refresh(report)
        return report
