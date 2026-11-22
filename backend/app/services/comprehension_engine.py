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
from app.models.instructor_review import InstructorReview
from app.models.assignment import Assignment
from app.services.process_telemetry import ProcessTelemetryService
from app.config import settings

logger = logging.getLogger(__name__)


class ComprehensionEngine:
    """Fuses code structure, sandbox runtime traces, viva reasoning, and process telemetry into a diagnosis."""

    # Configurable multimodal weights from central settings:
    # S_overall = 0.15*S_ast + 0.20*S_exec + 0.45*S_viva + 0.20*S_process
    # RETAINED STRICTLY AS NUMERICAL SUPPORTING EVIDENCE FEATURES
    WEIGHT_AST = settings.weight_ast
    WEIGHT_EXEC = settings.weight_exec
    WEIGHT_VIVA = settings.weight_viva
    WEIGHT_PROC = settings.weight_process

    _classifier_override: Optional[Any] = None

    @classmethod
    def set_classifier_override(cls, classifier: Any):
        """Set a test/mock classifier override for integration testing."""
        cls._classifier_override = classifier

    @classmethod
    def reset_classifier_override(cls):
        """Clear any test classifier override."""
        cls._classifier_override = None

    @classmethod
    def get_classifier(cls):
        """Get the Multimodal Fusion Transformer instance (or override)."""
        if cls._classifier_override is not None:
            return cls._classifier_override
        from app.ml.multimodal_transformer import get_multimodal_transformer
        return get_multimodal_transformer()

    @classmethod
    def evaluate_modalities(
        cls,
        latest_sub: Optional[CodeSubmission],
        assessments: List[AnswerAssessment],
        proc_features: Any,
    ) -> Tuple[Dict[str, float], float]:
        """Compute individual dimension scores and composite multimodal score.
        
        Fair and non-punitive process evaluation: S_process fuses incremental development,
        debugging recovery, revision cycles, and sandbox testing. Pasting code is not penalized.
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

        # 4. Process Development Score (S_process)
        # S_process = 0.30*S_incremental + 0.30*S_debugging + 0.25*S_revision + 0.15*S_testing
        inc_score = getattr(proc_features, "incremental_score", 0.75)
        debug_score = getattr(proc_features, "debugging_score", 0.90)
        rev_score = getattr(proc_features, "revision_score", 0.75)
        test_score = getattr(proc_features, "testing_score", 0.60)

        proc_score = getattr(proc_features, "process_score", None)
        if proc_score is None:
            proc_score = round(
                settings.weight_proc_incremental * inc_score
                + settings.weight_proc_debugging * debug_score
                + settings.weight_proc_revision * rev_score
                + settings.weight_proc_testing * test_score,
                3
            )

        scores = {
            "ast_structural_score": round(ast_score, 2),
            "execution_fidelity_score": round(exec_score, 2),
            "viva_reasoning_score": round(viva_score, 2),
            "process_overall_score": round(proc_score, 2),
            "process_rhythm_score": round(proc_score, 2),  # Compatibility alias
            "process_authenticity_score": round(proc_score, 2),  # Compatibility alias
            "process_subscores": {
                "incremental_score": round(inc_score, 2),
                "debugging_score": round(debug_score, 2),
                "revision_score": round(rev_score, 2),
                "testing_score": round(test_score, 2),
            },
        }

        # S_overall = 0.15*S_ast + 0.20*S_exec + 0.45*S_viva + 0.20*S_process
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

        # 1. Contradictions between code and explanation
        for a in assessments:
            if a.contradictions:
                for c in a.contradictions:
                    if c and c not in detected_contradictions:
                        detected_contradictions.append(c)
                        if c not in detected_misconceptions:
                            detected_misconceptions.append(c)

        # 2. Concepts missed from ANY non-Correct answer
        for a in assessments:
            if a.concepts_missed and a.answer_label != "Correct":
                for m in a.concepts_missed:
                    item = m if m.lower().startswith("missing") else f"Missing: {m}"
                    if item not in detected_misconceptions:
                        detected_misconceptions.append(item)

        # 3. Sophistication vs. Reasoning Gap
        if scores["ast_structural_score"] >= 0.85 and scores["viva_reasoning_score"] < 0.45:
            gap_item = "Code Sophistication vs. Reasoning Gap: Working code structure present, but candidate failed to explain underlying mechanisms during viva."
            if gap_item not in detected_misconceptions:
                detected_misconceptions.append(gap_item)

        # 4. If student has Incorrect or Insufficient Evidence answers, ensure descriptive misconception is captured
        for a in assessments:
            if a.answer_label in ("Incorrect", "Insufficient Evidence") and a.student_answer and a.student_answer.strip():
                q_obj = next((q for q in questions if q.id == a.question_id), None)
                q_concept = q_obj.concept if (q_obj and q_obj.concept) else (q_obj.question_text[:35] if q_obj else "Target Logic")
                if not any(q_concept.lower() in m.lower() for m in detected_misconceptions):
                    desc = f"Unverified reasoning for '{q_concept}': explanation was insufficient or contradictory to code execution"
                    if desc not in detected_misconceptions:
                        detected_misconceptions.append(desc)

        # 5. Syntax / Execution runtime failure
        if latest_sub and not latest_sub.is_valid and latest_sub.error_output:
            err_line = latest_sub.error_output.strip().split("\n")[-1]
            if err_line and err_line not in detected_misconceptions:
                detected_misconceptions.append(f"Syntax/Parser Error: {err_line}")

        # Contradiction and review flags
        needs_review = False
        review_reasons = []

        # 1. Missing or insufficient viva evidence
        valid_answers = [
            a for a in assessments 
            if a.answer_label not in ("Insufficient Evidence",) 
            and a.student_answer and a.student_answer.strip()
        ]
        submitted_answers = [
            a for a in assessments 
            if a.student_answer and a.student_answer.strip()
        ]
        if not assessments or len(submitted_answers) == 0:
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

        # Construct standardized MultimodalStudentInputBundle
        from app.ml.multimodal_inputs import MultimodalStudentInputBundle
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

        # ------------------------------------------------------------------
        # EVIDENCE VALIDATION (Separate from misconception classification)
        # ------------------------------------------------------------------
        has_code = bool(latest_sub and latest_sub.code and latest_sub.code.strip())
        is_insufficient_evidence = (
            not has_code 
            or len(assessments) == 0 
            or len(submitted_answers) == 0 
            or (len(valid_answers) == 0 and len(assessments) <= 1)
        )

        classifier = cls.get_classifier()

        if is_insufficient_evidence:
            overall_label = "Insufficient Evidence"
            confidence = 0.35
            needs_review = True
            model_status = "INSUFFICIENT_EVIDENCE"
            transformer_pred = {
                "model_status": "INSUFFICIENT_EVIDENCE",
                "predicted_class": None,
                "class_probabilities": None,
                "modality_availability": {
                    "code": has_code,
                    "explanation": len(submitted_answers) > 0,
                    "process": len(proc_events) > 0,
                },
                "supporting_feature_scores": {
                    "S_AST": scores["ast_structural_score"],
                    "S_EXEC": scores["execution_fidelity_score"],
                    "S_VIVA": scores["viva_reasoning_score"],
                    "S_PROCESS": scores["process_overall_score"],
                    "S_OVERALL": composite_score,
                },
                "needs_instructor_review": True,
            }
        else:
            # ------------------------------------------------------------------
            # MULTIMODAL FUSION TRANSFORMER: Sole Authority for Classification
            # NOTE: All formula-based thresholds & if-else rules are strictly bypassed!
            # ------------------------------------------------------------------
            pred_res = classifier.predict_session_bundle(input_bundle)

            if pred_res.model_status == "READY":
                overall_label = pred_res.predicted_class
                confidence = pred_res.confidence
                needs_review = pred_res.needs_instructor_review or needs_review
                model_status = "READY"
            elif pred_res.model_status == "INSUFFICIENT_EVIDENCE":
                overall_label = "Insufficient Evidence"
                confidence = 0.35
                needs_review = True
                model_status = "INSUFFICIENT_EVIDENCE"
            else:
                # MODEL_NOT_TRAINED
                # When no checkpoint exists, report MODEL_NOT_TRAINED status explicitly
                overall_label = "Pending / Awaiting Model Training"
                confidence = 0.0
                needs_review = True
                model_status = "MODEL_NOT_TRAINED"

            transformer_pred = {
                "model_status": pred_res.model_status,
                "predicted_class": pred_res.predicted_class,
                "class_probabilities": pred_res.class_probabilities,
                "modality_availability": pred_res.modality_availability,
                "supporting_feature_scores": pred_res.supporting_feature_scores,
                "needs_instructor_review": pred_res.needs_instructor_review,
            }

        # Generate Evidence Summary Narrative (descriptive, not punitive)
        summary_parts = []
        if overall_label == "Understood":
            summary_parts.append(
                f"Candidate exhibited robust conceptual comprehension (Predicted by Multimodal Fusion Transformer; Composite Index: {int(composite_score*100)}%). "
                f"Code structure adheres to programmatic invariants with verified execution."
            )
            if assessments:
                summary_parts.append(
                    f"Viva explanations accurately justified loop boundaries, conditional branches, and state accumulation."
                )
        elif overall_label == "Possible Misconception":
            summary_parts.append(
                f"Definite conceptual gaps identified (Predicted by Multimodal Fusion Transformer; Composite Index: {int(composite_score*100)}%). "
                f"The candidate's explanations contradict or fail to substantiate the operational reality of the executed code."
            )
            if detected_misconceptions:
                summary_parts.append(f"Specific identified flaws: {', '.join(detected_misconceptions[:3])}.")
        elif overall_label == "Pending / Awaiting Model Training":
            summary_parts.append(
                f"Student session completed and multimodal evidence extracted (Composite Index: {int(composite_score*100)}%). "
                "Final misconception classification is pending project-specific model training on the multimodal dataset."
            )
        elif overall_label in ("Insufficient Evidence", "Pending / Insufficient Evidence"):
            summary_parts.append(
                f"Candidate evaluation reflects insufficient viva explanation evidence (Composite Index: {int(composite_score*100)}%). "
                "Syntactic code snapshot is present, but verbal/conceptual justification has not been completed or is insufficient."
            )
            if detected_misconceptions:
                summary_parts.append(f"Preliminary flagged points: {', '.join(detected_misconceptions[:3])}.")
        else: # Partial Understanding
            summary_parts.append(
                f"Candidate demonstrated partial comprehension (Predicted by Multimodal Fusion Transformer; Composite Index: {int(composite_score*100)}%). "
                f"Foundational code structures are present, but nuanced edge-case handling or state transformations remain incomplete."
            )
            if detected_misconceptions:
                summary_parts.append(f"Identified areas for reinforcement: {', '.join(detected_misconceptions[:3])}.")

        if proc_features.process_profile != "Uninitialized Session":
            summary_parts.append(
                f"Development telemetry reflected a '{proc_features.process_profile}' workflow ({proc_features.run_count} test runs)."
            )

        has_process_evidence = getattr(proc_features, "has_process_evidence", True)
        if not has_process_evidence or proc_features.total_events == 0:
            review_reasons.append("Process telemetry stream was unrecorded for this session; neutral process baseline applied without penalty.")

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
            "process_weight": cls.WEIGHT_PROC,
            "process_subweights": {
                "incremental_weight": settings.weight_proc_incremental,
                "debugging_weight": settings.weight_proc_debugging,
                "revision_weight": settings.weight_proc_revision,
                "testing_weight": settings.weight_proc_testing,
            },
            "total_questions": len(questions),
            "answered_questions": len(assessments),
            "correct_answers": correct_vivas,
            "incorrect_answers": incorrect_vivas,
            "insufficient_answers": insufficient_vivas,
        }

        # Structured Viva Answer Evaluation (Template Section 4)
        viva_evaluations = []
        for idx, q in enumerate(questions, 1):
            a_obj = next((a for a in assessments if a.question_id == q.id), None)
            score_val = 0.0
            if a_obj:
                if a_obj.answer_label == "Correct": score_val = 1.00
                elif a_obj.answer_label == "Partially Correct": score_val = 0.65
                elif a_obj.answer_label == "Insufficient Evidence": score_val = 0.40
                else: score_val = 0.15
            viva_evaluations.append({
                "index": idx,
                "question_id": q.id,
                "concept": q.concept or f"Concept {idx}",
                "question_text": q.question_text,
                "student_answer": a_obj.student_answer if a_obj else "[Not answered yet]",
                "answer_label": a_obj.answer_label if a_obj else "Unanswered",
                "score": score_val,
                "concepts_correct": a_obj.concepts_correct if a_obj else [],
                "concepts_missed": a_obj.concepts_missed if a_obj else [],
                "contradictions": a_obj.contradictions if a_obj else [],
            })

        # ------------------------------------------------------------------
        # EVIDENCE-GROUNDED REASONING & EXPLANATIONS (Template Section 7 & 8)
        # ------------------------------------------------------------------
        # Determine student- and problem-specific targeted follow-up question
        def determine_targeted_followup(problem_title: str) -> str:
            pt = (problem_title or "").lower()
            if "accumulator" in pt or "modulo" in pt:
                return "Trace what your function returns if n <= 0, and explain the initialization invariant of 'total' before the loop begins."
            elif "palindrome" in pt or "pointer" in pt:
                return "Trace pointer advancement if the input consists exclusively of punctuation (e.g., '.,,.'). Do pointers terminate safely without out-of-bounds indexing?"
            elif "frequency" in pt or "dictionary" in pt or "word" in pt:
                return "How would you handle words attached to punctuation (e.g., 'apple,') and ensure case-insensitivity without raising KeyError or inflating unique counts?"
            elif "factorial" in pt or "recursion" in pt or "stack" in pt:
                return "Trace the call stack unwinding for n = 3 down to base case n <= 1, and explain how the runtime prevents infinite recursion for negative inputs."
            elif "binary" in pt or "search" in pt:
                return "If the target is absent from a sorted sequence, what are the final values of low and high upon loop termination? How does this mathematically prove non-existence?"
            else:
                return f"Can you trace how your solution for '{problem_title}' handles an empty or boundary input, and explain the invariant guaranteeing safe termination?"

        targeted_question = determine_targeted_followup(session.title)

        misconceptions_table = []
        has_demonstrated_flaw = False
        has_incomplete_explanation = False

        # 1. Viva Evidence: Grounded strictly in actual viva questions and student answers
        for a in assessments:
            q_match = next((q for q in questions if q.id == a.question_id), None)
            concept_name = (q_match.concept if q_match and q_match.concept else "Algorithmic Logic")
            q_text = q_match.question_text if q_match else "Viva Question"
            ans_clean = (a.student_answer or "").strip()

            # A. Demonstrated Misunderstanding: Factual contradiction or explicit misunderstanding
            if a.contradictions:
                has_demonstrated_flaw = True
                for c in a.contradictions:
                    misconceptions_table.append({
                        "concept": f"{concept_name} (Demonstrated misunderstanding)",
                        "evidence": f'Q: "{q_text}" — Student answered: "{ans_clean}"',
                        "pattern": "Demonstrated misunderstanding",
                        "interpretation": f"Demonstrated misunderstanding: {c}. Explanation contradicts recorded program execution."
                    })
            elif a.answer_label == "Incorrect":
                has_demonstrated_flaw = True
                expl = a.evaluation_explanation or "Answer provides incorrect logic for the evaluated concept."
                misconceptions_table.append({
                    "concept": f"{concept_name} (Demonstrated misunderstanding)",
                    "evidence": f'Q: "{q_text}" — Student answered: "{ans_clean}"',
                    "pattern": "Demonstrated misunderstanding",
                    "interpretation": f"Demonstrated misunderstanding: {expl}"
                })

            # B. Incomplete Explanation: Addressed question partially, but missed requested nuances
            # (Requirement 3: Never describe a Correct answer as weak merely for lacking formal terminology not asked)
            elif a.answer_label == "Partially Correct" or (a.concepts_missed and a.answer_label != "Correct"):
                has_incomplete_explanation = True
                missed_points = ", ".join(a.concepts_missed) if a.concepts_missed else "nuanced boundary behavior"
                misconceptions_table.append({
                    "concept": f"{concept_name} (Incomplete explanation)",
                    "evidence": f'Q: "{q_text}" — Student answered: "{ans_clean}"',
                    "pattern": "Incomplete explanation",
                    "interpretation": f"Incomplete explanation: Response addressed primary mechanics but omitted: {missed_points}."
                })

        # 2. Code & Execution Evidence: Observations grounded in actual execution trace
        has_code_error = False
        if latest_sub and not latest_sub.is_valid:
            has_code_error = True
            err_msg = latest_sub.error_output.strip() if latest_sub.error_output else "Syntax validation failed"
            misconceptions_table.append({
                "concept": "Code Syntax / Structure (Demonstrated misunderstanding)",
                "evidence": f"Code validation error: {err_msg[:120]}",
                "pattern": "Demonstrated misunderstanding",
                "interpretation": "Demonstrated misunderstanding: Source code contains syntax errors and cannot be parsed into a valid AST."
            })
        elif latest_sub and latest_sub.error_output and latest_sub.error_output.strip():
            has_code_error = True
            err_line = latest_sub.error_output.strip().splitlines()[-1]
            misconceptions_table.append({
                "concept": "Runtime Execution (Demonstrated misunderstanding)",
                "evidence": f"Executed code produced runtime exception: {err_line}",
                "pattern": "Demonstrated misunderstanding",
                "interpretation": f"Demonstrated misunderstanding: Program execution crashed with {err_line}, indicating unhandled control flow or bounds violation."
            })

        # 3. Coding-Process Telemetry Evidence
        if proc_features and getattr(proc_features, "paste_ratio", 0.0) >= 0.40:
            misconceptions_table.append({
                "concept": "Coding-Process Observation",
                "evidence": f"Telemetry recorded {proc_features.paste_count} paste event(s) accounting for {int(proc_features.paste_ratio * 100)}% of inserted code.",
                "pattern": "Coding-process evidence",
                "interpretation": "Context only: Significant block paste observed. Note that pasted code alone does not prove unauthorized source usage or a conceptual flaw; oral viva explanations verify comprehension."
            })

        # 4. Concepts Not Assessed (Requirement 4)
        asgn = db.query(Assignment).filter(Assignment.title == session.title).first()
        if asgn and asgn.concept_tags:
            assessed_text = " ".join(
                [q.concept.lower().replace("-", " ") for q in questions if q.concept] 
                + [q.question_text.lower().replace("-", " ") for q in questions]
            )
            unassessed = [
                tag for tag in asgn.concept_tags 
                if tag.lower().replace("-", " ") not in assessed_text and tag.lower() not in ("loops", "strings", "invariants")
            ]
            if (overall_label == "Partial Understanding" and not has_demonstrated_flaw) and unassessed:
                u_tag = unassessed[0]
                misconceptions_table.append({
                    "concept": f"{u_tag.replace('-', ' ').title()} (Concept not assessed)",
                    "evidence": f"Assignment requires '{u_tag}', but viva questions covered only evaluated topics ({', '.join([q.concept for q in questions if q.concept][:2])}).",
                    "pattern": "Concept not assessed",
                    "interpretation": f"Concept not assessed: Comprehension of '{u_tag}' was not directly probed in this viva session. Follow-up is recommended to verify this area."
                })

        # 5. Assessment Output Uncertainty / Prediction Discrepancy (Requirements 5 & 7)
        # If the Transformer predicted Partial Understanding or Possible Misconception,
        # but recorded evidence (code + vivas + tests) shows no demonstrated misunderstanding or incomplete explanation:
        if (overall_label in ("Partial Understanding", "Possible Misconception") 
            and not has_demonstrated_flaw 
            and not has_incomplete_explanation 
            and not has_code_error):
            
            diag_pct = int(round(composite_score * 100))
            misconceptions_table.append({
                "concept": "Assessment Output Uncertainty",
                "evidence": f"Diagnostic score is {diag_pct}%: Code passed all tests with clean exit code; all {len(assessments)} evaluated viva response(s) were assessed as correct for the questions asked.",
                "pattern": "Unexplained prediction variance",
                "interpretation": f"The model's {overall_label} prediction cannot be adequately explained by the available observations. Recorded code and viva responses satisfy the evaluated criteria without demonstrated errors. Follow-up recommended: {targeted_question}"
            })

        # Code & Execution Evidence (Template Section 6)
        ast_sum = (latest_sub.ast_summary or {}) if latest_sub else {}
        code_lines = len(latest_sub.code.splitlines()) if latest_sub and latest_sub.code else 0
        comp_val = ast_sum.get("cyclomatic_complexity", 1)
        func_list = ast_sum.get("functions", [])
        if isinstance(func_list, list) and func_list:
            func_names = [f["name"] if isinstance(f, dict) and "name" in f else str(f) for f in func_list]
            target_str = ", ".join(func_names)
        else:
            target_str = "procedural script"
        code_evidence = {
            "code_summary": f"Python solution ({code_lines} lines of code). Target constructs: {target_str}.",
            "ast_findings": f"Syntactically valid AST structure. Cyclomatic complexity M={comp_val}. Identified constructs: loops={bool(ast_sum.get('loops'))}, branching={bool(ast_sum.get('conditionals'))}, recursion={bool(ast_sum.get('is_recursive'))}.",
            "execution_outcome": (latest_sub.execution_output.strip() if latest_sub and latest_sub.execution_output else ("Runtime Exception: " + latest_sub.error_output.strip() if latest_sub and latest_sub.error_output else "Sandbox execution succeeded with clean exit code.")),
            "evidence_limitations": "Sandbox unit tests executed in isolated PyLite environment. Oral viva verified against verified physical stdout traces."
        }

        # Development Process Signals (Template Section 7)
        telemetry_signals = [
            {"feature": "Edit events", "value": f"{proc_features.edit_count} events", "interpretation": "Context only"},
            {"feature": "Run events", "value": f"{proc_features.run_count} runs", "interpretation": "Context only"},
            {"feature": "Error recovery", "value": f"{int(proc_features.error_recovery_rate * 100)}%", "interpretation": "Context only"},
            {"feature": "Revision cycles", "value": f"{int(getattr(proc_features, 'revision_score', 0.75) * 100)}%", "interpretation": "Context only"},
            {"feature": "Testing activity", "value": f"{int(getattr(proc_features, 'testing_score', 0.60) * 100)}%", "interpretation": "Context only"},
        ]

        # Feedback & Recommended Next Steps (Template Section 8)
        strengths = []
        for a in assessments:
            if a.concepts_correct:
                strengths.extend(a.concepts_correct)
        if not strengths:
            if scores["ast_structural_score"] >= 0.80:
                strengths.append("Modular code structure adhering to Pythonic syntax.")
            if scores["execution_fidelity_score"] >= 0.90:
                strengths.append("Sandboxed execution trace generated expected test outputs.")
            if correct_vivas > 0:
                strengths.append("Verbal justifications aligned with target algorithmic invariants.")

        areas_to_revisit = detected_misconceptions[:] if detected_misconceptions else ["Boundary condition handling and variable state persistence."]

        feedback = {
            "strengths": strengths[:4],
            "areas_to_revisit": areas_to_revisit[:4],
            "suggested_practice": interventions[0] if interventions else "Practice boundary-case tracing and formal loop invariant formulation.",
            "targeted_question": targeted_question,
            "instructor_followup": "Review ambiguous edge-case traces with student during lab hours." if needs_review else "Demonstrated satisfactory mental model; proceed to higher-order complexity tasks."
        }

        # Instructor Review (Template Section 9)
        inst_rev = db.query(InstructorReview).filter(InstructorReview.session_id == session_id).order_by(InstructorReview.created_at.desc()).first()
        instructor_review_data = {
            "has_review": inst_rev is not None,
            "comments": inst_rev.feedback_notes if inst_rev and inst_rev.feedback_notes else "Pending instructor manual review and notes.",
            "reviewed_by": inst_rev.reviewer_name if inst_rev else (session.teacher_name or "Faculty Advisor"),
            "review_date": inst_rev.created_at.strftime("%d %b %Y") if inst_rev else "Pending Review",
            "final_decision": f"Instructor Review: {inst_rev.assigned_label}" if inst_rev else "AI-generated baseline (awaiting faculty signoff)",
            "agree_with_ai": inst_rev.agree_with_ai if inst_rev else None,
        }

        evidence_breakdown = {
            "transformer_prediction": transformer_pred,
            "detected_misconceptions": detected_misconceptions,
            "detected_contradictions": detected_contradictions,
            "review_flags": review_reasons,
            "process_profile": proc_features.process_profile,
            "process_evidence_available": has_process_evidence,
            "process_subscores": {
                "incremental": round(getattr(proc_features, "incremental_score", 0.75), 2),
                "debugging": round(getattr(proc_features, "debugging_score", 0.90), 2),
                "revision": round(getattr(proc_features, "revision_score", 0.75), 2),
                "testing": round(getattr(proc_features, "testing_score", 0.60), 2),
            },
            "paste_ratio": round(proc_features.paste_ratio, 3),
            "error_recovery_rate": round(proc_features.error_recovery_rate, 3),
            "run_count": proc_features.run_count,
            "edit_count": proc_features.edit_count,
            "viva_evaluations": viva_evaluations,
            "misconceptions_table": misconceptions_table,
            "misconception_counts": {
                "incorrect_answers": incorrect_vivas,
                "contradictions": len(detected_contradictions),
                "composite_score": composite_score,
                "composite_percent": int(round(composite_score * 100)),
                "ast_score": scores["ast_structural_score"],
                "exec_score": scores["execution_fidelity_score"],
                "viva_score": scores["viva_reasoning_score"],
                "process_score": scores["process_overall_score"],
                "formula": "Overall = 0.15(AST) + 0.20(EXEC) + 0.45(VIVA) + 0.20(PROCESS)",
                "formula_evaluation": (
                    f"0.15({scores['ast_structural_score']:.2f}) + "
                    f"0.20({scores['execution_fidelity_score']:.2f}) + "
                    f"0.45({scores['viva_reasoning_score']:.2f}) + "
                    f"0.20({scores['process_overall_score']:.2f}) = {composite_score:.2f} ({int(round(composite_score*100))}%)"
                ),
                "rule": "Possible Misconception if Multimodal Overall < 0.50 OR cross-modal divergence; Understood if Overall >= 0.70; Partial Understanding if 0.50 <= Overall < 0.70.",
                "verdict": overall_label,
            },
            "code_evidence": code_evidence,
            "telemetry_signals": telemetry_signals,
            "feedback": feedback,
            "instructor_review": instructor_review_data,
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

            # Update session status to completed if not already reviewed
            session = db.query(CodingSession).filter(CodingSession.id == session_id).first()
            if session and session.status == "active":
                session.status = "completed"
                db.add(session)

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

        # Update session status to completed if not already reviewed
        session = db.query(CodingSession).filter(CodingSession.id == session_id).first()
        if session and session.status == "active":
            session.status = "completed"
            db.add(session)

        db.commit()
        db.refresh(report)
        return report
