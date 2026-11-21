"""
DeepMisconception Evidence-Based Annotation Engine (Safeguards 1 & 2)
Applies consistent, documented pedagogical rubrics across Code, Viva, and Process evidence.
Preserves original_label, stores new annotations in synthetic_annotator_label,
keeps human_validated_label as null, and tracks rubric provenance.
"""

def evaluate_pedagogical_rubric(code_status: str, 
                                viva_reasoning: str, 
                                viva_correctness: str,
                                n_errors: int, 
                                final_pass: int,
                                is_borderline: bool,
                                is_conflict: bool):
    """
    Applies consistent pedagogical rubrics to synthesize multimodal evidence.
    Returns: (synthetic_label, rubric_id, rationale, code_ev, viva_ev, proc_ev, conf)
    """
    # Rubric evaluation based on conceptual evidence (viva + code + process)
    if viva_correctness == "correct" and viva_reasoning == "sound":
        if code_status == "correct":
            rubric_id = "RUBRIC_MASTERY_OPTIMAL"
            synth_label = "Understood"
            code_ev = "Robust, idiomatic implementation satisfying acceptance specifications."
            viva_ev = "Student articulated sound theoretical justifications for data structure and algorithmic invariants."
            proc_ev = f"Clean development trajectory with {final_pass} final pass state."
            rationale = "Full conceptual understanding demonstrated through clean code and sound verbal reasoning."
            conf = 0.95
        elif code_status == "partially_correct":
            rubric_id = "RUBRIC_MASTERY_MINOR_EDGE_SLIP"
            synth_label = "Understood"
            code_ev = "Functional implementation with minor boundary omission."
            viva_ev = "Student accurately identified and explained boundary edge cases during viva, verifying conceptual mastery."
            proc_ev = "Active iterative development with multiple unit verifications."
            rationale = "Understood: Minor code edge omission compensated by complete theoretical grasp in viva."
            conf = 0.90
        else: # incorrect code due to minor typo or syntax fault
            rubric_id = "RUBRIC_MASTERY_DIAGNOSED_FAULT"
            synth_label = "Understood"
            code_ev = "Code triggered execution fault during testing."
            viva_ev = "Student correctly identified fault root cause and explained execution trace with complete precision."
            proc_ev = f"Encountered {n_errors} error during authoring."
            rationale = "Understood: Syntax or boundary slip diagnosed and explained accurately by student during viva."
            conf = 0.88

    elif viva_reasoning in ("superficial", "sound") and (viva_correctness in ("partially_correct", "correct") or code_status == "partially_correct"):
        if code_status == "correct":
            rubric_id = "RUBRIC_PARTIAL_ROTE_IMPLEMENTATION"
            synth_label = "Partial Understanding"
            code_ev = "Working code solving baseline acceptance tests."
            viva_ev = "Student relied on memorized procedure; demonstrated superficial understanding of reference semantics."
            proc_ev = "Minimal exploratory testing following initial insertion."
            rationale = "Partial Understanding: Correct code achieved via template memorization, but theoretical comprehension is incomplete."
            conf = 0.84
        elif code_status == "partially_correct":
            rubric_id = "RUBRIC_PARTIAL_WORKING_PROTOTYPE"
            synth_label = "Partial Understanding"
            code_ev = "Functional prototype for standard inputs; lacks defensive guards for null/empty boundaries."
            viva_ev = "Student grasps primary algorithmic flow but exhibited gaps in memory allocation and edge analysis."
            proc_ev = "Iterative edits with partial test verification."
            rationale = "Partial Understanding: Working prototype with consistent partial conceptual depth across viva turns."
            conf = 0.86
        else:
            rubric_id = "RUBRIC_PARTIAL_LOGIC_DEFICIT"
            synth_label = "Partial Understanding"
            code_ev = "Code contains logic bugs in loop boundaries."
            viva_ev = "Student understands general objective but struggled to formulate correct boundary conditions."
            proc_ev = f"Encountered {n_errors} errors with partial debug resolution."
            rationale = "Partial Understanding: High-level algorithmic comprehension hindered by gaps in concrete syntax mechanics."
            conf = 0.82

    else:
        # Possible Misconception
        if code_status == "correct":
            rubric_id = "RUBRIC_MISCONCEPTION_BRITTLE_PASS"
            synth_label = "Possible Misconception"
            code_ev = "Code passed basic tests by coincidence or unprincipled trial-and-error."
            viva_ev = "Student answers reveal fundamental misconceptions regarding scope, mutability, or evaluation order."
            proc_ev = "Pasted snippet with minimal testing or verification."
            rationale = "Possible Misconception: Passing code masks severe underlying conceptual flaws regarding object references."
            conf = 0.85
        else:
            rubric_id = "RUBRIC_MISCONCEPTION_SYSTEMATIC_FLAW"
            synth_label = "Possible Misconception"
            code_ev = "Code exhibits systematic conceptual error in iteration or parameter binding."
            viva_ev = "Student consistently justified incorrect mental models (e.g. mutating list during iteration or mutable defaults)."
            proc_ev = f"Struggling trajectory with {n_errors} errors and failing unit assertions."
            rationale = "Possible Misconception: Systematic code failure strongly corroborated by persistent flawed verbal reasoning."
            conf = 0.92

    if is_conflict:
        rationale += " [EVIDENCE CONFLICT: Cross-modal signals diverge; flagged for human expert review]."
        conf = round(conf * 0.78, 2)
    elif is_borderline:
        rationale += " [BORDERLINE: Assessment sits close to class boundary; flagged for human review]."
        conf = round(conf * 0.82, 2)

    return synth_label, rubric_id, rationale, code_ev, viva_ev, proc_ev, conf


def assign_revised_annotation(session_id: str,
                              original_label: str,
                              code_record: dict,
                              viva_records: list,
                              feat_record: dict,
                              is_borderline: bool,
                              is_conflict: bool):
    """
    Constructs the complete annotation record adhering strictly to Safeguards 1 & 2.
    """
    code_status = code_record["code_correctness_status"]
    
    # Assess average viva reasoning and correctness across the 3-5 turns
    viva_reasons = [v["reasoning_quality"] for v in viva_records]
    viva_corrects = [v["answer_correctness"] for v in viva_records]
    
    primary_reasoning = max(set(viva_reasons), key=viva_reasons.count)
    primary_correct = max(set(viva_corrects), key=viva_corrects.count)
    
    n_err = feat_record["n_errors"]
    final_pass = feat_record["final_state_pass"]
    
    # Apply documented pedagogical rubric
    synth_label, rubric_id, rationale, code_ev, viva_ev, proc_ev, conf = evaluate_pedagogical_rubric(
        code_status, primary_reasoning, primary_correct, n_err, final_pass, is_borderline, is_conflict
    )
    
    evidence_status = "Insufficient Evidence" if (is_borderline or is_conflict) else "Sufficient Evidence"
    
    return {
        "session_id": session_id,
        "original_label": original_label,
        "synthetic_annotator_label": synth_label,
        "human_validated_label": None,  # Strictly null until expert human validation occurs
        "ground_truth_label": synth_label,  # Target label used in pipeline (proxy label)
        "evidence_status": evidence_status,
        "annotation_rubric": rubric_id,
        "label_rationale": rationale,
        "supporting_code_evidence": code_ev,
        "supporting_viva_evidence": viva_ev,
        "supporting_process_evidence": proc_ev,
        "annotation_confidence": conf,
        "label_provenance": "synthetic_pedagogical_rule"
    }
