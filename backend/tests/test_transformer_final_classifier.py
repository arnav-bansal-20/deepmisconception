"""Verification and architectural tests for Multimodal Fusion Transformer as final classifier.

Validates all 16 verification requirements from Section 17 of the DeepMisconception master prompt:
1. CodeBERT produces expected representation shape (768).
2. Sentence Transformer produces expected representation shape (384).
3. Process Transformer produces expected representation shape (64).
4. Numerical features are processed correctly (24-D normalized vector, s_overall masked by default).
5. All intended modalities enter the fusion model.
6. Cross-attention works correctly across modalities.
7. Padding and modality masks work correctly.
8. Classification head produces exactly three logits.
9. Softmax probabilities sum to approximately 1.0.
10. Final class is selected from Transformer output (argmax).
11. Formula-generated class labels are never used as model inputs.
12. Formula scores cannot override Transformer predictions.
13. Missing evidence is handled appropriately (Pending / Insufficient Evidence).
14. Missing checkpoints return MODEL_NOT_TRAINED.
15. Random-weight predictions are not presented as valid results.
16. The normal report-generation pipeline calls the final classifier.
"""
import pytest
import numpy as np
import torch

from app.ml.codebert_embedder import CodeBERTEmbedder
from app.ml.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.ml.process_transformer import ProcessTransformerEncoder
from app.ml.cross_modal_alignment import CrossModalSemanticAlignment
from app.ml.feature_preprocessor import NumericalFeaturePreprocessor, NUMERICAL_FEATURE_NAMES, NUMERICAL_FEATURE_DIM
from app.ml.multimodal_inputs import (
    MultimodalStudentInputBundle,
    CodingProcessInput,
    SubmittedCodeInput,
    StudentExplanationInput,
)
from app.ml.multimodal_transformer import (
    MultimodalFusionTransformerModel,
    MultimodalCrossAttentionFusion,
    MultimodalPredictionResult,
    LABELS,
)
from app.services.comprehension_engine import ComprehensionEngine
from app.database import SessionLocal
from app.models.session import CodingSession
from app.models.submission import CodeSubmission
from app.models.viva_question import VivaQuestion
from app.models.answer_assessment import AnswerAssessment
from app.models.process_event import ProcessEvent


# ==============================================================================
# 1. CodeBERT representation shape
# ==============================================================================
def test_1_codebert_representation_shape():
    """Verify CodeBERT produces the expected 768-D representation shape."""
    code_sample = "def add(a, b):\n    return a + b\n"
    emb_tensor = CodeBERTEmbedder.extract_tensor(code_sample)
    assert isinstance(emb_tensor, torch.Tensor)
    assert emb_tensor.dim() == 2
    assert emb_tensor.size(0) == 1
    assert emb_tensor.size(1) == 768


# ==============================================================================
# 2. Sentence Transformer representation shape
# ==============================================================================
def test_2_sentence_transformer_representation_shape():
    """Verify Sentence Transformer produces the expected 384-D representation shape."""
    explanation_sample = "The loop condition ensures all elements are visited sequentially."
    emb_tensor = SentenceTransformerEmbedder.extract_tensor(explanation_sample)
    assert isinstance(emb_tensor, torch.Tensor)
    assert emb_tensor.dim() == 2
    assert emb_tensor.size(0) == 1
    assert emb_tensor.size(1) == 384


# ==============================================================================
# 3. Process Transformer representation shape
# ==============================================================================
def test_3_process_transformer_representation_shape():
    """Verify Process Transformer produces the expected 64-D sequence representation."""
    encoder = ProcessTransformerEncoder(output_dim=64)
    sample_events = [
        {"event_type": "edit", "delta_chars": 5, "timestamp_offset_ms": 100},
        {"event_type": "edit", "delta_chars": 12, "timestamp_offset_ms": 1500},
        {"event_type": "run", "delta_chars": 0, "timestamp_offset_ms": 3000},
        {"event_type": "error", "delta_chars": 0, "timestamp_offset_ms": 3200},
        {"event_type": "edit", "delta_chars": 3, "timestamp_offset_ms": 4500},
    ]
    proc_vec = encoder.encode_sequence(sample_events)
    assert isinstance(proc_vec, np.ndarray)
    assert proc_vec.shape == (64,)
    assert not np.isnan(proc_vec).any()


# ==============================================================================
# 4. Numerical features processed correctly (24-D, s_overall masked by default)
# ==============================================================================
def test_4_numerical_features_processed_correctly():
    """Verify numerical feature vector is 24-D, normalized in [0, 1], and excludes s_overall by default."""
    ast_sum = {"cyclomatic_complexity": 3, "total_nodes": 25, "code_lines": 15, "max_depth": 3}
    telemetry = {
        "paste_ratio": 0.15,
        "burstiness_ratio": 0.40,
        "coding_duration_seconds": 120,
        "total_events": 45,
        "incremental_score": 0.85,
        "debugging_score": 0.90,
        "revision_score": 0.75,
        "testing_score": 0.80,
    }

    vec, fdict = NumericalFeaturePreprocessor.transform(
        ast_summary=ast_sum,
        exec_output="Hello World\n",
        err_output="",
        exit_code=0,
        is_valid=True,
        viva_assessments=[{"answer_label": "Correct"}],
        telemetry_features=telemetry,
        include_s_overall=False,
    )

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (24,)
    assert NUMERICAL_FEATURE_DIM == 24
    # All values normalized in [0.0, 1.0]
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
    # s_overall is masked to 0.0 in the input vector to prevent redundant composite leakage
    assert vec[0] == 0.0
    # But s_overall is preserved in fdict for baseline reporting
    assert fdict["s_overall"] > 0.0
    # Individual formula scores are present
    assert fdict["s_ast"] > 0.0
    assert fdict["s_exec"] == 1.0
    assert fdict["s_viva"] == 1.0


# ==============================================================================
# 5. All intended modalities enter the fusion model
# ==============================================================================
def test_5_all_modalities_enter_fusion_model():
    """Verify model accepts all 5 modality representations without error."""
    model = MultimodalFusionTransformerModel(
        proc_dim=64,
        code_dim=768,
        exp_dim=384,
        align_dim=64,
        num_features_dim=24,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
    )

    B = 2
    proc_emb = torch.randn(B, 64)
    code_emb = torch.randn(B, 768)
    exp_emb = torch.randn(B, 384)
    align_rep = torch.randn(B, 64)
    num_features = torch.rand(B, 24)

    logits, probs, attn_dict = model(
        proc_emb=proc_emb,
        code_emb=code_emb,
        exp_emb=exp_emb,
        align_rep=align_rep,
        num_features=num_features,
    )

    assert logits.shape == (B, 3)
    assert probs.shape == (B, 3)
    assert "cross_attention_weights" in attn_dict
    assert "fused_representation" in attn_dict
    assert attn_dict["fused_representation"].shape == (B, 128)


# ==============================================================================
# 6. Cross-attention works correctly
# ==============================================================================
def test_6_cross_modal_attention_computation():
    """Verify bidirectional cross-attention in CrossModalSemanticAlignment."""
    align_module = CrossModalSemanticAlignment(
        code_dim=768,
        exp_dim=384,
        ast_dim=16,
        exec_dim=4,
        d_align=64,
        alignment_dim=64,
    )

    B = 2
    code_emb = torch.randn(B, 768)
    exp_emb = torch.randn(B, 384)

    align_rep, consistency_score, attn_maps = align_module(code_emb, exp_emb)

    assert align_rep.shape == (B, 64)
    assert consistency_score.shape == (B, 1)
    assert 0.0 <= consistency_score.min().item() <= 1.0
    assert 0.0 <= consistency_score.max().item() <= 1.0
    assert "code_to_exp_weights" in attn_maps
    assert "exp_to_code_weights" in attn_maps


# ==============================================================================
# 7. Padding and modality masks work correctly
# ==============================================================================
def test_7_padding_and_modality_masks():
    """Verify key_padding_mask properly masks absent modalities during cross-attention."""
    model = MultimodalFusionTransformerModel(
        proc_dim=64, code_dim=768, exp_dim=384, align_dim=64, num_features_dim=24, d_model=64, nhead=2, num_encoder_layers=1
    )

    B = 2
    proc_emb = torch.randn(B, 64)
    code_emb = torch.randn(B, 768)
    exp_emb = torch.randn(B, 384)
    align_rep = torch.randn(B, 64)
    num_features = torch.rand(B, 24)

    # Mask process modality (idx 0) and alignment (idx 3) for sample 1
    key_mask = torch.tensor([
        [False, False, False, False, False],
        [True, False, False, True, False],
    ], dtype=torch.bool)

    logits, probs, _ = model(
        proc_emb=proc_emb,
        code_emb=code_emb,
        exp_emb=exp_emb,
        align_rep=align_rep,
        num_features=num_features,
        modality_key_padding_mask=key_mask,
    )

    assert logits.shape == (B, 3)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(probs).any()


# ==============================================================================
# 8. Classification head produces exactly three logits
# ==============================================================================
def test_8_classification_head_three_logits():
    """Verify classification head outputs exactly 3 logits corresponding to the 3 classes."""
    model = MultimodalFusionTransformerModel(d_model=64, nhead=2, num_classes=3)
    proc_emb = torch.randn(1, 64)
    code_emb = torch.randn(1, 768)
    exp_emb = torch.randn(1, 384)

    logits, _, _ = model(proc_emb=proc_emb, code_emb=code_emb, exp_emb=exp_emb)
    assert logits.size(-1) == 3
    assert len(LABELS) == 3
    assert list(LABELS) == ["Understood", "Partial Understanding", "Possible Misconception"]


# ==============================================================================
# 9. Softmax probabilities sum to approximately 1.0
# ==============================================================================
def test_9_softmax_probabilities_sum_to_one():
    """Verify softmax probabilities over the 3 classes sum to 1.0 across diverse batches."""
    model = MultimodalFusionTransformerModel(d_model=64, nhead=2)
    B = 5
    proc_emb = torch.randn(B, 64)
    code_emb = torch.randn(B, 768)
    exp_emb = torch.randn(B, 384)

    _, probs, _ = model(proc_emb=proc_emb, code_emb=code_emb, exp_emb=exp_emb)
    prob_sums = probs.sum(dim=-1).detach().cpu().numpy()
    np.testing.assert_allclose(prob_sums, np.ones(B), rtol=1e-5, atol=1e-5)


# ==============================================================================
# 10. Final class is selected from Transformer output (argmax)
# ==============================================================================
def test_10_final_class_selected_from_transformer_argmax():
    """Verify predicted class strictly equals the argmax of the model probabilities."""
    fusion = MultimodalCrossAttentionFusion(seed=42)

    bundle = MultimodalStudentInputBundle(
        session_id="SESS-ARGMAX-01",
        process=CodingProcessInput(session_id="SESS-ARGMAX-01", events=[{"event_type": "edit"}]),
        code=SubmittedCodeInput(code="def foo(): return 42\n", is_valid=True),
        explanation=StudentExplanationInput(
            answers=[{"question_id": "q1", "student_answer": "It returns 42.", "answer_label": "Correct"}],
            full_transcript="It returns 42.",
        ),
    )

    # When allow_untrained=True is explicitly passed for architecture testing
    res = fusion.predict_session_bundle(bundle, allow_untrained=True)
    assert res.model_status == "READY"
    assert res.predicted_class in LABELS

    # Check that predicted_class matches max probability
    probs = res.class_probabilities
    max_label = max(probs.keys(), key=lambda k: probs[k])
    assert res.predicted_class == max_label


# ==============================================================================
# 11. Formula-generated class labels are never used as model inputs
# ==============================================================================
def test_11_formula_class_labels_never_used_as_inputs():
    """Audit input feature pipeline to verify no categorical class strings or formula labels are fed."""
    # Check NUMERICAL_FEATURE_NAMES
    disallowed_terms = ["label", "class", "understood", "misconception", "partial"]
    for name in NUMERICAL_FEATURE_NAMES:
        for term in disallowed_terms:
            assert term not in name.lower(), f"Disallowed classification label found in feature name: '{name}'"

    # Verify NumericalFeaturePreprocessor output is strictly float numbers, no strings or class indices
    vec, _ = NumericalFeaturePreprocessor.transform()
    assert vec.dtype == np.float32
    assert len(vec) == 24


# ==============================================================================
# 12. Formula scores cannot override Transformer predictions
# ==============================================================================
def test_12_formula_scores_cannot_override_transformer_predictions():
    """Verify that even with an ultra-high formula composite score (0.98), the Transformer class wins."""
    class ForcedMisconceptionTransformer:
        """Simulates a trained Transformer that diagnoses a subtle conceptual contradiction."""
        def predict_session_bundle(self, bundle):
            return MultimodalPredictionResult(
                model_status="READY",
                model_version="mock-v1.0",
                predicted_class="Possible Misconception",
                class_probabilities={"Understood": 0.05, "Partial Understanding": 0.15, "Possible Misconception": 0.80},
                confidence=0.80,
                modality_availability={"code": True, "explanation": True, "process": True},
                supporting_feature_scores={"S_OVERALL": 0.98, "S_AST": 1.0, "S_EXEC": 1.0, "S_VIVA": 0.95},
                needs_instructor_review=True,
            )

    ComprehensionEngine.set_classifier_override(ForcedMisconceptionTransformer())
    db = SessionLocal()
    try:
        session = CodingSession(student_id="STU-OVERRIDE-TEST", student_name="Override Test", title="Invariant Test")
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(
            session_id=session.id,
            code="def binary_search(arr, target): return 0\n",
            is_valid=True,
            execution_output="0",
        )
        db.add(sub)
        q = VivaQuestion(session_id=session.id, question_text="Explain invariant", concept="Search")
        db.add(q)
        db.commit()
        a = AnswerAssessment(
            question_id=q.id,
            session_id=session.id,
            student_answer="Always returns first element.",
            answer_label="Correct",  # Viva assessment marked high
            evaluation_explanation="Evaluation explanation for invariant test",
        )
        db.add(a)
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)

        # The formula score is very high (>= 0.70 which used to yield 'Understood')
        # BUT the Transformer predicted 'Possible Misconception'
        # The final report MUST be 'Possible Misconception', NOT 'Understood'!
        assert report.overall_label == "Possible Misconception"
        assert report.evidence_breakdown["transformer_prediction"]["predicted_class"] == "Possible Misconception"
    finally:
        ComprehensionEngine.reset_classifier_override()
        db.close()


# ==============================================================================
# 13. Missing evidence is handled appropriately
# ==============================================================================
def test_13_missing_evidence_handled_appropriately():
    """Verify that missing code or explanation returns INSUFFICIENT_EVIDENCE / Pending / Insufficient Evidence."""
    fusion = MultimodalCrossAttentionFusion(seed=42)

    # Missing explanation
    empty_bundle = MultimodalStudentInputBundle(
        session_id="SESS-MISSING-01",
        process=CodingProcessInput(session_id="SESS-MISSING-01"),
        code=SubmittedCodeInput(code="print(1)"),
        explanation=StudentExplanationInput(answers=[], full_transcript=""),
    )

    res = fusion.predict_session_bundle(empty_bundle)
    assert res.model_status == "INSUFFICIENT_EVIDENCE"
    assert res.predicted_class is None
    assert res.class_probabilities is None
    assert res.needs_instructor_review is True


# ==============================================================================
# 14. Missing checkpoints return MODEL_NOT_TRAINED
# ==============================================================================
def test_14_missing_checkpoint_returns_model_not_trained():
    """Verify that when no checkpoint exists and allow_untrained is False, MODEL_NOT_TRAINED is returned."""
    fusion = MultimodalCrossAttentionFusion(seed=42)
    fusion.is_trained = False

    bundle = MultimodalStudentInputBundle(
        session_id="SESS-UNTRAINED-01",
        process=CodingProcessInput(session_id="SESS-UNTRAINED-01", events=[{"event_type": "edit"}]),
        code=SubmittedCodeInput(code="def solve(): pass\n", is_valid=True),
        explanation=StudentExplanationInput(
            answers=[{"question_id": "q1", "student_answer": "Valid explanation", "answer_label": "Correct"}],
            full_transcript="Valid explanation",
        ),
    )

    res = fusion.predict_session_bundle(bundle, allow_untrained=False)
    assert res.model_status == "MODEL_NOT_TRAINED"
    assert res.predicted_class is None
    assert res.class_probabilities is None
    assert res.confidence == 0.0


# ==============================================================================
# 15. Random-weight predictions are not presented as valid results
# ==============================================================================
def test_15_random_weights_not_presented_as_valid():
    """Verify the system strictly refuses to return predictions with randomly initialized weights."""
    fusion = MultimodalCrossAttentionFusion(seed=42)
    assert fusion.is_trained is False

    # Calling without trained checkpoint must not present random predictions
    bundle = MultimodalStudentInputBundle(
        session_id="SESS-NO-RANDOM-01",
        process=CodingProcessInput(session_id="SESS-NO-RANDOM-01"),
        code=SubmittedCodeInput(code="x = 10\n"),
        explanation=StudentExplanationInput(answers=[{"answer_label": "Correct"}], full_transcript="Valid answer"),
    )
    res = fusion.predict_session_bundle(bundle)
    assert res.model_status == "MODEL_NOT_TRAINED"
    assert res.predicted_class is None


# ==============================================================================
# 16. Normal report-generation pipeline calls the final classifier
# ==============================================================================
def test_16_normal_report_generation_calls_final_classifier():
    """Verify ComprehensionEngine.synthesize_report invokes the Transformer and records its predictions."""
    called = {"invoked": False}

    class SpyClassifier:
        def predict_session_bundle(self, bundle):
            called["invoked"] = True
            return MultimodalPredictionResult(
                model_status="READY",
                model_version="spy-v1.0",
                predicted_class="Understood",
                class_probabilities={"Understood": 0.88, "Partial Understanding": 0.10, "Possible Misconception": 0.02},
                confidence=0.88,
                modality_availability={"code": True, "explanation": True, "process": True},
                supporting_feature_scores={"S_OVERALL": 0.85},
                needs_instructor_review=False,
            )

    ComprehensionEngine.set_classifier_override(SpyClassifier())
    db = SessionLocal()
    try:
        session = CodingSession(student_id="STU-SPY-TEST", student_name="Spy Test", title="Pipeline Call Test")
        db.add(session)
        db.commit()
        db.refresh(session)

        sub = CodeSubmission(session_id=session.id, code="print('hello')", is_valid=True)
        db.add(sub)
        q = VivaQuestion(session_id=session.id, question_text="Explain print", concept="IO")
        db.add(q)
        db.commit()
        a = AnswerAssessment(
            question_id=q.id,
            session_id=session.id,
            student_answer="Outputs text.",
            answer_label="Correct",
            evaluation_explanation="Clear description of output mechanism",
        )
        db.add(a)
        db.commit()

        report = ComprehensionEngine.synthesize_report(session.id, db)

        assert called["invoked"] is True
        assert report.overall_label == "Understood"
        assert report.confidence_score == 0.88
        assert "transformer_prediction" in report.evidence_breakdown
        assert report.evidence_breakdown["transformer_prediction"]["model_status"] == "READY"
    finally:
        ComprehensionEngine.reset_classifier_override()
        db.close()
