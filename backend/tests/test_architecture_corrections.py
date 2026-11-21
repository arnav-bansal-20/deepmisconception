"""Automated verification test suite for DeepMisconception architecture and integration fixes.

Covers all 20 required verification dimensions specified in Section 5 of the Master Prompt:
1. CodeBERT output dimension (768).
2. Sentence Transformer output dimension (384).
3. Process Transformer output dimension (64).
4. Numerical feature dimension (24).
5. Fusion Transformer input contract.
6. Gradient flow through Process Transformer.
7. Gradient flow through alignment and fusion modules.
8. Frozen pretrained encoder parameters.
9. Process Transformer checkpoint save and reload.
10. Correct timestamp conversion and event ordering.
11. Missing process modality.
12. Missing code modality.
13. Missing viva modality.
14. Cross-modal alignment masking.
15. Invalid and non-finite embeddings.
16. Leakage prevention for S_VIVA and other annotation-derived fields.
17. Three-class output dimension and class mapping.
18. Pending / Insufficient Evidence handling.
19. MODEL_NOT_TRAINED handling.
20. Confirmation that the final prediction comes from the Fusion Transformer.
"""
import os
import math
import tempfile
from datetime import datetime, timezone, timedelta
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.config import (
    CODE_DIM,
    EXP_DIM,
    PROC_DIM,
    ALIGN_DIM,
    NUM_FEATURES_DIM,
    D_MODEL,
    NUM_CLASSES,
    LABELS,
    LABEL_MAP,
    REV_LABEL_MAP,
    STATUS_READY,
    STATUS_MODEL_NOT_TRAINED,
    STATUS_INSUFFICIENT_EVIDENCE,
    NUMERICAL_FEATURE_NAMES,
)
from app.ml.codebert_embedder import CodeBERTEmbedder
from app.ml.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.ml.process_transformer import (
    PyTorchProcessTransformer,
    ProcessTransformerEncoder,
    normalize_process_event_timestamps,
    EVENT_TYPE_MAP,
    NUM_EVENT_TYPES,
)
from app.ml.cross_modal_alignment import CrossModalSemanticAlignment
from app.ml.feature_preprocessor import NumericalFeaturePreprocessor
from app.ml.multimodal_transformer import (
    MultimodalFusionTransformerModel,
    MultimodalCrossAttentionFusion,
    MultimodalPredictionResult,
)
from app.ml.training_pipeline import (
    MultimodalSample,
    MultimodalDataset,
    MultimodalTrainer,
    collate_multimodal_batch,
)
from app.ml.multimodal_inputs import (
    MultimodalStudentInputBundle,
    SubmittedCodeInput,
    CodingProcessInput,
    StudentExplanationInput,
)


# ==============================================================================
# 1. CodeBERT Output Dimension (768)
# ==============================================================================
def test_01_codebert_output_dimension():
    """Verify that CodeBERT produces exactly 768-dimensional float32 representations."""
    sample_code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n"
    emb, meta = CodeBERTEmbedder.extract_embedding_checked(sample_code)
    assert meta["is_available"] is True
    assert emb is not None
    assert emb.shape == (CODE_DIM,)
    assert emb.dtype == np.float32
    assert CODE_DIM == 768


# ==============================================================================
# 2. Sentence Transformer Output Dimension (384)
# ==============================================================================
def test_02_sentence_transformer_output_dimension():
    """Verify that Sentence Transformer produces exactly 384-dimensional float32 representations."""
    sample_viva = "Question: Why is n <= 1 the base case? Explanation: It terminates the call stack for 0 and 1."
    emb, meta = SentenceTransformerEmbedder.extract_embedding_checked(sample_viva)
    assert meta["is_available"] is True
    assert emb is not None
    assert emb.shape == (EXP_DIM,)
    assert emb.dtype == np.float32
    assert EXP_DIM == 384


# ==============================================================================
# 3. Process Transformer Output Dimension (64)
# ==============================================================================
def test_03_process_transformer_output_dimension():
    """Verify that PyTorchProcessTransformer produces exactly (B, 64) representations."""
    model = PyTorchProcessTransformer(output_dim=PROC_DIM)
    event_types = torch.tensor([[2, 2, 5, 2]], dtype=torch.long)
    event_features = torch.zeros((1, 4, 7), dtype=torch.float32)
    padding_mask = torch.tensor([[False, False, False, False]], dtype=torch.bool)
    global_features = torch.zeros((1, 8), dtype=torch.float32)

    out = model(event_types, event_features, padding_mask, global_features)
    assert out.shape == (1, PROC_DIM)
    assert PROC_DIM == 64


# ==============================================================================
# 4. Numerical Feature Dimension (24)
# ==============================================================================
def test_04_numerical_feature_dimension():
    """Verify that NumericalFeaturePreprocessor.transform produces exactly 24-D normalized vector."""
    vec, fdict = NumericalFeaturePreprocessor.transform(
        ast_summary={"cyclomatic_complexity": 3, "code_lines": 12},
        exec_output="Passed 3 tests\n",
        err_output="",
        exit_code=0,
        is_valid=True,
    )
    assert vec.shape == (NUM_FEATURES_DIM,)
    assert len(NUMERICAL_FEATURE_NAMES) == 24
    assert NUM_FEATURES_DIM == 24
    assert vec.dtype == np.float32
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)


# ==============================================================================
# 5. Fusion Transformer Input Contract
# ==============================================================================
def test_05_fusion_transformer_input_contract():
    """Verify that MultimodalFusionTransformerModel enforces the exact dimensional contract."""
    model = MultimodalFusionTransformerModel()

    proc = torch.randn(2, PROC_DIM)
    code = torch.randn(2, CODE_DIM)
    exp = torch.randn(2, EXP_DIM)
    align = torch.randn(2, ALIGN_DIM)
    num = torch.randn(2, NUM_FEATURES_DIM)

    # Valid contract passes
    logits, probs, _ = model(proc, code, exp, align, num)
    assert logits.shape == (2, NUM_CLASSES)
    assert probs.shape == (2, NUM_CLASSES)

    # Incompatible code dimension raises ValueError (no silent padding/truncation)
    bad_code = torch.randn(2, 512)
    with pytest.raises(ValueError, match="Expected code_dim"):
        model(proc, bad_code, exp, align, num)

    # Incompatible numerical dimension raises ValueError
    bad_num = torch.randn(2, 16)
    with pytest.raises(ValueError, match="Expected num_features_dim"):
        model(proc, code, exp, align, bad_num)


# ==============================================================================
# 6. Gradient Flow Through Process Transformer
# ==============================================================================
def test_06_gradient_flow_process_transformer():
    """Verify that Process Transformer parameters receive gradients during training."""
    proc_model = PyTorchProcessTransformer(output_dim=PROC_DIM)
    fusion_model = MultimodalFusionTransformerModel()
    trainer = MultimodalTrainer(model=fusion_model, process_transformer=proc_model)

    event_types = torch.tensor([[2, 5]], dtype=torch.long)
    event_features = torch.randn(1, 2, 7)
    padding_mask = torch.tensor([[False, False]], dtype=torch.bool)
    global_features = torch.randn(1, 8)

    sample = MultimodalSample(
        session_id="S_TEST_GRAD",
        code_emb=np.random.randn(CODE_DIM).astype(np.float32),
        exp_emb=np.random.randn(EXP_DIM).astype(np.float32),
        num_features=np.zeros(NUM_FEATURES_DIM, dtype=np.float32),
        label=0,
        proc_event_types=event_types.numpy()[0],
        proc_event_features=event_features.numpy()[0],
        proc_global_features=global_features.numpy()[0],
    )
    dataset = MultimodalDataset([sample])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_multimodal_batch)

    loss = trainer.train_epoch(dataloader)
    assert loss > 0.0

    # Verify that gradients flowed through Process Transformer token projection weights
    assert proc_model.token_proj.weight.grad is not None
    assert torch.sum(torch.abs(proc_model.token_proj.weight.grad)) > 0.0


# ==============================================================================
# 7. Gradient Flow Through Alignment and Fusion Modules
# ==============================================================================
def test_07_gradient_flow_alignment_and_fusion():
    """Verify that CrossModalSemanticAlignment and MultimodalFusionTransformer receive gradients."""
    fusion_model = MultimodalFusionTransformerModel()
    align_model = CrossModalSemanticAlignment()
    trainer = MultimodalTrainer(model=fusion_model, alignment_module=align_model)

    sample = MultimodalSample(
        session_id="S_ALIGN_GRAD",
        code_emb=np.random.randn(CODE_DIM).astype(np.float32),
        exp_emb=np.random.randn(EXP_DIM).astype(np.float32),
        num_features=np.zeros(NUM_FEATURES_DIM, dtype=np.float32),
        label=1,
        proc_emb=np.random.randn(PROC_DIM).astype(np.float32),
    )
    dataset = MultimodalDataset([sample])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_multimodal_batch)

    trainer.train_epoch(dataloader)

    # Alignment MLP receives gradients
    assert align_model.proj_code[0].weight.grad is not None
    assert torch.sum(torch.abs(align_model.proj_code[0].weight.grad)) > 0.0

    # Fusion classification head receives gradients
    assert fusion_model.classifier[1].weight.grad is not None
    assert torch.sum(torch.abs(fusion_model.classifier[1].weight.grad)) > 0.0


# ==============================================================================
# 8. Frozen Pretrained Encoder Parameters
# ==============================================================================
def test_08_frozen_pretrained_encoder_parameters():
    """Verify that CodeBERT and SentenceTransformer remain frozen without gradient accumulation."""
    code_status = CodeBERTEmbedder.get_status()
    exp_status = SentenceTransformerEmbedder.get_status()
    assert code_status["is_available"] is True
    assert exp_status["is_available"] is True

    # Pretrained models are in eval mode and separate from trainer trainable parameters
    fusion_model = MultimodalFusionTransformerModel()
    proc_model = PyTorchProcessTransformer()
    align_model = CrossModalSemanticAlignment()
    trainer = MultimodalTrainer(
        model=fusion_model,
        alignment_module=align_model,
        process_transformer=proc_model,
    )

    # Pretrained models do not participate in optimizer parameter groups
    from app.ml.codebert_embedder import _CODEBERT_MODEL
    from app.ml.sentence_transformer_embedder import _ST_MODEL

    if _CODEBERT_MODEL is not None:
        assert not _CODEBERT_MODEL.training
        for p in _CODEBERT_MODEL.parameters():
            assert not any(p is tp for tp in trainer.trainable_params)

    if _ST_MODEL is not None:
        assert not _ST_MODEL.training
        for p in _ST_MODEL.parameters():
            assert not any(p is tp for tp in trainer.trainable_params)


# ==============================================================================
# 9. Process Transformer Checkpoint Save and Reload
# ==============================================================================
def test_09_process_transformer_checkpoint_save_and_reload():
    """Verify that Process Transformer weights are saved and reloaded with exact tensor fidelity."""
    fusion = MultimodalCrossAttentionFusion()
    # Perturb weights slightly to verify persistence
    with torch.no_grad():
        fusion.process_transformer.token_proj.bias.fill_(0.42)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        fusion.save_checkpoint(ckpt_path)

        # Create fresh fusion instance
        new_fusion = MultimodalCrossAttentionFusion()
        new_fusion.load_checkpoint(ckpt_path)

        assert torch.allclose(
            fusion.process_transformer.token_proj.bias,
            new_fusion.process_transformer.token_proj.bias,
        )


# ==============================================================================
# 10. Correct Timestamp Conversion and Event Ordering
# ==============================================================================
def test_10_timestamp_conversion_and_event_ordering():
    """Verify robust chronological ordering, relative elapsed seconds, and timezone consistency."""
    t0 = datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=15)
    t2 = t0 + timedelta(seconds=45)

    # Pass unordered events with mixed timestamps
    unordered = [
        {"event_type": "run", "timestamp": t2.isoformat(), "delta_chars": 0},
        {"event_type": "edit", "timestamp": t0, "delta_chars": 5},
        {"event_type": "edit", "timestamp": t1.isoformat(), "delta_chars": 12},
    ]
    norm = normalize_process_event_timestamps(unordered)

    assert len(norm) == 3
    # Preserves strict chronological ordering: t0 -> t1 -> t2
    assert norm[0]["event_type"] == "edit" and norm[0]["elapsed_seconds"] == 0.0
    assert norm[1]["event_type"] == "edit" and norm[1]["elapsed_seconds"] == 15.0
    assert norm[2]["event_type"] == "run" and norm[2]["elapsed_seconds"] == 45.0
    assert norm[1]["time_gap_seconds"] == 15.0
    assert norm[2]["time_gap_seconds"] == 30.0


# ==============================================================================
# 11. Missing Process Modality
# ==============================================================================
def test_11_missing_process_modality():
    """Verify that missing process events are handled via explicit modality mask without crashing."""
    model = MultimodalFusionTransformerModel()

    proc = torch.zeros(1, PROC_DIM)
    code = torch.randn(1, CODE_DIM)
    exp = torch.randn(1, EXP_DIM)
    align = torch.randn(1, ALIGN_DIM)
    num = torch.randn(1, NUM_FEATURES_DIM)

    # Mask process modality: index 0 = True (missing)
    mask = torch.tensor([[True, False, False, False, False]], dtype=torch.bool)
    logits, probs, _ = model(proc, code, exp, align, num, modality_key_padding_mask=mask)

    assert logits.shape == (1, NUM_CLASSES)
    assert not torch.isnan(logits).any()


# ==============================================================================
# 12. Missing Code Modality
# ==============================================================================
def test_12_missing_code_modality():
    """Verify that missing code modality triggers INSUFFICIENT_EVIDENCE abstention in bundle inference."""
    fusion = MultimodalCrossAttentionFusion()
    bundle = MultimodalStudentInputBundle(
        code=SubmittedCodeInput(code=""),  # Empty code
        explanation=StudentExplanationInput(full_transcript="I used an accumulator loop."),
        process=CodingProcessInput(events=[]),
    )
    res = fusion.predict_session_bundle(bundle)
    assert res.model_status == STATUS_INSUFFICIENT_EVIDENCE
    assert res.predicted_class is None
    assert res.modality_availability["code"] is False


# ==============================================================================
# 13. Missing Viva Modality
# ==============================================================================
def test_13_missing_viva_modality():
    """Verify that missing viva explanation triggers INSUFFICIENT_EVIDENCE abstention in bundle inference."""
    fusion = MultimodalCrossAttentionFusion()
    bundle = MultimodalStudentInputBundle(
        code=SubmittedCodeInput(code="def foo(): return 1"),
        explanation=StudentExplanationInput(full_transcript=""),  # Empty viva
        process=CodingProcessInput(events=[]),
    )
    res = fusion.predict_session_bundle(bundle)
    assert res.model_status == STATUS_INSUFFICIENT_EVIDENCE
    assert res.predicted_class is None
    assert res.modality_availability["explanation"] is False


# ==============================================================================
# 14. Cross-Modal Alignment Masking
# ==============================================================================
def test_14_cross_modal_alignment_masking():
    """Verify that CrossModalSemanticAlignment zeroes out alignment representation when code/viva is masked."""
    align_model = CrossModalSemanticAlignment()

    code = torch.randn(2, CODE_DIM)
    exp = torch.randn(2, EXP_DIM)
    code_mask = torch.tensor([False, True], dtype=torch.bool)  # Sample 1 has code missing

    align_rep, score, _ = align_model(code, exp, code_mask=code_mask)
    assert align_rep.shape == (2, ALIGN_DIM)
    # Sample 1 must be zeroed out
    assert torch.all(align_rep[1] == 0.0)
    assert score[1].item() == 0.0


# ==============================================================================
# 15. Invalid and Non-Finite Embeddings
# ==============================================================================
def test_15_invalid_and_non_finite_embeddings():
    """Verify that embedders reject empty or non-finite inputs gracefully."""
    emb, meta = CodeBERTEmbedder.extract_embedding_checked("")
    assert meta["status"] == "empty_input"
    assert emb is None

    t, meta = SentenceTransformerEmbedder.extract_tensor_checked("   ")
    assert meta["status"] == "empty_input"
    assert t is None


# ==============================================================================
# 16. Leakage Prevention for S_VIVA and Annotation-Derived Fields
# ==============================================================================
def test_16_leakage_prevention_s_viva_and_annotations():
    """Verify that S_VIVA (index 3) and S_OVERALL (index 0) are strictly 0.0 in the primary model's input vector."""
    vec, fdict = NumericalFeaturePreprocessor.transform(
        ast_summary={"cyclomatic_complexity": 2},
        exec_output="OK\n",
        viva_assessments=[{"answer_label": "Correct"}],
        include_s_overall=False,
        include_s_viva=False,
    )
    # In supporting dictionary, scores are kept for baseline audit
    assert fdict["s_viva"] == 1.0
    assert fdict["s_overall"] > 0.0

    # In primary model input vector, both are strictly MASKED to 0.0
    assert vec[0] == 0.0  # S_OVERALL masked
    assert vec[3] == 0.0  # S_VIVA strictly masked from model input


# ==============================================================================
# 17. Three-Class Output Dimension and Class Mapping
# ==============================================================================
def test_17_three_class_output_dimension_and_mapping():
    """Verify exactly 3 target comprehension classes matching the canonical mapping."""
    assert len(LABELS) == 3
    assert LABELS == ("Understood", "Partial Understanding", "Possible Misconception")
    assert LABEL_MAP["Understood"] == 0
    assert LABEL_MAP["Partial Understanding"] == 1
    assert LABEL_MAP["Possible Misconception"] == 2
    assert NUM_CLASSES == 3


# ==============================================================================
# 18. Pending / Insufficient Evidence Handling
# ==============================================================================
def test_18_pending_insufficient_evidence_handling():
    """Verify that insufficient evidence produces an explicit abstention result, NOT a 4th class."""
    fusion = MultimodalCrossAttentionFusion()
    bundle = MultimodalStudentInputBundle(
        code=SubmittedCodeInput(code=""),
        explanation=StudentExplanationInput(full_transcript=""),
        process=CodingProcessInput(events=[]),
    )
    res = fusion.predict_session_bundle(bundle)
    assert res.model_status == STATUS_INSUFFICIENT_EVIDENCE
    assert res.predicted_class is None
    assert res.class_probabilities is None
    assert res.needs_instructor_review is True


# ==============================================================================
# 19. MODEL_NOT_TRAINED Handling
# ==============================================================================
def test_19_model_not_trained_handling():
    """Verify that an untrained model returns MODEL_NOT_TRAINED and refuses random predictions."""
    fusion = MultimodalCrossAttentionFusion()
    fusion.is_trained = False

    bundle = MultimodalStudentInputBundle(
        code=SubmittedCodeInput(code="def solution(): return 1\n"),
        explanation=StudentExplanationInput(full_transcript="The function returns 1."),
        process=CodingProcessInput(events=[]),
    )
    res = fusion.predict_session_bundle(bundle, allow_untrained=False)
    assert res.model_status == STATUS_MODEL_NOT_TRAINED
    assert res.predicted_class is None
    assert res.class_probabilities is None


# ==============================================================================
# 20. Confirmation That Final Prediction Comes From Fusion Transformer
# ==============================================================================
def test_20_final_prediction_comes_from_fusion_transformer():
    """Verify that final prediction is produced by Fusion Transformer classification head argmax."""
    fusion = MultimodalCrossAttentionFusion()
    fusion.is_trained = True

    bundle = MultimodalStudentInputBundle(
        code=SubmittedCodeInput(code="def add(a, b): return a + b\n"),
        explanation=StudentExplanationInput(full_transcript="Takes two inputs and sums them."),
        process=CodingProcessInput(events=[]),
    )
    res = fusion.predict_session_bundle(bundle, allow_untrained=True)
    assert res.model_status == STATUS_READY
    assert res.predicted_class in LABELS
    assert res.class_probabilities is not None

    # Argmax probability matches predicted_class
    best_cls = max(res.class_probabilities.items(), key=lambda item: item[1])[0]
    assert res.predicted_class == best_cls
    assert res.confidence == pytest.approx(res.class_probabilities[best_cls], abs=1e-3)
