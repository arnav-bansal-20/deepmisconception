"""Unit tests verifying the Multimodal Transformer Methodology (Phases 1-5).

Verifies:
1. Code embedding input is accepted (CodeBERT 768-D).
2. Explanation embedding input is accepted (Sentence-Transformer 384-D).
3. Process embedding input is accepted (Process Transformer 64-D).
4. Alignment representation is accepted (Cross-Modal Alignment 64-D).
5. Numerical features are accepted (AST + Execution + Telemetry 24-D).
6. Output tensor has the expected dimensions (B, 3).
7. Softmax probabilities sum to approximately 1.0.
8. Missing modality masks are handled correctly.
9. Batch processing works for single (B=1) and batch (B=4) inputs.
10. Gradients flow through all trainable fusion and alignment layers.
"""
import pytest
import numpy as np
import torch

from app.ml.process_transformer import PyTorchProcessTransformer, ProcessTransformerEncoder
from app.ml.codebert_embedder import CodeBERTEmbedder
from app.ml.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.ml.cross_modal_alignment import CrossModalSemanticAlignment
from app.ml.multimodal_transformer import (
    MultimodalFusionTransformerModel,
    MultimodalCrossAttentionFusion,
    LABELS,
)
from app.ml.multimodal_inputs import (
    CodingProcessInput,
    SubmittedCodeInput,
    StudentExplanationInput,
    MultimodalStudentInputBundle,
)
from app.ml.feature_extractor import MultimodalFeatureExtractor


# =========================================================================
# Phase 2 & 3: Input Bundle and Process Transformer Tests
# =========================================================================

def test_multimodal_student_input_bundle():
    """Verify Phase 2 multimodal input bundling across the 3 modalities."""
    events = [
        {"event_type": "edit", "delta_chars": 5, "timestamp_s": 0.1},
        {"event_type": "run", "delta_chars": 0, "timestamp_s": 1.2},
    ]
    bundle = MultimodalStudentInputBundle.from_session_data(
        session_id="SESS-TEST-001",
        events=events,
        telemetry_features={"paste_ratio": 0.25, "edit_count": 10, "run_count": 2},
        code="def solve(x):\n    return x * 2\n",
        ast_summary={"loops": 0, "functions": 1},
        exec_output="4",
        viva_assessments=[{"question_id": "q1", "student_answer": "It doubles the input."}],
        viva_questions=[{"id": "q1", "question_text": "What does solve do?", "concept": "Multiplication"}],
    )
    assert bundle.process.get_event_count() == 2
    assert bundle.process.numerical_features["paste_ratio"] == 0.25
    assert "def solve" in bundle.code.code
    assert len(bundle.explanation.question_answer_pairs) == 1
    assert bundle.explanation.question_answer_pairs[0][2] == "Multiplication"


def test_process_transformer_architecture_and_forward():
    """Verify Phase 3.A Process Transformer forward pass, padding mask, and output shape."""
    model = PyTorchProcessTransformer(
        event_emb_dim=16,
        num_event_features=7,
        global_feature_dim=8,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        output_dim=64,
    )
    B, T = 3, 10
    event_types = torch.randint(1, 8, (B, T))
    event_nums = torch.randn(B, T, 7)
    global_feats = torch.randn(B, 8)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    pad_mask[0, 5:] = True  # Padded positions for sequence 0

    out = model(event_types, event_nums, padding_mask=pad_mask, global_features=global_feats)
    assert out.shape == (B, 64)
    assert not torch.isnan(out).any()

    # Test empty sequence fallback handling
    empty_mask = torch.ones(1, 5, dtype=torch.bool)  # Entirely padded
    empty_types = torch.zeros(1, 5, dtype=torch.long)
    empty_nums = torch.zeros(1, 5, 7)
    out_empty = model(empty_types, empty_nums, padding_mask=empty_mask, global_features=torch.randn(1, 8))
    assert out_empty.shape == (1, 64)
    assert not torch.isnan(out_empty).any()


def test_process_transformer_encoder_wrapper():
    """Verify high-level ProcessTransformerEncoder wrapper."""
    encoder = ProcessTransformerEncoder(output_dim=64)
    events = [
        {"event_type": "edit", "delta_chars": 10, "paste_length": 0, "cursor_line": 1, "timestamp_s": 0.5},
        {"event_type": "paste", "delta_chars": 50, "paste_length": 50, "cursor_line": 2, "timestamp_s": 2.0},
        {"event_type": "run", "delta_chars": 0, "paste_length": 0, "cursor_line": 2, "timestamp_s": 3.0},
    ]
    rep = encoder.encode_sequence(events)
    assert rep.shape == (64,)
    assert not np.isnan(rep).any()


# =========================================================================
# Phase 4: Cross-Modal Semantic Alignment Module Tests
# =========================================================================

def test_cross_modal_semantic_alignment_forward_and_shapes():
    """Verify Phase 4 Cross-Modal Semantic Alignment forward pass and tensor shapes."""
    alignment = CrossModalSemanticAlignment(
        code_dim=768,
        exp_dim=384,
        ast_dim=16,
        exec_dim=4,
        d_align=128,
        nhead=4,
        alignment_dim=64,
    )
    B = 2
    code_emb = torch.randn(B, 768)
    exp_emb = torch.randn(B, 384)
    ast_feat = torch.randn(B, 16)
    exec_feat = torch.randn(B, 4)

    align_rep, score, attn_maps = alignment(code_emb, exp_emb, ast_feat, exec_feat)

    # 1. Output representation dimensions
    assert align_rep.shape == (B, 64)
    assert not torch.isnan(align_rep).any()

    # 2. Consistency score bounded in [0, 1]
    assert score.shape == (B, 1)
    assert (score >= 0.0).all() and (score <= 1.0).all()

    # 3. Attention maps present
    assert "code_to_exp_weights" in attn_maps
    assert "exp_to_code_weights" in attn_maps


def test_cross_modal_alignment_gradient_flow():
    """Verify gradients propagate through trainable parameters of alignment module."""
    alignment = CrossModalSemanticAlignment(
        code_dim=768,
        exp_dim=384,
        d_align=64,
        alignment_dim=32,
    )
    code_emb = torch.randn(2, 768, requires_grad=True)
    exp_emb = torch.randn(2, 384, requires_grad=True)

    align_rep, score, _ = alignment(code_emb, exp_emb)
    loss = align_rep.sum() + score.sum()
    loss.backward()

    # Check parameter gradients
    assert alignment.proj_code[0].weight.grad is not None
    assert alignment.proj_exp[0].weight.grad is not None
    assert alignment.bilinear_matrix.grad is not None
    assert alignment.alignment_mlp[0].weight.grad is not None


# =========================================================================
# Phase 5: Multimodal Fusion Transformer Tests (10 Verification Criteria)
# =========================================================================

def test_multimodal_fusion_transformer_ten_criteria():
    """Complete 10-point verification of the Multimodal Fusion Transformer."""
    fusion_model = MultimodalFusionTransformerModel(
        proc_dim=64,
        code_dim=768,
        exp_dim=384,
        align_dim=64,
        num_features_dim=24,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_classes=3,
    )

    B = 4
    # Criterion 1: Code embedding input is accepted (768-D)
    code_emb = torch.randn(B, 768)
    # Criterion 2: Explanation embedding input is accepted (384-D)
    exp_emb = torch.randn(B, 384)
    # Criterion 3: Process embedding input is accepted (64-D)
    proc_emb = torch.randn(B, 64)
    # Criterion 4: Alignment representation is accepted (64-D)
    align_rep = torch.randn(B, 64)
    # Criterion 5: Numerical features are accepted (24-D)
    num_features = torch.randn(B, 24)

    # Criterion 9: Batch processing works (B=4)
    logits, probs, attn_dict = fusion_model(
        proc_emb=proc_emb,
        code_emb=code_emb,
        exp_emb=exp_emb,
        align_rep=align_rep,
        num_features=num_features,
    )

    # Criterion 6: Output tensor has expected dimensions (B, 3)
    assert logits.shape == (B, 3)
    assert probs.shape == (B, 3)

    # Criterion 7: Softmax probabilities sum to approximately 1.0
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Criterion 8: Missing modality masks are handled correctly
    # Mask out Explanation (index 2) and Alignment (index 3) for sample 0
    modality_mask = torch.zeros(B, 5, dtype=torch.bool)
    modality_mask[0, 2] = True
    modality_mask[0, 3] = True

    masked_logits, masked_probs, _ = fusion_model(
        proc_emb=proc_emb,
        code_emb=code_emb,
        exp_emb=exp_emb,
        align_rep=align_rep,
        num_features=num_features,
        modality_key_padding_mask=modality_mask,
    )
    assert masked_logits.shape == (B, 3)
    assert not torch.isnan(masked_logits).any()
    assert torch.allclose(masked_probs.sum(dim=-1), torch.ones(B), atol=1e-5)

    # Criterion 10: Gradients flow through all trainable fusion and alignment layers
    fusion_model.train()
    target = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    loss.backward()

    assert fusion_model.proj_proc[0].weight.grad is not None
    assert fusion_model.proj_code[0].weight.grad is not None
    assert fusion_model.proj_exp[0].weight.grad is not None
    assert fusion_model.proj_align[0].weight.grad is not None
    assert fusion_model.proj_num[0].weight.grad is not None
    assert fusion_model.modality_type_embeddings.weight.grad is not None
    assert fusion_model.classifier[1].weight.grad is not None


def test_single_sample_batch_and_numpy_integration():
    """Verify single-sample inference (B=1) and end-to-end numpy bridge in MultimodalCrossAttentionFusion."""
    fusion_api = MultimodalCrossAttentionFusion(
        code_dim=768,
        viva_dim=384,
        proc_dim=64,
        align_dim=64,
        num_features_dim=24,
        hidden_dim=128,
        num_classes=3,
    )

    code_vec = np.random.randn(768).astype(np.float32)
    exp_vec = np.random.randn(384).astype(np.float32)
    proc_vec = np.random.randn(64).astype(np.float32)

    probs, fused = fusion_api.forward(code_vec, exp_vec, proc_vec)

    assert probs.shape == (1, 3)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
    assert fused.shape == (1, 128)
