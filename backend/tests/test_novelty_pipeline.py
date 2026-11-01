"""Unit and integration tests for the DeepMisconception Novelty Schematic Architecture.

Validates:
1. Coding-Process Data -> Process Transformer Encoder -> Process Representation
2. Source Code -> AST + CodeBERT -> Code Representation
3. Student Explanation -> Sentence-Transformer -> Explanation Representation
4. Multimodal Fusion (Proposed Model) -> Cross-Attention Encoder -> 3-Class Comprehension Output
"""
import pytest
import numpy as np
from app.ml.codebert_embedder import CodeBERTEmbedder
from app.ml.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.ml.process_transformer import ProcessTransformerEncoder
from app.ml.fusion_model import MultimodalTransformerFusionModel, LABELS


def test_code_feature_extractor_ast_and_codebert():
    """Verify Code Feature Extractor extracts CodeBERT and AST representations."""
    code = (
        "def find_duplicates(arr):\n"
        "    seen = set()\n"
        "    for x in arr:\n"
        "        if x in seen:\n"
        "            return True\n"
        "        seen.add(x)\n"
        "    return False\n"
    )
    # 768-D dense CodeBERT representation
    emb_768 = CodeBERTEmbedder.extract_embedding(code)
    assert emb_768.shape == (768,)
    assert not np.isnan(emb_768).any()

    # 16-D projected code representation
    emb_16 = CodeBERTEmbedder.extract_projected(code, target_dim=16)
    assert emb_16.shape == (16,)
    assert not np.isnan(emb_16).any()


def test_explanation_feature_extractor_sentence_transformer():
    """Verify Explanation Feature Extractor extracts Sentence-Transformer embeddings."""
    explanation = (
        "I used a set because checking whether an element exists is fast on average, "
        "allowing O(1) membership lookups to detect duplicates in linear O(N) time."
    )
    # 384-D Sentence-Transformer representation
    emb_384 = SentenceTransformerEmbedder.extract_embedding(explanation)
    assert emb_384.shape == (384,)
    assert not np.isnan(emb_384).any()

    # 16-D projected explanation representation
    emb_16 = SentenceTransformerEmbedder.extract_projected(explanation, target_dim=16)
    assert emb_16.shape == (16,)
    assert not np.isnan(emb_16).any()


def test_process_feature_extractor_transformer_encoder():
    """Verify Process Feature Extractor encodes temporal telemetry sequences."""
    events = [
        {"event_type": "edit", "delta_chars": 4, "paste_length": 0, "cursor_line": 1, "timestamp_s": 0.5},
        {"event_type": "edit", "delta_chars": 12, "paste_length": 0, "cursor_line": 2, "timestamp_s": 1.2},
        {"event_type": "paste", "delta_chars": 60, "paste_length": 60, "cursor_line": 3, "timestamp_s": 4.5},
        {"event_type": "run", "delta_chars": 0, "paste_length": 0, "cursor_line": 3, "timestamp_s": 6.0},
        {"event_type": "runtime_error", "delta_chars": 0, "paste_length": 0, "cursor_line": 3, "timestamp_s": 6.2},
        {"event_type": "edit", "delta_chars": -2, "paste_length": 0, "cursor_line": 3, "timestamp_s": 8.0},
    ]
    encoder = ProcessTransformerEncoder(seed=42)
    proc_rep = encoder.encode_sequence(events)
    assert proc_rep.shape == (16,)
    assert not np.isnan(proc_rep).any()


def test_transformer_fusion_model_cross_attention_and_classification():
    """Verify Proposed Multimodal Cross-Attention Fusion Model produces 3-class comprehension assessment."""
    model = MultimodalTransformerFusionModel(
        d_code=16,
        d_exp=16,
        d_proc=16,
        d_k=16,
        hidden_dim=32,
        num_classes=3,
        seed=42
    )

    code_rep = np.random.randn(16).astype(np.float32)
    exp_rep = np.random.randn(16).astype(np.float32)
    proc_rep = np.random.randn(16).astype(np.float32)

    probs, attn_weights = model.forward(code_rep, exp_rep, proc_rep)

    assert probs.shape == (3,)
    assert 0.99 <= float(np.sum(probs)) <= 1.01
    assert not np.isnan(probs).any()

    label, conf, dist = model.predict(code_rep, exp_rep, proc_rep)
    assert label in LABELS
    assert 0.0 <= conf <= 1.0
    assert len(dist) == 3
    assert set(dist.keys()) == set(LABELS)


def test_end_to_end_novelty_pipeline():
    """Verify complete pipeline from raw student inputs to comprehension assessment."""
    # 1. Inputs
    code = "def is_even(n):\n    return n % 2 == 0"
    explanation = "The modulo operator calculates remainder after division by 2. If remainder is 0, number is even."
    events = [
        {"event_type": "edit", "delta_chars": 15, "cursor_line": 1, "timestamp_s": 0.5},
        {"event_type": "edit", "delta_chars": 20, "cursor_line": 2, "timestamp_s": 1.5},
        {"event_type": "run", "delta_chars": 0, "cursor_line": 2, "timestamp_s": 2.5},
    ]

    # 2. Feature Extraction
    code_rep = CodeBERTEmbedder.extract_projected(code, target_dim=16)
    exp_rep = SentenceTransformerEmbedder.extract_projected(explanation, target_dim=16)
    proc_rep = ProcessTransformerEncoder().encode_sequence(events)

    # 3. Multimodal Fusion Model
    fusion_model = MultimodalTransformerFusionModel(seed=100)
    label, conf, dist = fusion_model.predict(code_rep, exp_rep, proc_rep)

    # 4. Output Assessment
    assert label in ["Understood", "Partial Understanding", "Possible Misconception"]
    assert conf > 0.0
    assert "Understood" in dist

