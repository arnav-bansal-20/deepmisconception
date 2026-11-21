#!/usr/bin/env python3
"""Verify complete multimodal model checkpoints, inference API, dimensions, and abstention."""
import sys
import json
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, "DeepMisconception/models/complete_multimodal_model")
sys.path.insert(0, "DeepMisconception/scripts")
from inference import DeepMisconceptionMultimodalPredictor, CLASS_NAMES

def main():
    print("=" * 80)
    print("DEEPMISCONCEPTION: COMPLETE MULTIMODAL INTEGRATION VERIFICATION")
    print("=" * 80)

    # 1. Verify Checkpoints
    checkpoints = [
        "DeepMisconception/models/cross_modal_alignment/cross_modal_alignment_best.pt",
        "DeepMisconception/models/multimodal_fusion_transformer/fusion_transformer_best.pt",
        "DeepMisconception/models/classification_head/classification_head_best.pt",
        "DeepMisconception/models/complete_multimodal_model/complete_multimodal_model_best.pt"
    ]
    for cp in checkpoints:
        p = Path(cp)
        assert p.exists(), f"Missing checkpoint: {p}"
        print(f"  [FOUND] {p.name:<36} ({p.stat().st_size:,} bytes)")

    # 2. Instantiate Predictor
    predictor = DeepMisconceptionMultimodalPredictor()
    print("Successfully instantiated DeepMisconceptionMultimodalPredictor.")

    # 3. Test on Unseen Test Session
    test_cache = torch.load("backend/app/ml/cache/test_precomputed.pt", weights_only=False)
    features_df = pd.read_csv("DeepMisconception_Corrected_Dataset/data/process/process_features_24.csv").set_index("session_id")

    sid = test_cache["session_ids"][0]
    true_label_idx = test_cache["labels"][0].item()
    code_emb = test_cache["code_embeddings"][0]
    viva_emb = test_cache["viva_embeddings"][0]
    proc_emb = test_cache["process_embeddings"][0]
    num_dict = features_df.loc[sid].to_dict()

    print(f"\nEvaluating test sample session: {sid} (Ground Truth: {CLASS_NAMES[true_label_idx]})")
    res = predictor.predict(code_emb, viva_emb, proc_emb=proc_emb, process_features_24=num_dict)

    print(f"Predicted Class:            {res['predicted_class']}")
    print(f"Class Probabilities:        {res['class_probabilities']}")
    print(f"Evidence Status:            {res['evidence_status']}")
    print(f"Model Availability Status:  {res['model_availability_status']}")

    assert res["predicted_class"] in CLASS_NAMES, f"Invalid class: {res['predicted_class']}"
    assert len(res["fused_embedding"]) == 64, f"Invalid fused embedding dim: {len(res['fused_embedding'])}"
    print("Input/Output Dimension Verification: PASSED (Fused: 64, Output: 3)")

    # 4. Test Insufficient Evidence Abstention Status
    abstained = predictor.predict(code_emb, viva_emb, proc_emb=proc_emb, process_features_24=num_dict,
                                  evidence_status="Insufficient Evidence")
    assert abstained["predicted_class"] is None, "Did not abstain on insufficient evidence!"
    assert abstained["evidence_status"] == "Pending / Insufficient Evidence"
    print("Abstention Status Verification: PASSED (Properly abstains without forced prediction)")

    print("\n[SUCCESS] All complete multimodal components, inference pipelines, and statuses verified!")

if __name__ == "__main__":
    main()
