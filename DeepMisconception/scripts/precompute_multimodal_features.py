#!/usr/bin/env python3
"""Precompute and cache multimodal representations for DeepMisconception.
Extracts CodeBERT (768-D), SentenceTransformer (384-D), Process Transformer (64-D),
and normalized 24-feature numerical vectors for all 5,000 sessions.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

DATA_DIR = Path("DeepMisconception_Corrected_Dataset")
MODELS_DIR = Path("DeepMisconception/models/complete_multimodal_model")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("backend/app/ml/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EXACT_24_FEATURES = [
    "n_events", "duration_s", "n_insert", "n_edit", "n_delete", "n_paste",
    "typed_chars", "paste_chars", "paste_char_ratio", "largest_insert_share",
    "n_runs", "n_syntax_errors", "n_runtime_errors", "n_errors", "n_debug_actions",
    "n_tests", "n_test_pass", "n_revisions", "n_errors_resolved", "idle_total_s",
    "max_idle_s", "mean_gap_s", "time_to_first_run_s", "final_state_pass"
]

CLASS_MAP = {
    "Understood": 0,
    "Partial Understanding": 1,
    "Possible Misconception": 2
}

def precompute_all_modalities():
    print("=" * 80)
    print("PRECOMPUTING MULTIMODAL FEATURE TENSORS FOR ALL 5,000 SESSIONS")
    print("=" * 80)

    # 1. Load session manifests
    sessions = pd.read_csv(DATA_DIR / "data/integrated/multimodal_sessions.csv")
    code_df = pd.read_csv(DATA_DIR / "data/code/code_submissions.csv").set_index("session_id")
    viva_df = pd.read_csv(DATA_DIR / "data/viva/viva_questions_answers.csv")
    features_df = pd.read_csv(DATA_DIR / "data/process/process_features_24.csv").set_index("session_id")

    # Load precomputed process embeddings from Process Transformer
    proc_data = torch.load("process_transformer_results/checkpoints/precomputed_process_embeddings.pt", weights_only=False)
    norm_stats = json.load(open("process_transformer_results/checkpoints/normalization_stats.json"))

    # Map process embeddings by session_id
    process_embs_map = {}
    for part in ["train", "validation", "test"]:
        sids = proc_data[part]["session_ids"]
        embs = proc_data[part]["embeddings"]
        for sid, emb in zip(sids, embs):
            process_embs_map[sid] = emb

    print(f"Mapped {len(process_embs_map)} Process Transformer 64-D embeddings.")

    # 2. Precompute CodeBERT Embeddings (768-D)
    print("\n--- Extracting CodeBERT (768-D) Representations ---")
    cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    cb_model = AutoModel.from_pretrained("microsoft/codebert-base")
    cb_model.eval()

    session_ids = sessions["session_id"].tolist()
    code_texts = [str(code_df.loc[sid, "python_code"]) for sid in session_ids]

    code_embs = []
    batch_size = 128
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(code_texts), batch_size):
            batch = code_texts[i:i + batch_size]
            inputs = cb_tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = cb_model(**inputs)
            cls_embs = out.last_hidden_state[:, 0, :].cpu()
            code_embs.append(cls_embs)
    code_embs_cat = torch.cat(code_embs, dim=0)
    print(f"CodeBERT extracted in {time.time() - t0:.1f}s. Shape: {code_embs_cat.shape}")

    # 3. Precompute SentenceTransformer Viva Embeddings (384-D)
    print("\n--- Extracting SentenceTransformer (384-D) Viva Representations ---")
    st_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    st_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    st_model.eval()

    # Group viva by session_id
    viva_grouped = viva_df.groupby("session_id")
    viva_texts_all = []
    viva_slices = []

    cur_idx = 0
    for sid in session_ids:
        grp = viva_grouped.get_group(sid)
        qa_texts = [f"Question: {row['question']} Answer: {row['student_answer']}" for _, row in grp.iterrows()]
        viva_texts_all.extend(qa_texts)
        viva_slices.append((cur_idx, cur_idx + len(qa_texts)))
        cur_idx += len(qa_texts)

    t0 = time.time()
    viva_encoded_list = []
    v_bs = 256
    with torch.no_grad():
        for i in range(0, len(viva_texts_all), v_bs):
            batch = viva_texts_all[i:i + v_bs]
            inp = st_tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = st_model(**inp)
            token_emb = out.last_hidden_state
            mask = inp["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            sum_emb = torch.sum(token_emb * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_emb / sum_mask
            # L2 normalize
            norm = torch.norm(mean_pooled, p=2, dim=1, keepdim=True).clamp(min=1e-9)
            viva_encoded_list.append((mean_pooled / norm).cpu())

    viva_encoded = torch.cat(viva_encoded_list, dim=0)

    # Mean pool per session
    viva_embs = []
    for s_start, s_end in viva_slices:
        session_qa = viva_encoded[s_start:s_end]
        viva_embs.append(session_qa.mean(dim=0, keepdim=True))
    viva_embs_cat = torch.cat(viva_embs, dim=0)
    print(f"SentenceTransformer viva extracted in {time.time() - t0:.1f}s. Shape: {viva_embs_cat.shape}")

    # 4. Prepare Normalized 24 Numerical Features
    means = np.array([norm_stats["means"][f] for f in EXACT_24_FEATURES], dtype=np.float32)
    stds = np.array([norm_stats["stds"][f] for f in EXACT_24_FEATURES], dtype=np.float32)

    raw_features = features_df.reindex(session_ids)[EXACT_24_FEATURES].astype(float).values
    norm_features = (raw_features - means) / stds
    num_features_tensor = torch.tensor(norm_features, dtype=torch.float32)
    print(f"Numerical features prepared. Shape: {num_features_tensor.shape}")

    # 5. Partition and Save
    dataset_partitions = {}
    for part in ["train", "validation", "test"]:
        part_mask = (sessions["partition"] == part).values
        part_sids = [session_ids[i] for i in range(len(session_ids)) if part_mask[i]]
        part_labels = [CLASS_MAP[sessions.iloc[i]["ground_truth_label"]] for i in range(len(session_ids)) if part_mask[i]]
        part_evidence = [sessions.iloc[i]["evidence_status"] for i in range(len(session_ids)) if part_mask[i]]
        part_families = [sessions.iloc[i]["scenario_family_id"] for i in range(len(session_ids)) if part_mask[i]]

        p_code = code_embs_cat[part_mask]
        p_viva = viva_embs_cat[part_mask]
        p_proc = torch.stack([process_embs_map[sid] for sid in part_sids], dim=0)
        p_num = num_features_tensor[part_mask]

        is_sufficient = torch.tensor([e == "Sufficient Evidence" for e in part_evidence], dtype=torch.bool)

        data_bundle = {
            "session_ids": part_sids,
            "scenario_families": part_families,
            "labels": torch.tensor(part_labels, dtype=torch.long),
            "evidence_status": part_evidence,
            "is_sufficient": is_sufficient,
            "code_embeddings": p_code,       # [N, 768]
            "viva_embeddings": p_viva,       # [N, 384]
            "process_embeddings": p_proc,    # [N, 64]
            "numerical_features": p_num      # [N, 24]
        }
        dataset_partitions[part] = data_bundle
        
        # Save cache files
        cache_path = CACHE_DIR / f"{part}_precomputed.pt"
        torch.save(data_bundle, cache_path)
        print(f"Partition '{part}': {len(part_sids)} sessions saved to {cache_path}")

    # Save to complete_multimodal_model as well
    out_all = MODELS_DIR / "precomputed_multimodal_features.pt"
    torch.save(dataset_partitions, out_all)
    print(f"\nAll precomputed multimodal representations saved to {out_all}")

if __name__ == "__main__":
    precompute_all_modalities()
