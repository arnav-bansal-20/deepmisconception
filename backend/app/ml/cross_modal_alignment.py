"""Cross-Modal Semantic Alignment Module for DeepMisconception.

Architecture Notice:
This module performs vector-level cross-modal interaction between pooled contextual
code representations (CodeBERT 768-D) and pooled student explanation representations
(Sentence-Transformer 384-D). It computes bidirectional cross-attention, contradiction
difference vectors (|c - e|), element-wise product interactions (c * e), and bilinear
similarity over projected pooled vectors. It does NOT operate on token-level sequence grids.
"""
from typing import Dict, Any, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from app.ml.config import CODE_DIM, EXP_DIM, ALIGN_DIM, FEATURE_CONTRACT


class CrossModalSemanticAlignment(nn.Module):
    """Dedicated Neural Alignment Module evaluating Code-Explanation consistency.
    
    Evaluates consistency between pooled contextual code representations and
    student explanations using vector-level bidirectional cross-attention,
    contradiction-sensitive difference and product interactions, and execution signals.
    """

    def __init__(
        self,
        code_dim: int = CODE_DIM,         # Pretrained CodeBERT dimension (768)
        exp_dim: int = EXP_DIM,           # Pretrained Sentence-Transformer dimension (384)
        ast_dim: int = 16,                # AST structural metric dimension
        exec_dim: int = 4,                # Execution outcome feature dimension
        d_align: int = 128,               # Common latent alignment dimension
        nhead: int = 4,                   # Multihead cross-attention heads
        alignment_dim: int = ALIGN_DIM,   # Output alignment representation dimension (64)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.exp_dim = exp_dim
        self.ast_dim = ast_dim
        self.exec_dim = exec_dim
        self.d_align = d_align
        self.alignment_dim = alignment_dim

        # 1. Modality Input Projections to uniform alignment latent space
        self.proj_code = nn.Sequential(
            nn.Linear(code_dim, d_align),
            nn.LayerNorm(d_align),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_exp = nn.Sequential(
            nn.Linear(exp_dim, d_align),
            nn.LayerNorm(d_align),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_evidence = nn.Sequential(
            nn.Linear(ast_dim + exec_dim, d_align),
            nn.LayerNorm(d_align),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Bidirectional Multi-Head Cross-Attention
        # A. Code Query attends to Explanation (which parts of explanation match code logic)
        self.cross_attn_code_to_exp = nn.MultiheadAttention(
            embed_dim=d_align,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_c2e = nn.LayerNorm(d_align)

        # B. Explanation Query attends to Code (which code constructs are cited or contradicted)
        self.cross_attn_exp_to_code = nn.MultiheadAttention(
            embed_dim=d_align,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_e2c = nn.LayerNorm(d_align)

        # 3. Bilinear Semantic Interaction Matrix for direct similarity representation
        self.bilinear_matrix = nn.Parameter(torch.empty(d_align, d_align))
        nn.init.xavier_uniform_(self.bilinear_matrix)

        # 4. Dense Alignment & Contradiction Representation Fusion
        # Concatenation of:
        # [attended_code (d_align) || attended_exp (d_align) || |c - e| (d_align) || c * e (d_align) || evidence (d_align) || bilinear_sim (1)]
        fusion_dim = (d_align * 5) + 1
        self.alignment_mlp = nn.Sequential(
            nn.Linear(fusion_dim, d_align * 2),
            nn.LayerNorm(d_align * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_align * 2, alignment_dim),
            nn.LayerNorm(alignment_dim),
        )

        # Auxiliary consistency score prediction head (calibrates alignment score in [0, 1])
        self.consistency_head = nn.Sequential(
            nn.Linear(alignment_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        code_emb: torch.Tensor,                # (B, code_dim) e.g. 768-D
        exp_emb: torch.Tensor,                 # (B, exp_dim) e.g. 384-D
        ast_features: Optional[torch.Tensor] = None,   # (B, ast_dim) e.g. 16-D
        exec_features: Optional[torch.Tensor] = None,  # (B, exec_dim) e.g. 4-D
        code_mask: Optional[torch.Tensor] = None,      # (B,) BoolTensor (True if code is missing/masked)
        exp_mask: Optional[torch.Tensor] = None,       # (B,) BoolTensor (True if viva is missing/masked)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute cross-modal semantic alignment.
        
        Args:
            code_emb: CodeBERT representation (B, 768)
            exp_emb: Sentence-Transformer representation (B, 384)
            ast_features: Structural AST features (B, 16) or None
            exec_features: Execution trace features (B, 4) or None
            code_mask: Optional boolean mask where True indicates missing code
            exp_mask: Optional boolean mask where True indicates missing viva
            
        Returns:
            alignment_rep: (B, alignment_dim) representation for Multimodal Fusion Transformer
            consistency_score: (B, 1) scalar consistency score in [0, 1]
            attention_maps: Dictionary of cross-attention weights
        """
        # Shape validation
        if code_emb.dim() == 1:
            code_emb = code_emb.unsqueeze(0)
        if exp_emb.dim() == 1:
            exp_emb = exp_emb.unsqueeze(0)

        batch_size = code_emb.size(0)
        device = code_emb.device

        # Validate feature dimensions
        if code_emb.size(-1) != self.code_dim:
            raise ValueError(f"Expected code_dim={self.code_dim}, got {code_emb.size(-1)}")
        if exp_emb.size(-1) != self.exp_dim:
            raise ValueError(f"Expected exp_dim={self.exp_dim}, got {exp_emb.size(-1)}")

        # Prepare default auxiliary features if not provided
        if ast_features is None:
            ast_features = torch.zeros(batch_size, self.ast_dim, device=device)
        elif ast_features.dim() == 1:
            ast_features = ast_features.unsqueeze(0)
        if ast_features.size(-1) != self.ast_dim:
            if ast_features.size(-1) < self.ast_dim:
                pad = torch.zeros(batch_size, self.ast_dim - ast_features.size(-1), device=device)
                ast_features = torch.cat([ast_features, pad], dim=-1)
            else:
                ast_features = ast_features[:, :self.ast_dim]

        if exec_features is None:
            exec_features = torch.zeros(batch_size, self.exec_dim, device=device)
        elif exec_features.dim() == 1:
            exec_features = exec_features.unsqueeze(0)
        if exec_features.size(-1) != self.exec_dim:
            if exec_features.size(-1) < self.exec_dim:
                pad = torch.zeros(batch_size, self.exec_dim - exec_features.size(-1), device=device)
                exec_features = torch.cat([exec_features, pad], dim=-1)
            else:
                exec_features = exec_features[:, :self.exec_dim]

        # 1. Project to shared alignment space: (B, d_align)
        h_code = self.proj_code(code_emb)
        h_exp = self.proj_exp(exp_emb)
        evidence_in = torch.cat([ast_features, exec_features], dim=-1)
        h_evidence = self.proj_evidence(evidence_in)

        # 2. Multi-Head Cross-Attention (treat each vector as 1 token: (B, 1, d_align))
        c_tok = h_code.unsqueeze(1)
        e_tok = h_exp.unsqueeze(1)

        # Code queries Explanation: (B, 1, d_align)
        attn_c2e, weights_c2e = self.cross_attn_code_to_exp(c_tok, e_tok, e_tok)
        attended_code = self.ln_c2e(h_code + attn_c2e.squeeze(1))

        # Explanation queries Code: (B, 1, d_align)
        attn_e2c, weights_e2c = self.cross_attn_exp_to_code(e_tok, c_tok, c_tok)
        attended_exp = self.ln_e2c(h_exp + attn_e2c.squeeze(1))

        # 3. Contradiction-Sensitive Interaction Features
        diff = torch.abs(h_code - h_exp)        # Difference vector: detects divergence
        prod = h_code * h_exp                   # Element-wise product: detects semantic overlap

        # Bilinear interaction scalar: h_code @ W @ h_exp^T: (B, 1)
        bilinear = torch.sum((h_code @ self.bilinear_matrix) * h_exp, dim=-1, keepdim=True) / math.sqrt(self.d_align)

        # 4. Dense Alignment Fusion
        interaction_vector = torch.cat(
            [attended_code, attended_exp, diff, prod, h_evidence, bilinear],
            dim=-1,
        )
        alignment_rep = self.alignment_mlp(interaction_vector)  # (B, alignment_dim)
        consistency_score = self.consistency_head(alignment_rep) # (B, 1)

        # 5. Gating: Mask alignment representation if code or viva is missing
        if code_mask is not None or exp_mask is not None:
            c_m = code_mask if code_mask is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
            e_m = exp_mask if exp_mask is not None else torch.zeros(batch_size, dtype=torch.bool, device=device)
            missing = (c_m | e_m)
            if missing.any():
                alignment_rep = alignment_rep.clone()
                alignment_rep[missing] = 0.0
                consistency_score = consistency_score.clone()
                consistency_score[missing] = 0.0

        attention_maps = {
            "code_to_exp_weights": weights_c2e,
            "exp_to_code_weights": weights_e2c,
        }

        return alignment_rep, consistency_score, attention_maps
