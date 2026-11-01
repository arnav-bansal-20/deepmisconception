"""Modular Feature Extraction pipeline across AST, Explanation, and Process modalities."""
import ast
import re
import math
from typing import Dict, Any, List, Optional, Union
import numpy as np
from app.ml.codebert_embedder import CodeBERTEmbedder


class ASTFeatureExtractor:
    """Extracts a fixed 16-dimensional structural feature vector from Python code."""

    FEATURE_DIM = 16

    @classmethod
    def extract(cls, code: str, ast_summary: Optional[Dict[str, Any]] = None) -> np.ndarray:
        vec = np.zeros(cls.FEATURE_DIM, dtype=np.float32)
        if not code or not code.strip():
            return vec

        try:
            tree = ast.parse(code)
            has_syntax_error = 0.0
        except SyntaxError:
            tree = None
            has_syntax_error = 1.0

        if tree is None:
            vec[14] = has_syntax_error
            return vec

        # Walk AST nodes and aggregate counts
        nodes = list(ast.walk(tree))
        total_nodes = len(nodes)
        
        funcs = sum(1 for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        for_loops = sum(1 for n in nodes if isinstance(n, (ast.For, ast.AsyncFor)))
        while_loops = sum(1 for n in nodes if isinstance(n, ast.While))
        if_branches = sum(1 for n in nodes if isinstance(n, ast.If))
        assigns = sum(1 for n in nodes if isinstance(n, (ast.Assign, ast.AugAssign)))
        returns = sum(1 for n in nodes if isinstance(n, ast.Return))
        calls = sum(1 for n in nodes if isinstance(n, ast.Call))
        bin_ops = sum(1 for n in nodes if isinstance(n, ast.BinOp))
        compares = sum(1 for n in nodes if isinstance(n, ast.Compare))

        # Check recursion
        func_names = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
        has_recursion = 0.0
        for n in nodes:
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id in func_names:
                    has_recursion = 1.0
                    break

        lines = [line for line in code.split("\n") if line.strip()]
        total_lines = len(lines)
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        comment_ratio = (comment_lines / total_lines) if total_lines > 0 else 0.0

        # Nesting depth calculation
        max_depth = 1
        for line in lines:
            indent = len(line) - len(line.lstrip(" "))
            depth = (indent // 4) + 1
            if depth > max_depth:
                max_depth = depth

        cyclomatic_complexity = 1 + if_branches + for_loops + while_loops

        # Populate normalized 16-D vector
        vec[0] = min(1.0, cyclomatic_complexity / 10.0)
        vec[1] = min(1.0, total_nodes / 100.0)
        vec[2] = min(1.0, total_lines / 50.0)
        vec[3] = min(1.0, max_depth / 8.0)
        vec[4] = min(1.0, funcs / 4.0)
        vec[5] = min(1.0, for_loops / 5.0)
        vec[6] = min(1.0, while_loops / 4.0)
        vec[7] = min(1.0, if_branches / 6.0)
        vec[8] = min(1.0, assigns / 10.0)
        vec[9] = min(1.0, returns / 4.0)
        vec[10] = min(1.0, calls / 10.0)
        vec[11] = min(1.0, bin_ops / 10.0)
        vec[12] = min(1.0, compares / 8.0)
        vec[13] = has_recursion
        vec[14] = has_syntax_error
        vec[15] = min(1.0, comment_ratio)

        return vec


class ExplanationFeatureExtractor:
    """Extracts a fixed 16-dimensional linguistic/conceptual feature vector from viva explanations."""

    FEATURE_DIM = 16

    CONCEPT_KEYWORDS = {
        "loop": ["loop", "iteration", "iterate", "while", "for", "range", "step"],
        "boundary": ["boundary", "edge", "limit", "bound", "off-by-one", "end", "stop", "inclusive"],
        "condition": ["if", "branch", "condition", "skip", "continue", "filter", "modulo", "remainder"],
        "state": ["accumulator", "total", "sum", "state", "variable", "update", "increment"],
        "recursion": ["recursion", "recursive", "base case", "call stack", "unwind", "overflow"],
        "logic": ["and", "or", "not", "equal", "greater", "less"],
        "causality": ["because", "therefore", "ensures", "prevents", "causes", "handles", "since"],
        "evasive": ["idk", "dunno", "maybe", "guess", "not sure", "don't know", "dont know", "works though"],
    }

    @classmethod
    def extract(cls, text: str, code_context: Optional[str] = None) -> np.ndarray:
        vec = np.zeros(cls.FEATURE_DIM, dtype=np.float32)
        if not text or not text.strip():
            return vec

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        total_words = len(words)
        if total_words == 0:
            return vec

        # 1-8: Concept Keyword Density
        for idx, (cat, kws) in enumerate(cls.CONCEPT_KEYWORDS.items()):
            hits = sum(1 for kw in kws if kw in text_lower)
            vec[idx] = min(1.0, hits / 4.0)

        # 9-16: Structural text metrics
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = max(1, len(sentences))
        unique_words = len(set(words))
        avg_word_len = sum(len(w) for w in words) / total_words

        # Code identifier citations
        citations = 0
        if code_context:
            identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code_context))
            # Ignore standard keywords
            ignore = {"def", "return", "for", "if", "in", "range", "import", "print", "while", "pass"}
            identifiers -= ignore
            for ident in identifiers:
                if ident.lower() in text_lower:
                    citations += 1

        vec[8] = min(1.0, math.log1p(len(text)) / 7.0)        # Text length (log scaled)
        vec[9] = min(1.0, math.log1p(total_words) / 5.0)      # Word count (log scaled)
        vec[10] = min(1.0, sentence_count / 6.0)              # Sentence count
        vec[11] = min(1.0, avg_word_len / 10.0)               # Average word length
        vec[12] = min(1.0, citations / 5.0)                   # Code identifier citations
        vec[13] = 1.0 if any(w in text_lower for w in ["correct", "true", "guarantees", "always"]) else 0.0
        vec[14] = min(1.0, sum(1 for w in ["not", "never", "no", "fails", "doesn't", "doesnt"] if w in words) / 3.0)
        vec[15] = unique_words / total_words                  # Vocabulary diversity

        return vec


class ProcessFeatureExtractor:
    """Extracts a fixed 8-dimensional behavioral sequence vector from coding telemetry."""

    FEATURE_DIM = 8

    @classmethod
    def extract(cls, features: Any) -> np.ndarray:
        vec = np.zeros(cls.FEATURE_DIM, dtype=np.float32)
        if features is None:
            return vec

        # Support dictionary or object model
        get_val = lambda key, default: getattr(features, key, None) if hasattr(features, key) else (features.get(key, default) if isinstance(features, dict) else default)

        paste_ratio = float(get_val("paste_ratio", 0.0) or 0.0)
        burstiness_ratio = float(get_val("burstiness_ratio", 0.0) or 0.0)
        duration_s = float(get_val("coding_duration_seconds", 0.0) or 0.0)
        total_events = float(get_val("total_events", 0) or 0)
        run_count = float(get_val("run_count", 0) or 0)
        error_count = float(get_val("error_count", 0) or 0)
        recovery_rate = float(get_val("error_recovery_rate", 1.0) or 1.0)
        cpm = float(get_val("typing_speed_cpm", 0.0) or 0.0)

        vec[0] = min(1.0, max(0.0, paste_ratio))
        vec[1] = min(1.0, max(0.0, burstiness_ratio))
        vec[2] = min(1.0, math.log1p(duration_s) / 6.0)
        vec[3] = min(1.0, math.log1p(total_events) / 6.0)
        vec[4] = min(1.0, run_count / 10.0)
        vec[5] = min(1.0, error_count / 5.0)
        vec[6] = min(1.0, max(0.0, recovery_rate))
        vec[7] = min(1.0, cpm / 300.0)

        return vec


class MultimodalFeatureExtractor:
    """Combines AST (16-D), Explanation (16-D), Process (8-D), and optional CodeBERT (768-D)."""

    TOTAL_DIM = 40
    CODEBERT_DIM = 768

    @classmethod
    def is_codebert_available(cls) -> bool:
        """Check if CodeBERT is available in the current runtime."""
        return CodeBERTEmbedder.is_available()

    @classmethod
    def extract_codebert(cls, code: str) -> np.ndarray:
        """Extract 768-dimensional CodeBERT embedding directly."""
        return CodeBERTEmbedder.extract_embedding(code)

    @classmethod
    def extract(
        cls,
        code: str,
        explanation: str,
        process_features: Any,
        ast_summary: Optional[Dict[str, Any]] = None,
        modality_mask: Optional[str] = "all",  # all, code_only, explanation_only, process_only, code_plus_explanation
        blend_codebert: bool = False,
    ) -> np.ndarray:
        ast_vec = ASTFeatureExtractor.extract(code, ast_summary)
        exp_vec = ExplanationFeatureExtractor.extract(explanation, code)
        proc_vec = ProcessFeatureExtractor.extract(process_features)

        # If CodeBERT blending is requested, fuse 16-D pooled CodeBERT into code representation
        if blend_codebert and CodeBERTEmbedder.is_available():
            codebert_proj = CodeBERTEmbedder.extract_projected(code, target_dim=16)
            ast_vec = 0.5 * ast_vec + 0.5 * codebert_proj

        # Apply modality ablation masks if requested
        if modality_mask == "code_only":
            exp_vec = np.zeros_like(exp_vec)
            proc_vec = np.zeros_like(proc_vec)
        elif modality_mask == "explanation_only":
            ast_vec = np.zeros_like(ast_vec)
            proc_vec = np.zeros_like(proc_vec)
        elif modality_mask == "process_only":
            ast_vec = np.zeros_like(ast_vec)
            exp_vec = np.zeros_like(exp_vec)
        elif modality_mask == "code_plus_explanation":
            proc_vec = np.zeros_like(proc_vec)

        return np.concatenate([ast_vec, exp_vec, proc_vec], axis=0).astype(np.float32)
