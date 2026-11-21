"""
DeepMisconception Diversity & Duplicate Detection Engine (Corrected)
Measures exact and AST structural code diversity, viva question/answer uniqueness,
and process event-type sequence entropy across all 5,000 sessions.
"""

import ast
import pandas as pd

class ASTStructureNormalizer(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = "id"
        return node
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            node.value = 0
        elif isinstance(node.value, str):
            node.value = "s"
        elif isinstance(node.value, bool):
            node.value = True
        return node
    def visit_arg(self, node):
        node.arg = "p"
        return node
    def visit_FunctionDef(self, node):
        node.name = "fn"
        node.decorator_list = []
        self.generic_visit(node)
        return node
    def visit_ClassDef(self, node):
        node.name = "Cls"
        node.decorator_list = []
        self.generic_visit(node)
        return node

def normalize_ast(code_str: str) -> str:
    try:
        tree = ast.parse(code_str)
        normalizer = ASTStructureNormalizer()
        return ast.dump(normalizer.visit(tree))
    except Exception:
        return "PARSE_ERROR"

def audit_dataset_diversity(code_df: pd.DataFrame, viva_df: pd.DataFrame, events_df: pd.DataFrame):
    # Code diversity
    n_code = len(code_df)
    unique_exact_code = code_df["python_code"].nunique()
    ast_representations = code_df["python_code"].apply(normalize_ast)
    unique_ast = ast_representations.nunique()
    
    # Viva diversity
    n_viva = len(viva_df)
    unique_q = viva_df["question"].nunique()
    unique_a = viva_df["student_answer"].nunique()
    
    # Process sequence diversity
    seqs = events_df.groupby("session_id")["event_type"].apply(tuple)
    unique_seqs = seqs.nunique()
    top_seq_freq = seqs.value_counts().iloc[0]
    top_seq_pct = round(top_seq_freq / len(seqs) * 100, 2)
    
    return {
        "code_total": n_code,
        "code_unique_exact": unique_exact_code,
        "code_unique_ast": unique_ast,
        "viva_total": n_viva,
        "viva_unique_questions": unique_q,
        "viva_unique_answers": unique_a,
        "process_sessions": len(seqs),
        "process_unique_sequences": unique_seqs,
        "process_top_seq_pct": top_seq_pct
    }
