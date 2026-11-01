"""Python AST analysis engine for DeepMisconception.

Parses student Python source code, extracts syntactic structures, identifies
computational constructs, calculates complexity metrics, and generates code-specific
viva candidate concepts.
"""
import ast
import re
from typing import List, Dict, Any, Optional, Set
from app.schemas.ast_analysis import (
    ASTAnalysisResult,
    VariableInfo,
    LoopInfo,
    ConditionInfo,
    FunctionInfo,
    VivaCandidateConcept,
)


class ASTAnalyzer:
    """Performs deep AST inspection and concept extraction on Python code."""

    def __init__(self, code: str):
        self.code = code
        self.lines = code.splitlines()
        self.total_lines = len(self.lines)

    def analyze(self) -> ASTAnalysisResult:
        """Execute full AST parsing and extraction pipeline."""
        # 1. Check for syntax validity
        try:
            tree = ast.parse(self.code)
        except SyntaxError as e:
            return ASTAnalysisResult(
                is_syntax_valid=False,
                syntax_error_details={
                    "line": e.lineno or 1,
                    "column": e.offset or 1,
                    "message": str(e.msg),
                    "text": (e.text or "").strip(),
                    "diagnostic": self._format_syntax_diagnostic(e),
                },
                total_lines=self.total_lines,
                code_lines=self._count_code_lines(),
                comment_lines=self._count_comment_lines(),
            )

        # 2. Extract constructs
        visitor = CodeStructureVisitor(self.lines)
        visitor.visit(tree)

        # 3. Compute metrics
        complexity = self._calculate_cyclomatic_complexity(tree)
        code_lines = self._count_code_lines()
        comment_lines = self._count_comment_lines()

        # 4. Generate candidate concepts for viva questions
        candidates = self._identify_viva_concepts(
            visitor.variables,
            visitor.loops,
            visitor.conditions,
            visitor.functions,
            visitor.builtin_calls,
        )

        return ASTAnalysisResult(
            is_syntax_valid=True,
            syntax_error_details=None,
            variables=visitor.variables,
            loops=visitor.loops,
            conditions=visitor.conditions,
            functions=visitor.functions,
            total_lines=self.total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            cyclomatic_complexity=complexity,
            max_nesting_depth=visitor.max_nesting_depth,
            builtin_calls=sorted(list(visitor.builtin_calls)),
            candidate_concepts=candidates,
        )

    def _format_syntax_diagnostic(self, error: SyntaxError) -> str:
        line = error.lineno or 1
        msg = error.msg
        text = (error.text or "").strip()
        
        if "invalid syntax" in msg and ":" in text and not text.endswith(":"):
            return f"Syntax Error on line {line}: You might be missing a colon (:) at the end of the statement or have an unclosed bracket."
        if "unexpected EOF" in msg:
            return f"Syntax Error on line {line}: Unexpected end of file. Check for unclosed parentheses, brackets, or string literals."
        if "was never closed" in msg:
            return f"Syntax Error on line {line}: An opening parenthesis, bracket, or brace was never closed."
        if "indent" in msg.lower():
            return f"Indentation Error on line {line}: Inconsistent or missing indentation block."
        return f"Syntax Error on line {line}: {msg}. Code: '{text}'"

    def _count_code_lines(self) -> int:
        return sum(1 for line in self.lines if line.strip() and not line.strip().startswith("#"))

    def _count_comment_lines(self) -> int:
        return sum(1 for line in self.lines if line.strip().startswith("#"))

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """McCabe cyclomatic complexity: 1 + number of decision points."""
        decision_points = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
                decision_points += 1
            elif isinstance(node, ast.BoolOp):
                # 'and' / 'or' add decision paths
                decision_points += len(node.values) - 1
        return 1 + decision_points

    def _identify_viva_concepts(
        self,
        variables: List[VariableInfo],
        loops: List[LoopInfo],
        conditions: List[ConditionInfo],
        functions: List[FunctionInfo],
        builtin_calls: Set[str],
    ) -> List[VivaCandidateConcept]:
        """Automatically identify pedagogical concepts directly grounded in student code."""
        concepts: List[VivaCandidateConcept] = []

        # Concept 1: Recursive Base Case and Call Stack
        for fn in functions:
            if fn.is_recursive:
                focal = self._get_code_slice(fn.line_number, min(fn.line_number + 8, fn.end_line_number or fn.line_number + 8))
                concepts.append(
                    VivaCandidateConcept(
                        concept_key="recursion-base-case-and-stack",
                        concept_name="Recursion Termination & Base Case Invariants",
                        line_number=fn.line_number,
                        focal_code=focal,
                        description=f"Function '{fn.name}' recursively invokes itself. The viva can probe how the base case terminates the call stack.",
                        suggested_difficulty="hard",
                    )
                )

        # Concept 2: Loop Boundary and Off-by-One Invariants
        for loop in loops:
            focal = self._get_code_slice(loop.line_number, min(loop.line_number + 4, loop.end_line_number or loop.line_number + 4))
            iterable = loop.iterable_expression or ""
            if "range" in iterable:
                concepts.append(
                    VivaCandidateConcept(
                        concept_key="loop-bounds-and-range-invariants",
                        concept_name="Loop Range Bounds & Off-by-One Awareness",
                        line_number=loop.line_number,
                        focal_code=focal,
                        description=f"Loop on line {loop.line_number} uses '{iterable}'. The viva can test exact start/end iteration values and off-by-one boundaries.",
                        suggested_difficulty="medium",
                    )
                )
            if loop.has_continue:
                concepts.append(
                    VivaCandidateConcept(
                        concept_key="loop-continue-control-flow",
                        concept_name="Loop Cycle Skip via 'continue'",
                        line_number=loop.line_number,
                        focal_code=focal,
                        description="Loop contains a 'continue' statement. The viva can test whether the student understands which lines get skipped.",
                        suggested_difficulty="medium",
                    )
                )

        # Concept 3: Modulo Arithmetic & Conditional Branching
        for cond in conditions:
            if "%" in cond.test_expression:
                focal = self._get_code_slice(cond.line_number, cond.line_number + 3)
                concepts.append(
                    VivaCandidateConcept(
                        concept_key="modulo-conditional-filtering",
                        concept_name="Modulo Arithmetic & Even/Divisibility Filtering",
                        line_number=cond.line_number,
                        focal_code=focal,
                        description=f"Condition '{cond.test_expression}' on line {cond.line_number} tests remainder. The viva can test what numbers satisfy this branch.",
                        suggested_difficulty="easy",
                    )
                )

        # Concept 4: Variable Accumulation / State Updates
        accumulators = [v for v in variables if v.name in ("total", "sum", "acc", "count", "res", "result")]
        if accumulators and loops:
            acc = accumulators[0]
            focal = self._get_code_slice(acc.line_number, min(acc.line_number + 4, self.total_lines))
            concepts.append(
                VivaCandidateConcept(
                    concept_key="variable-accumulator-invariants",
                    concept_name="Accumulator Variable Lifecycle & State Evolution",
                    line_number=acc.line_number,
                    focal_code=focal,
                    description=f"Variable '{acc.name}' accumulates state across computations. The viva can test trace values before and after loop passes.",
                    suggested_difficulty="medium",
                )
            )

        # Concept 5: Function Return Placement & Scope
        for fn in functions:
            if fn.has_return and loops:
                # Check if return is inside or outside loop
                for r_line in fn.return_line_numbers:
                    for loop in loops:
                        if loop.line_number < r_line <= (loop.end_line_number or loop.line_number + 10):
                            focal = self._get_code_slice(loop.line_number, r_line)
                            concepts.append(
                                VivaCandidateConcept(
                                    concept_key="early-return-in-loop",
                                    concept_name="Premature Loop Termination via Return",
                                    line_number=r_line,
                                    focal_code=focal,
                                    description=f"Return statement on line {r_line} is positioned inside the loop on line {loop.line_number}. Probes early termination.",
                                    suggested_difficulty="medium",
                                )
                            )

        # Fallback if no specific pattern triggered: General Control Flow
        if not concepts and (variables or loops or conditions or functions):
            primary_line = functions[0].line_number if functions else (loops[0].line_number if loops else 1)
            concepts.append(
                VivaCandidateConcept(
                    concept_key="general-algorithmic-control-flow",
                    concept_name="Algorithmic Execution Order & Variable Tracing",
                    line_number=primary_line,
                    focal_code=self._get_code_slice(1, min(10, self.total_lines)),
                    description="General program logic and variable updates across statement execution.",
                    suggested_difficulty="easy",
                )
            )

        return concepts

    def _get_code_slice(self, start_line: int, end_line: int) -> str:
        s = max(0, start_line - 1)
        e = min(self.total_lines, end_line)
        return "\n".join(self.lines[s:e])


class CodeStructureVisitor(ast.NodeVisitor):
    """Visits AST nodes to extract variables, loops, conditions, and functions."""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.variables: List[VariableInfo] = []
        self.loops: List[LoopInfo] = []
        self.conditions: List[ConditionInfo] = []
        self.functions: List[FunctionInfo] = []
        self.builtin_calls: Set[str] = set()
        
        self.current_scope = "global"
        self.current_function: Optional[FunctionInfo] = None
        self.nesting_depth = 0
        self.max_nesting_depth = 0
        self._seen_vars: Set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_scope = self.current_scope
        self.current_scope = node.name
        
        args = [arg.arg for arg in node.args.args]
        fn_info = FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            end_line_number=getattr(node, "end_lineno", node.lineno),
            arguments=args,
            has_return=False,
            return_line_numbers=[],
            is_recursive=False,
            docstring=ast.get_docstring(node),
        )
        self.functions.append(fn_info)
        prev_fn = self.current_function
        self.current_function = fn_info

        self.nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_depth)
        
        self.generic_visit(node)
        
        self.nesting_depth -= 1
        self.current_function = prev_fn
        self.current_scope = prev_scope

    def visit_Return(self, node: ast.Return):
        if self.current_function:
            self.current_function.has_return = True
            self.current_function.return_line_numbers.append(node.lineno)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_depth)

        target_var = ast.unparse(node.target) if hasattr(ast, "unparse") else "i"
        iter_expr = ast.unparse(node.iter) if hasattr(ast, "unparse") else ""

        has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
        has_continue = any(isinstance(n, ast.Continue) for n in ast.walk(node))

        self.loops.append(
            LoopInfo(
                loop_type="for",
                line_number=node.lineno,
                end_line_number=getattr(node, "end_lineno", node.lineno),
                target_variable=target_var,
                iterable_expression=iter_expr,
                has_break=has_break,
                has_continue=has_continue,
                nesting_depth=self.nesting_depth,
            )
        )

        self.generic_visit(node)
        self.nesting_depth -= 1

    def visit_While(self, node: ast.While):
        self.nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_depth)

        test_expr = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
        has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
        has_continue = any(isinstance(n, ast.Continue) for n in ast.walk(node))

        self.loops.append(
            LoopInfo(
                loop_type="while",
                line_number=node.lineno,
                end_line_number=getattr(node, "end_lineno", node.lineno),
                iterable_expression=test_expr,
                has_break=has_break,
                has_continue=has_continue,
                nesting_depth=self.nesting_depth,
            )
        )

        self.generic_visit(node)
        self.nesting_depth -= 1

    def visit_If(self, node: ast.If):
        self.nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.nesting_depth)

        test_expr = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
        has_elif = any(isinstance(n, ast.If) for n in node.orelse)
        has_else = bool(node.orelse and not has_elif)

        self.conditions.append(
            ConditionInfo(
                line_number=node.lineno,
                test_expression=test_expr,
                has_elif=has_elif,
                has_else=has_else,
                nesting_depth=self.nesting_depth,
            )
        )

        self.generic_visit(node)
        self.nesting_depth -= 1

    def visit_Assign(self, node: ast.Assign):
        val_type = type(node.value).__name__
        for target in node.targets:
            if isinstance(target, ast.Name):
                key = f"{self.current_scope}:{target.id}"
                self.variables.append(
                    VariableInfo(
                        name=target.id,
                        line_number=node.lineno,
                        is_assignment=True,
                        assigned_value_type=val_type,
                        target_scope=self.current_scope,
                    )
                )
                self._seen_vars.add(key)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name):
            self.variables.append(
                VariableInfo(
                    name=node.target.id,
                    line_number=node.lineno,
                    is_assignment=True,
                    assigned_value_type="AugmentedAssign",
                    target_scope=self.current_scope,
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Check for recursion
            if self.current_function and self.current_function.name == func_name:
                self.current_function.is_recursive = True

            # Check for builtins
            if func_name in (
                "print", "range", "len", "sum", "min", "max",
                "enumerate", "zip", "sorted", "reversed", "map", "filter", "input"
            ):
                self.builtin_calls.add(func_name)

        self.generic_visit(node)
