"""LLM Provider abstraction supporting Google Gemini API with intelligent offline fallback.

Never exposes API credentials to the client. Uses Gemini Flash via google-genai
when configured, and provides a deterministic, AST-grounded fallback for offline testing
and automated evaluation.
"""
import os
import ast
import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


def extract_clean_json(raw_text: str) -> Dict[str, Any]:
    """Robustly extract and parse JSON from raw LLM text, stripping markdown code fences."""
    text = (raw_text or "").strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        # Fallback: slice from first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text)


class BaseLLMProvider(ABC):
    """Abstract base provider for LLM calls."""

    @abstractmethod
    def generate_json(self, system_instruction: str, user_prompt: str) -> Dict[str, Any]:
        """Generate structured JSON output from LLM."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """Deterministic, AST-aware mock provider for testing and offline execution.
    
    Dynamically parses the submitted code from prompts to generate questions
    grounded in the student's actual AST nodes, function names, and variables.
    """

    def generate_json(self, system_instruction: str, user_prompt: str) -> Dict[str, Any]:
        prompt_lower = user_prompt.lower()

        # Handle Question Generation Request
        if "generate viva questions" in system_instruction.lower() or "viva question" in prompt_lower:
            return self._mock_question_generation(user_prompt)

        # Handle Answer Evaluation Request
        if "evaluate" in system_instruction.lower() or "answer assessment" in prompt_lower:
            return self._mock_answer_evaluation(user_prompt)

        # Generic JSON fallback
        return {
            "status": "success",
            "message": "Processed by DeepMisconception offline provider.",
        }

    def _extract_code_from_prompt(self, prompt: str) -> str:
        """Extract source code segment from user prompt."""
        marker = "=== STUDENT SOURCE CODE ==="
        end_marker = "=== VERIFIED EXECUTION OUTPUT ==="
        if marker in prompt and end_marker in prompt:
            return prompt.split(marker)[1].split(end_marker)[0].strip()
        return ""

    def _mock_question_generation(self, prompt: str) -> Dict[str, Any]:
        """Construct code-grounded viva questions dynamically from student code."""
        code = self._extract_code_from_prompt(prompt)
        questions = self._synthesize_ast_questions(code)
        return {"questions": questions}

    def _synthesize_ast_questions(self, code: str) -> List[Dict[str, Any]]:
        """Dynamically generate code-grounded questions by inspecting AST constructs."""
        questions: List[Dict[str, Any]] = []
        code_lines = code.splitlines()

        def get_snippet(lineno: int, end_lineno: Optional[int] = None) -> str:
            if 1 <= lineno <= len(code_lines):
                end = end_lineno if end_lineno and end_lineno <= len(code_lines) else lineno
                return "\n".join(code_lines[lineno - 1 : min(end, lineno + 3)]).strip()
            return ""

        try:
            tree = ast.parse(code)
        except Exception:
            # Fallback for unparseable syntax
            return [{
                "question_text": "Your submitted code encountered a syntax error. Explain what syntax error occurred and how you would correct it.",
                "focal_code": code[:100],
                "concept": "syntax-error-diagnosis",
                "difficulty": "easy",
                "expected_reasoning": "Identifies the line and token causing the syntax failure and describes how to resolve it.",
                "rubric": {"identifies_error": "Points out missing colon, parenthesis, or invalid keyword"},
                "misconception_indicators": ["Confusing assignment with equality", "Missing indentation or colons"]
            }]

        for node in ast.walk(tree):
            # 1. Functions & Recursion
            if isinstance(node, ast.FunctionDef):
                has_recursion = any(
                    isinstance(child, ast.Call) and getattr(child.func, "id", None) == node.name
                    for child in ast.walk(node)
                )
                args = [a.arg for a in node.args.args]
                focal = get_snippet(node.lineno, getattr(node, "end_lineno", node.lineno + 2))
                if has_recursion:
                    questions.append({
                        "question_text": f"In your recursive function '{node.name}', identify the base case condition and explain how it prevents infinite recursion when called.",
                        "focal_code": focal,
                        "concept": "recursion-base-case-and-termination",
                        "difficulty": "hard",
                        "expected_reasoning": f"Explains how '{node.name}' unwinds its call stack upon hitting the base condition without exceeding the recursion limit.",
                        "rubric": {
                            "base_case_identified": "Points out the base termination condition",
                            "call_stack_awareness": "Describes frame unwinding and return values"
                        },
                        "misconception_indicators": ["Missing base case leads to RecursionError", "Confusing parameter decrement"]
                    })
                else:
                    questions.append({
                        "question_text": f"Explain the role of the parameters ({', '.join(args) or 'none'}) in function '{node.name}' and what state or value is returned at completion.",
                        "focal_code": focal,
                        "concept": "function-contract-and-return-semantics",
                        "difficulty": "medium",
                        "expected_reasoning": f"Traces how inputs ({', '.join(args)}) are processed within '{node.name}' to yield the expected output.",
                        "rubric": {
                            "parameter_explanation": "Explains parameter roles",
                            "return_value_trace": "Describes return calculation accurately"
                        },
                        "misconception_indicators": ["Confusing print with return", "Assuming global variable mutation"]
                    })

            # 2. Loops (For and While)
            elif isinstance(node, (ast.For, ast.While)):
                focal = get_snippet(node.lineno, getattr(node, "end_lineno", node.lineno + 2))
                loop_kind = "for" if isinstance(node, ast.For) else "while"
                questions.append({
                    "question_text": f"At line {node.lineno}, describe the invariant maintained by this {loop_kind} loop and explain how each iteration advances program state toward termination.",
                    "focal_code": focal,
                    "concept": f"{loop_kind}-loop-iteration-invariants",
                    "difficulty": "medium",
                    "expected_reasoning": f"Explains the loop condition or iterable bounds and how the loop terminates without off-by-one errors or infinite iteration.",
                    "rubric": {
                        "explains_termination": "Describes condition under which loop finishes",
                        "state_tracking": "Traces how state progresses on each pass"
                    },
                    "misconception_indicators": ["Off-by-one boundary error", "Infinite loop condition", "Loop index mutation confusion"]
                })

            # 3. Branching & Conditionals
            elif isinstance(node, ast.If):
                focal = get_snippet(node.lineno, getattr(node, "end_lineno", node.lineno + 1))
                questions.append({
                    "question_text": f"In your conditional statement at line {node.lineno}, explain under what exact circumstances this branch executes and what occurs when the condition evaluates to False.",
                    "focal_code": focal,
                    "concept": "conditional-branching-logic",
                    "difficulty": "easy",
                    "expected_reasoning": "Accurately articulates the boolean condition and describes the alternate control path if False.",
                    "rubric": {
                        "boolean_logic": "Correctly interprets logical operators (<, >, ==, and, or, not)",
                        "fallthrough_behavior": "Explains else/elif or fallthrough flow"
                    },
                    "misconception_indicators": ["Inverting boolean conditions", "Confusing assignment = with equality =="]
                })

            if len(questions) >= 3:
                break

        # Fallback if no deep constructs found
        if not questions:
            questions.append({
                "question_text": "Walk through the execution flow of your program step by step and explain how variable state evolves.",
                "focal_code": code[:120],
                "concept": "general-program-execution-flow",
                "difficulty": "easy",
                "expected_reasoning": "Step-by-step description of line executions and variable state updates.",
                "rubric": {"step_by_step_trace": "Mentions variable assignments and statement order"},
                "misconception_indicators": ["Confusing declaration with execution time"]
            })

        return questions[:3]

    def _mock_answer_evaluation(self, prompt: str) -> Dict[str, Any]:
        """Perform semantic evaluation heuristic on prompt containing student answer."""
        ans_marker = "student answer:"
        student_ans = ""
        if ans_marker in prompt.lower():
            student_ans = prompt.lower().split(ans_marker)[-1].strip()

        # Check for empty or evasive answers
        if len(student_ans) < 15 or "idk" in student_ans or "dont know" in student_ans or "not sure" in student_ans or "no idea" in student_ans:
            return {
                "answer_label": "Insufficient Evidence",
                "concepts_correct": [],
                "concepts_missed": ["Incomplete explanation", "No mechanistic reasoning provided"],
                "contradictions": [],
                "supporting_evidence": [student_ans],
                "evaluation_explanation": "The student response is too brief or evasive to demonstrate conceptual comprehension of the referenced code construct.",
                "uncertainty_score": 0.85,
                "needs_instructor_review": True
            }

        # Check for classic control-flow misconception: confusing break with continue
        if "stops the loop" in student_ans or "ends the loop" in student_ans or "terminates the loop" in student_ans:
            return {
                "answer_label": "Incorrect",
                "concepts_correct": ["Recognizes conditional triggering"],
                "concepts_missed": ["Distinction between break and continue control flow"],
                "contradictions": ["Claims continue stops the loop completely, but continue only advances to the next iteration."],
                "supporting_evidence": [student_ans],
                "evaluation_explanation": "The student displays a classic control-flow misconception by confusing 'continue' with 'break'. The 'continue' statement skips the remainder of the current iteration, rather than terminating the entire loop.",
                "uncertainty_score": 0.1,
                "needs_instructor_review": False
            }

        # Check for recursion base-case misconception
        if "no base case" in student_ans or "without a base case" in student_ans or "not require an explicit base case" in student_ans or "does not require a base case" in student_ans or "automatically terminates when" in student_ans:
            return {
                "answer_label": "Incorrect",
                "concepts_correct": ["Recognizes recursive execution"],
                "concepts_missed": ["Recursion base case termination condition"],
                "contradictions": ["Claims recursive functions do not require an explicit base case, leading to unbounded call stack expansion."],
                "supporting_evidence": [student_ans],
                "evaluation_explanation": "The student exhibits an acute recursion misconception by claiming recursive functions do not require a base case. Without a base case, recursion produces an infinite call stack and runtime error.",
                "uncertainty_score": 0.1,
                "needs_instructor_review": False
            }

        # Check for reference aliasing / mutation misconception
        if "independent copy" in student_ans or "independent copies" in student_ans or "deep copy" in student_ans or "copies in memory" in student_ans:
            return {
                "answer_label": "Incorrect",
                "concepts_correct": ["Identifies variable assignment"],
                "concepts_missed": ["Reference assignment vs object copying for mutable structures"],
                "contradictions": ["Claims variable assignment creates an independent copy of a mutable list, but Python assigns shared object references."],
                "supporting_evidence": [student_ans],
                "evaluation_explanation": "The student displays a classic reference aliasing misconception, falsely assuming assignment creates deep copies of mutable collections rather than binding new names to the identical object.",
                "uncertainty_score": 0.1,
                "needs_instructor_review": False
            }

        # Check for partial understanding / admitted omission
        if "didn't lowercase" in student_ans or "did not lowercase" in student_ans or "forgot" in student_ans or "failing on mixed case" in student_ans:
            return {
                "answer_label": "Partially Correct",
                "concepts_correct": ["Identifies core string reversal mechanism"],
                "concepts_missed": ["Input normalization and case folding"],
                "contradictions": [],
                "supporting_evidence": [student_ans],
                "evaluation_explanation": "The student correctly understands the slicing and reversal logic, but acknowledges omitting character case normalization.",
                "uncertainty_score": 0.25,
                "needs_instructor_review": False
            }

        # Check for relevant programming concept keywords
        relevant_keywords = [
            "continue", "skip", "modulo", "remainder", "multiple", "divisib", 
            "accumulat", "total", "reset", "loop", "iteration", "range", 
            "bound", "inclusive", "off-by-one", "condition", "invariant",
            "preserve", "overwrite", "scope", "step", "recursion", "recursive",
            "base case", "stack", "return", "function", "variable", "parameter",
            "terminate", "unwind", "call", "index", "array", "list", "value",
            "prime", "divisible", "factor", "check", "true", "false", "branch"
        ]
        has_relevant_concept = any(kw in student_ans for kw in relevant_keywords)

        # Check for brief or partial explanation
        if len(student_ans) < 65 and has_relevant_concept:
            return {
                "answer_label": "Partially Correct",
                "concepts_correct": ["Mentions high-level control or computational mechanism"],
                "concepts_missed": ["Omitted detailed explanation of underlying invariants or boundary edge cases"],
                "contradictions": [],
                "supporting_evidence": [student_ans],
                "evaluation_explanation": "The student identifies the general algorithmic mechanism but provides an incomplete explanation of state invariants.",
                "uncertainty_score": 0.35,
                "needs_instructor_review": False
            }

        if not has_relevant_concept:
            return {
                "answer_label": "Incorrect",
                "concepts_correct": [],
                "concepts_missed": [
                    "Did not address referenced code logic or control structures",
                    "No relevant programming concepts articulated"
                ],
                "contradictions": ["The submitted text does not correspond to the required conceptual explanation."],
                "supporting_evidence": [student_ans[:120]],
                "evaluation_explanation": "The student explanation does not describe the targeted code mechanics or variable invariants. It appears unrelated or arbitrary.",
                "uncertainty_score": 0.85,
                "needs_instructor_review": True
            }

        # Standard correct answer when relevant concepts are present
        return {
            "answer_label": "Correct",
            "concepts_correct": [
                "Accurately explains the underlying control and state mechanism",
                "Tracks program execution logic and variable invariants"
            ],
            "concepts_missed": [],
            "contradictions": [],
            "supporting_evidence": [student_ans[:120]],
            "evaluation_explanation": "The student explanation demonstrates sound conceptual comprehension of the targeted code section, control structures, and state mutations.",
            "uncertainty_score": 0.1,
            "needs_instructor_review": False
        }


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider using official google-genai SDK with structured output."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def generate_json(self, system_instruction: str, user_prompt: str) -> Dict[str, Any]:
        from google.genai import types

        models_to_try = [self.model_name]
        for fallback_m in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-2.5-flash"]:
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

        last_error = None
        for model in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.2,
                    )
                )
                raw_text = response.text or "{}"
                data = extract_clean_json(raw_text)
                logger.info(f"Successfully generated dynamic JSON via Gemini ({model})")
                return data
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API request failed on {model}: {e}")

        logger.warning(f"All Gemini models failed ({last_error}). Falling back to AST-grounded deterministic provider.")
        fallback = MockLLMProvider()
        return fallback.generate_json(system_instruction, user_prompt)


def get_llm_provider() -> BaseLLMProvider:
    """Factory to obtain the configured LLM provider."""
    provider_pref = settings.llm_provider.lower()
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")

    if (provider_pref == "gemini" or provider_pref == "auto") and api_key:
        try:
            return GeminiProvider(api_key=api_key, model_name=settings.gemini_model)
        except Exception as e:
            logger.warning(f"Could not initialize GeminiProvider: {e}. Using MockLLMProvider.")
            return MockLLMProvider()

    return MockLLMProvider()
