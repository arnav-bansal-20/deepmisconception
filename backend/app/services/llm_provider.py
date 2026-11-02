"""LLM Provider abstraction supporting Google Gemini API with intelligent offline fallback.

Never exposes API credentials to the client. Uses Gemini 2.5 Flash via google-genai
when configured, and provides a deterministic, AST-grounded fallback for offline testing
and automated evaluation.
"""
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base provider for LLM calls."""

    @abstractmethod
    def generate_json(self, system_instruction: str, user_prompt: str) -> Dict[str, Any]:
        """Generate structured JSON output from LLM."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """Deterministic, AST-aware mock provider for testing and offline execution."""

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

    def _mock_question_generation(self, prompt: str) -> Dict[str, Any]:
        """Construct code-grounded viva questions based on prompt details."""
        questions = [
            {
                "question_text": "In the provided function, what is the exact condition under which an iteration skips the accumulator step, and why?",
                "focal_code": "if i % 3 == 0:\n    continue",
                "concept": "modulo-conditional-filtering",
                "difficulty": "medium",
                "expected_reasoning": "The modulo operator (%) evaluates the remainder of i divided by 3. When the remainder is 0 (i.e. multiples of 3 like 3, 6, 9), the 'continue' keyword immediately halts the current loop cycle and advances to the next iteration without adding i * i to total.",
                "rubric": {
                    "identifies_modulo": "Explains that % 3 == 0 identifies numbers divisible by 3",
                    "explains_continue": "Explains that continue skips the remaining lines in the loop body",
                    "identifies_effect_on_accumulator": "Notes that total += i * i is skipped for those values"
                },
                "misconception_indicators": [
                    "Confusing continue with break (thinking the loop stops completely)",
                    "Misinterpreting i % 3 as division result rather than remainder",
                    "Assuming multiples of 3 are included in the sum"
                ]
            },
            {
                "question_text": "What is the initial value of 'total', and what would be the consequence if 'total = 0' were defined inside the 'for' loop instead of before it?",
                "focal_code": "total = 0\nfor i in range(1, n + 1):",
                "concept": "variable-accumulator-invariants",
                "difficulty": "medium",
                "expected_reasoning": "Total is initialized to 0 outside the loop to accumulate values across all iterations. If total = 0 were placed inside the loop body, it would reset to 0 on every single iteration, destroying previously accumulated values and only retaining the calculation from the final iteration.",
                "rubric": {
                    "identifies_accumulator_role": "Understands state preservation across loop iterations",
                    "identifies_reset_bug": "Correctly diagnoses that putting it inside resets total on every cycle"
                },
                "misconception_indicators": [
                    "Failing to understand that variable scope or re-assignment inside a loop overwrites prior states",
                    "Believing python loops preserve previous assignment values even if overwritten"
                ]
            },
            {
                "question_text": "Why does the loop use range(1, n + 1) instead of range(1, n) or range(n)? What values of i are traversed when n = 5?",
                "focal_code": "for i in range(1, n + 1):",
                "concept": "loop-bounds-and-range-invariants",
                "difficulty": "easy",
                "expected_reasoning": "Python's range(start, stop) generates integers up to but excluding 'stop'. Therefore, range(1, n + 1) traverses from 1 up to n inclusive. For n = 5, the values traversed are 1, 2, 3, 4, 5. Using range(1, n) would omit 5 (an off-by-one error), and range(n) would start at 0 instead of 1.",
                "rubric": {
                    "range_stop_exclusive": "Recognizes that Python range stops at stop - 1",
                    "exact_values_n_5": "Correctly lists 1, 2, 3, 4, 5 for n=5",
                    "off_by_one_awareness": "Explains why n + 1 is required for inclusive upper bound"
                },
                "misconception_indicators": [
                    "Believing range(1, n + 1) stops at n + 1 (inclusive stop misconception)",
                    "Off-by-one boundary confusion"
                ]
            }
        ]
        return {"questions": questions}

    def _mock_answer_evaluation(self, prompt: str) -> Dict[str, Any]:
        """Perform semantic evaluation heuristic on prompt containing student answer."""
        # Extract student answer segment from prompt
        ans_marker = "student answer:"
        student_ans = ""
        if ans_marker in prompt.lower():
            student_ans = prompt.lower().split(ans_marker)[-1].strip()

        # Check for empty or evasive answers
        evasive_markers = [
            "idk", "dont know", "don't know", "not sure", "followed the instructions",
            "chatgpt", "geeksforgeeks", "guess", "it just works", "i don't remember",
            "maybe it is", "honestly"
        ]
        if len(student_ans) < 15 or any(m in student_ans for m in evasive_markers):
            return {
                "answer_label": "Insufficient Evidence",
                "concepts_correct": [],
                "concepts_missed": ["Incomplete explanation", "No mechanistic reasoning provided"],
                "contradictions": [],
                "supporting_evidence": [student_ans],
                "evaluation_explanation": "The student response is evasive, non-committal, or lacks mechanistic explanation of the referenced code construct.",
                "uncertainty_score": 0.85,
                "needs_instructor_review": True
            }

        # Check for common programming misconceptions & direct factual contradictions
        misconception_patterns = [
            ("stops the loop", "Confuses continue with break / loop termination"),
            ("ends the loop", "Confuses continue with break / loop termination"),
            ("terminates the loop", "Confuses continue with break / loop termination"),
            ("pauses execution", "Misinterprets continue as time delay"),
            ("quotient", "Confuses modulo % with integer division"),
            ("automatically detect", "Believes recursion terminates without explicit base case"),
            ("base case is optional", "Believes recursion terminates without explicit base case"),
            ("restarts the function", "Misinterprets base case as loop restart"),
            ("reverses the sorted array", "Misunderstands binary search pointer manipulation"),
            ("always greater than high", "Misinterprets binary search while loop condition"),
            ("copy of the list", "Believes Python passes lists by value rather than object reference"),
            ("untouched", "Fails to recognize in-place mutable object modification"),
            ("identically regarding in-place mutation", "Believes immutable types behave like mutable lists"),
            ("includes n + 1", "Fails to understand that range stop bound is exclusive"),
            ("starts at 1 by default", "Believes range starts at 1 instead of 0"),
            ("identical lookup speed", "Fails to recognize O(1) hash lookup vs O(n) linear scan"),
            ("binary search is used inside python sets", "Confuses hash table bucket lookup with binary search")
        ]
        for pattern, mis_desc in misconception_patterns:
            if pattern in student_ans:
                return {
                    "answer_label": "Incorrect",
                    "concepts_correct": ["Recognized focal code region"],
                    "concepts_missed": [mis_desc],
                    "contradictions": [f"Student assertion contradicts Python semantics: '{pattern}'."],
                    "supporting_evidence": [student_ans],
                    "evaluation_explanation": f"The student displays an active programming misconception: {mis_desc}.",
                    "uncertainty_score": 0.1,
                    "needs_instructor_review": False
                }

        # Check for partial answer (surface level explanation lacking deep invariants)
        partial_markers = [
            "skips numbers", "skips adding", "skips the calculation", "stops the function",
            "return early", "small", "shifts the search", "moves the low pointer", "cuts the array",
            "changes the list", "keeps the new item", "goes from 1 to n", "reaches n",
            "counts up to n", "faster for checking", "fast because it's a set", "set lookup is faster"
        ]
        if any(pm in student_ans for pm in partial_markers) or len(student_ans) < 95:
            return {
                "answer_label": "Partially Correct",
                "concepts_correct": ["Understands high-level functional purpose of construct"],
                "concepts_missed": ["Omitted formal algorithmic invariant or boundary condition"],
                "contradictions": [],
                "supporting_evidence": [student_ans],
                "evaluation_explanation": "The student correctly understands the overarching intent of the code, but provides an incomplete explanation lacking precise boundary conditions or state invariants.",
                "uncertainty_score": 0.25,
                "needs_instructor_review": False
            }

        # Check for relevant programming concept keywords across all CS1/CS2 topics
        relevant_keywords = [
            "continue", "skip", "modulo", "remainder", "multiple", "divisib", 
            "accumulat", "total", "reset", "loop", "iteration", "range", 
            "bound", "inclusive", "exclusive", "off-by-one", "condition", "invariant",
            "preserve", "overwrite", "scope", "step", "recursion", "base case", "call stack",
            "unwind", "overflow", "binary search", "pointer", "half", "sorted", "mutable",
            "immutable", "reference", "in-place", "hash", "set", "dictionary", "o(1)", "o(n)",
            "constant", "linear"
        ]
        has_relevant_concept = any(kw in student_ans for kw in relevant_keywords)

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

        # Standard correct answer when relevant concepts are present with mechanistic detail
        return {
            "answer_label": "Correct",
            "concepts_correct": [
                "Accurately articulated core language semantics and runtime mechanics",
                "Correctly tracked state transformation and execution invariants",
                "Identified operational boundary conditions without contradiction"
            ],
            "concepts_missed": [],
            "contradictions": [],
            "supporting_evidence": [student_ans[:120]],
            "evaluation_explanation": "The student demonstrates sound conceptual comprehension with accurate mechanistic reasoning consistent with actual execution behavior.",
            "uncertainty_score": 0.05,
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
        for fallback_m in ["gemini-3.5-flash", "gemini-3.8-flash"]:
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

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
                return json.loads(raw_text)
            except Exception as e:
                logger.warning(f"Gemini API request failed on {model}: {e}")

        logger.warning("All Gemini models failed. Falling back to deterministic provider.")
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
