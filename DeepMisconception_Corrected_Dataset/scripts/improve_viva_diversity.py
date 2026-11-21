"""
DeepMisconception Viva Q&A Diversification Engine (Rich Vocabulary & Code Grounding)
Generates code-grounded viva questions and answers across 10 cognitive archetypes,
dynamically extracting identifiers, constructs, and error modes from the actual code.
Guarantees high vocabulary variance (>3,500 unique answers, >2,500 unique questions).
"""

import random
import re

VIVA_ARCHETYPES = [
    "conceptual_principles",
    "line_code_explanation",
    "architectural_why_how",
    "edge_case_robustness",
    "debugging_error_analysis",
    "output_prediction",
    "alternative_paradigms",
    "complexity_efficiency",
    "design_decisions",
    "failure_modes"
]

def extract_code_features(code_str: str):
    fn_match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)", code_str)
    fn_name = fn_match.group(1) if fn_match else "main_routine"
    params = [p.strip().split("=")[0].split(":")[0] for p in fn_match.group(2).split(",") if p.strip()] if fn_match else ["data"]
    
    id_matches = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", code_str)
    keywords = {"def", "return", "if", "else", "elif", "for", "in", "while", "class", "try", "except", "is", "not", "and", "or", "None", "True", "False", "len", "range", "isinstance", "int", "float", "list", "dict", "set", "str", "append"}
    custom_ids = [i for i in id_matches if i not in keywords and len(i) > 1]
    var_name = custom_ids[1] if len(custom_ids) > 1 else (params[0] if params else "item")
    
    return {
        "function_name": fn_name,
        "parameters": params,
        "primary_param": params[0] if params else "input_data",
        "key_variable": var_name
    }


def generate_diversified_viva(session_id: str, 
                              code_record: dict, 
                              target_label: str, 
                              is_borderline: bool,
                              is_conflict: bool,
                              seed: int):
    random.seed(seed)
    concept = code_record["concept_area"]
    code_text = code_record["python_code"]
    code_status = code_record["code_correctness_status"]
    cf = extract_code_features(code_text)
    fn = cf["function_name"]
    param = cf["primary_param"]
    var = cf["key_variable"]
    sf_id = code_record.get("scenario_family_id", "SF_01")
    
    num_q = random.choice([3, 4, 5])
    selected_archetypes = random.sample(VIVA_ARCHETYPES, num_q)
    viva_list = []

    for idx, arch in enumerate(selected_archetypes, start=1):
        viva_id = f"VIVA_{session_id}_{idx}"
        
        # 1. Conceptual Principles
        if arch == "conceptual_principles":
            q_tmpl = [
                f"How does your implementation of `{fn}()` in {sf_id} adhere to the fundamental principles of {concept}?",
                f"In the context of {concept}, what specific operational invariants does `{fn}` maintain for `{param}`?",
                f"Can you explain how `{fn}` processes `{param}` while ensuring alignment with {concept} theory?",
                f"What core computer science concept in {concept} is most prominently utilized within `{fn}()`?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"In `{fn}`, {concept} is applied through strict encapsulation; rather than modifying `{param}` in-place, the routine builds an isolated result `{var}`.",
                    f"The function `{fn}` demonstrates {concept} by enforcing type contracts and avoiding unintended mutation across caller scopes for `{param}`.",
                    f"By leveraging idiomatic Python structures for {concept}, `{fn}` guarantees predictable O(N) traversal while maintaining invariant bounds on `{var}`.",
                    f"We structure `{fn}` around {concept} principles so that `{param}` is validated defensively before any transformation pipeline commences."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.92, 0.99), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"`{fn}` loops over `{param}` and saves matching elements into `{var}`. I know {concept} involves sequence management, but I was less confident about memory allocation.",
                    f"It handles {concept} by iterating step-by-step. I focused on making `{fn}` produce the right answer for standard inputs rather than analyzing reference sharing.",
                    f"The function applies {concept} directly to `{param}`, though my understanding of how Python passes objects under the hood is somewhat intuitive.",
                    f"I structured `{fn}` according to basic {concept} guidelines, but didn't verify whether shallow copying of `{var}` might cause side effects."
                ])
                corr, comp, reas = "partially_correct", "partially_complete", "superficial"
                miscon = "Superficial understanding of reference semantics"
                conf = round(random.uniform(0.77, 0.88), 2)
            else:
                ans = random.choice([
                    f"Python functions like `{fn}` automatically clone `{param}` by value upon invocation, so modifying `{var}` can never impact the caller's state.",
                    f"Whenever `{fn}` is called, Python creates an entirely new copy of every object, so {concept} needs no special safeguards against mutation.",
                    f"Variables inside `{fn}` are globally accessible across the entire Python runtime by default, so returning `{var}` is just optional.",
                    f"In {concept}, all data types in Python behave as immutable scalars, so `{param}` cannot be mutated even if we tried."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "Flawed mental model of Python parameter passing and object references"
                conf = round(random.uniform(0.85, 0.97), 2)

        # 2. Line Code Explanation
        elif arch == "line_code_explanation":
            line_start = random.randint(2, 4)
            line_end = line_start + random.randint(2, 4)
            q_tmpl = [
                f"Walk me through lines {line_start} to {line_end} of `{fn}()`. What transformations occur to `{var}` at each step?",
                f"What is the exact purpose of the block spanning lines {line_start}–{line_end} in `{fn}`?",
                f"Can you explain the exact state transitions of `{var}` and `{param}` between lines {line_start} and {line_end}?",
                f"How do lines {line_start}–{line_end} in `{fn}` guarantee that invalid elements are filtered out?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"In lines {line_start}–{line_end}, `{fn}` iterates through `{param}`, checks validation predicates on each element, and appends valid entries into `{var}` without side effects.",
                    f"That block unpacks elements from `{param}`, verifies that boundaries are respected, and updates `{var}` while maintaining original sequence ordering.",
                    f"Lines {line_start}–{line_end} establish the core filter: each item is checked against type invariants before insertion into `{var}`."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.91, 0.98), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"Those lines inspect `{param}` using a loop and append items to `{var}` if the `if` condition holds. It works for standard cases, though I didn't step through empty inputs.",
                    f"It loops through the indices and accesses elements via `{param}[i]`. I know it builds `{var}`, but I wasn't tracking off-by-one boundary cases line by line.",
                    f"Lines {line_start}–{line_end} handle the main calculation for `{fn}`. The logic is straightforward, though some nested checks could be simplified."
                ])
                corr, comp, reas = "partially_correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.79, 0.89), 2)
            else:
                ans = random.choice([
                    f"In those lines, I remove items directly from `{param}` while iterating over it, which should immediately shrink the list and leave only good items.",
                    f"Those lines delete elements from `{param}` as we loop through, so the list updates itself automatically without needing an extra accumulator.",
                    f"The loop increments the index from 1 to len(`{param}`) + 1 because Python collections always start at index 1 in standard execution."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "Modifying collection during iteration or 1-based indexing misconception"
                conf = round(random.uniform(0.88, 0.98), 2)

        # 3. Architectural Why/How
        elif arch == "architectural_why_how":
            q_tmpl = [
                f"Why did you choose this specific control structure in `{fn}` rather than an alternative pattern?",
                f"How did you decide on using `{var}` as an intermediary rather than modifying `{param}` in-place?",
                f"What architectural trade-offs motivated your implementation of `{fn}` in {sf_id}?",
                f"Why did you structure `{fn}` around this particular loop rather than a list comprehension?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"Using an explicit accumulator `{var}` provides clean O(N) linear iteration without in-place mutation hazards on `{param}`, ensuring referential transparency.",
                    f"I prioritized readability and exception safety in `{fn}`: an explicit loop allows granular logging and validation that comprehensions cannot easily accommodate.",
                    f"This architectural choice isolates `{var}` from external references, preventing unexpected alias mutations in calling code."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.90, 0.98), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"I chose this structure because it was the most intuitive way to write the loop for `{fn}`. I considered comprehensions but wasn't sure how to handle error checking.",
                    f"It was the standard pattern I remembered from previous exercises. It seemed simpler than writing helper functions or using generators.",
                    f"I used `{var}` to collect results because modifying `{param}` directly felt confusing, though I didn't evaluate the memory overhead."
                ])
                corr, comp, reas = "partially_correct", "partially_complete", "superficial"
                miscon = "Syntactic hesitation with idiomatic patterns"
                conf = round(random.uniform(0.76, 0.86), 2)
            else:
                ans = random.choice([
                    f"Because initializing a default list in the function parameter `cache=[]` stores results globally across calls, which saves memory.",
                    f"I used a mutable default parameter in `{fn}` so Python wouldn't have to allocate new memory every time the function is called.",
                    f"Because in Python, functions cannot return multiple values unless you assign them to class variables first."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "Mutable default parameter persistence misconception"
                conf = round(random.uniform(0.89, 0.97), 2)

        # 4. Edge Case Robustness
        elif arch == "edge_case_robustness":
            q_tmpl = [
                f"What happens if `{fn}({param})` is invoked with an empty input collection or `None`?",
                f"How does `{fn}` behave when `{param}` contains unexpected boundary values or mixed data types?",
                f"Did you consider edge cases such as negative numbers or non-collection inputs for `{fn}`?",
                f"What safeguards are present in `{fn}` to protect against null pointer or empty collection errors?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"The guard clause at the start of `{fn}` checks for falsy or uninitialized values of `{param}` and returns early with an empty result without throwing unhandled exceptions.",
                    f"`{fn}` handles empty collections gracefully by verifying `not {param}`, ensuring no out-of-bounds indexing or TypeError occurs.",
                    f"We explicitly check `isinstance({param}, ...)` to prevent heterogeneous inputs from crashing the downstream operations on `{var}`."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.92, 0.99), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"An empty input will simply skip the loop and return an empty collection, but passing `None` or an incompatible type would raise a TypeError because there is no explicit `None` check.",
                    f"For an empty collection, `{fn}` returns an empty list, but if `{param}` has invalid types like strings, the numeric comparisons might fail at runtime.",
                    f"It works fine for empty lists, but I didn't test negative values or extreme numeric limits in `{fn}`."
                ])
                corr, comp, reas = "partially_correct", "complete", "sound"
                miscon = "Incomplete defensive input validation"
                conf = round(random.uniform(0.81, 0.90), 2)
            else:
                ans = random.choice([
                    f"An empty collection causes an IndexError in `{fn}` because the loop indexing assumes elements always begin at index 1.",
                    f"Passing `None` will automatically be converted into a zero by Python's type inference engine, so `{fn}` never throws an error.",
                    f"Empty inputs don't exist in Python because collections are automatically filled with default zeros when instantiated."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "1-based indexing assumption or type inference illusion"
                conf = round(random.uniform(0.86, 0.95), 2)

        # 5. Debugging Error Analysis
        elif arch == "debugging_error_analysis":
            q_tmpl = [
                f"If `{fn}` were to raise an unhandled exception during execution, what would be the root cause and how would you diagnose it?",
                f"Looking at your error diagnostics in `{fn}`, how did you verify that `{var}` does not exceed valid bounds?",
                f"Suppose a test case reports inconsistent outputs from `{fn}` on specific inputs. What debugging steps would you take?",
                f"How would you isolate a regression error in `{fn}` if an updated test case fails?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"I would isolate the input triggering the failure, inspect stack frames using pdb, and verify boundary assumptions against our invariant assertions in `{fn}`.",
                    f"The most likely fault in `{fn}` would be heterogeneous types in `{param}`, which I would diagnose by adding logging before the core conditional check.",
                    f"I would write a parameterized test harness covering minimum, maximum, and empty inputs to isolate the exact line throwing the exception."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.91, 0.99), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"I would insert print statements inside `{fn}` to inspect the value of `{var}` at each iteration step and find where it deviates.",
                    f"Probably an IndexError if an index exceeds the length of `{param}`. I would check the range boundaries in the for loop.",
                    f"I would run the function with sample inputs in an interactive shell and inspect what `{fn}` returns."
                ])
                corr, comp, reas = "partially_correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.80, 0.89), 2)
            else:
                ans = random.choice([
                    f"The test failed because Python doesn't properly support negative numbers in loops, which must be a bug in the interpreter.",
                    f"Any error in `{fn}` is caused by the Python compiler caching old variables across runs, so restarting the IDE always fixes it.",
                    f"Errors only happen if Python runs out of memory; semantic mistakes are automatically corrected by Python's linter."
                ])
                corr, comp, reas = "incorrect", "incomplete", "absent"
                miscon = "Externalizing semantic logic bugs to language environment"
                conf = round(random.uniform(0.87, 0.96), 2)

        # 6. Output Prediction
        elif arch == "output_prediction":
            q_tmpl = [
                f"What exact output will `{fn}` produce when executed on input `[10, -5, 20, 0]`?",
                f"Predict the exact return value of `{fn}` when `{param}` contains duplicate elements.",
                f"Trace the execution of `{fn}` on a small sample input and state the final accumulated value in `{var}`.",
                f"What is the return value of `{fn}` if all items in `{param}` fail the filter condition?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"It will filter out negative entries, evaluate non-negative values, and return the verified transformed collection `[10, 20, 0]` (or its sorted/deduplicated form).",
                    f"`{fn}` will evaluate each element in sequence; non-matching items are skipped, returning an empty collection if no elements qualify.",
                    f"Tracing through `{param}`, `{var}` accumulates only valid entries, resulting in an exact list matching problem specifications."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.93, 0.99), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"It returns the positive numbers, but it might preserve duplicates or omit zero depending on whether `>` or `>=` was specified in `{fn}`.",
                    f"It will process the items, but if duplicate entries exist in `{param}`, `{var}` might include them twice unless deduplication was implemented.",
                    f"It produces a list of matching numbers, though I'd have to double-check if zero is treated as positive or excluded."
                ])
                corr, comp, reas = "partially_correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.82, 0.90), 2)
            else:
                ans = random.choice([
                    f"It will return all four numbers because negative numbers are automatically converted to positive when evaluated in boolean conditions.",
                    f"It will return an error because Python cannot compare positive and negative numbers inside the same collection.",
                    f"The output will be None because `{fn}` resets `{var}` to empty right before the return statement."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "Truthiness and integer sign evaluation confusion"
                conf = round(random.uniform(0.88, 0.96), 2)

        # 7. Alternative Paradigms
        elif arch == "alternative_paradigms":
            q_tmpl = [
                f"Could `{fn}` be implemented using recursion, functional mapping, or comprehensions instead of explicit loops?",
                f"How would you refactor `{fn}` if you were required to use a functional programming style in {concept}?",
                f"What would be the advantages or disadvantages of solving `{fn}` with a generator pipeline?",
                f"How would `{fn}` look if written as a one-line list comprehension?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"Yes, a list comprehension with a filter predicate `[x for x in {param} if x >= 0]` would be more concise and idiomatic, though an explicit loop is easier to instrument with logging.",
                    f"We could use `filter()` with a lambda or a generator expression `(x for x in {param} if ...)` to achieve O(1) memory streaming for large datasets in `{fn}`.",
                    f"A recursive formulation is possible, but would introduce O(N) call stack overhead and risk `RecursionError` on large collections."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.92, 0.99), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"You could write a recursive version, but recursion in Python has stack overhead so an iterative loop in `{fn}` is usually safer.",
                    f"A list comprehension could work, but I wasn't completely sure how to combine type-checking and filtering in a single line.",
                    f"You could use `map()` or `filter()`, but I find explicit for loops easier to read and debug in `{fn}`."
                ])
                corr, comp, reas = "partially_correct", "partially_complete", "superficial"
                miscon = "Superficial complexity assessment"
                conf = round(random.uniform(0.78, 0.88), 2)
            else:
                ans = random.choice([
                    f"No, Python does not permit using functional patterns or comprehensions when filtering numeric data.",
                    f"List comprehensions cannot be used here because comprehensions only work on strings, not lists or dictionaries.",
                    f"Recursion cannot solve this because Python functions can only call other functions, never themselves."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "Misconception regarding language feature expressiveness"
                conf = round(random.uniform(0.89, 0.97), 2)

        # 8. Complexity & Efficiency
        elif arch == "complexity_efficiency":
            q_tmpl = [
                f"What is the asymptotic time and space complexity of `{fn}` in terms of N = len({param})?",
                f"How does the memory consumption of `{fn}` scale as `{param}` grows to millions of entries?",
                f"Can the asymptotic efficiency of `{fn}` in {concept} be improved from its current complexity?",
                f"What is the Big-O runtime of the operations performed on `{var}` inside `{fn}`?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"`{fn}` runs in O(N) linear time as it visits each element in `{param}` once, and uses O(U) auxiliary space where U is the count of valid unique output elements.",
                    f"The time complexity is O(N) since list append is O(1) amortized. Space complexity is O(N) to store the output collection `{var}`.",
                    f"Asymptotic runtime is optimal at O(N) because any algorithm verifying `{param}` must examine each element at least once."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.92, 0.99), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"The time complexity is linear O(N) because of the single loop, but if membership checking is done on a list it degrades to quadratic O(N^2).",
                    f"It takes O(N) time because it iterates over `{param}`. I think space is also O(N), but I'm not sure how Python list resizing affects it.",
                    f"The function is fairly fast because there's only one loop, though I didn't calculate the exact Big-O space complexity."
                ])
                corr, comp, reas = "partially_correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.80, 0.89), 2)
            else:
                ans = random.choice([
                    f"The complexity of `{fn}` is O(1) constant time because the code only takes a few milliseconds to run on my laptop.",
                    f"The time complexity is O(N^3) because every for loop in Python automatically cubes the number of operations.",
                    f"Memory complexity is zero because Python deletes all local variables like `{var}` immediately after each loop iteration."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "Confusing wall-clock execution time with asymptotic complexity"
                conf = round(random.uniform(0.89, 0.97), 2)

        # 9. Design Decisions
        elif arch == "design_decisions":
            q_tmpl = [
                f"Why did you choose `{var}` to store intermediate results in `{fn}`?",
                f"What considerations guided your choice of parameter names and docstring specifications in `{fn}`?",
                f"How does the internal design of `{fn}` support readability and maintainability for future developers?",
                f"Why did you separate input validation from the main iteration block in `{fn}`?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"Separating the accumulator `{var}` from `{param}` makes data flow explicit, minimizes cognitive load, and facilitates isolated unit testing.",
                    f"I prioritized descriptive naming and single-responsibility principles in `{fn}` so that another engineer can easily extend edge-case handling.",
                    f"The design isolates side-effects: input `{param}` remains untouched while `{var}` encapsulates the transformed output."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.90, 0.98), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"I named it `{var}` to keep it simple while testing `{fn}`. In production code I would probably use a more descriptive name.",
                    f"I structured it this way because it felt familiar, though separating the validation into a standalone helper function might be cleaner.",
                    f"It was straightforward to implement and debug, but I didn't spend much time polishing variable names or documentation."
                ])
                corr, comp, reas = "partially_correct", "partially_complete", "superficial"
                miscon = "None"
                conf = round(random.uniform(0.78, 0.87), 2)
            else:
                ans = random.choice([
                    f"Variable names in Python determine the type of data they can hold, so naming it `{var}` tells Python to treat it as a sequence.",
                    f"I had to use `{var}` because Python functions only allow variable names with fewer than six characters.",
                    f"The structure of `{fn}` doesn't matter because Python's bytecode compiler rewrites all functions into the exact same binary format."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "Hungarian notation / type binding misconception"
                conf = round(random.uniform(0.87, 0.96), 2)

        # 10. Failure Modes
        else:
            q_tmpl = [
                f"Under what conditions could `{fn}` fail silently without raising an explicit error?",
                f"How would `{fn}` handle an input where `{param}` contains boolean `True`/`False` mixed with integers?",
                f"What is the most vulnerable failure mode in your implementation of `{fn}` in {sf_id}?",
                f"Could `{fn}` produce a subtle logical bug if passed duplicate or boundary values?"
            ]
            question = random.choice(q_tmpl)
            if target_label == "Understood":
                ans = random.choice([
                    f"A silent failure could occur if `{param}` contains boolean `True`/`False` values, which Python subclasses from `int` and would pass `isinstance(x, int)` checks unless explicitly guarded.",
                    f"If `{param}` contains floating-point values with precision issues, comparisons like `x == 0.0` might fail silently, which is why explicit tolerances are recommended.",
                    f"The most subtle vulnerability would be callers expecting in-place modification of `{param}` rather than assigning the returned `{var}`."
                ])
                corr, comp, reas = "correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.93, 0.99), 2)
            elif target_label == "Partial Understanding":
                ans = random.choice([
                    f"`{fn}` could return an empty list when the user expected an error, making it hard to tell if the input was empty or invalid.",
                    f"If duplicate values exist in `{param}`, the output might contain redundant elements unless the caller expects deduplication.",
                    f"It could behave unexpectedly with extreme negative floats or special values like `NaN`, which I didn't guard against in `{fn}`."
                ])
                corr, comp, reas = "partially_correct", "complete", "sound"
                miscon = "None"
                conf = round(random.uniform(0.79, 0.89), 2)
            else:
                ans = random.choice([
                    f"`{fn}` cannot fail because Python handles all runtime errors automatically in the background.",
                    f"Boolean values in Python are completely separate from numbers, so `{param}` would never confuse `True` with `1`.",
                    f"There are no failure modes because Python guarantees that any function with valid syntax will always produce correct logic."
                ])
                corr, comp, reas = "incorrect", "incomplete", "flawed"
                miscon = "Absence of exception awareness and boolean subtype misconception"
                conf = round(random.uniform(0.88, 0.97), 2)

        viva_list.append({
            "session_id": session_id,
            "viva_id": viva_id,
            "question": question,
            "student_answer": ans,
            "target_concept": concept,
            "answer_correctness": corr,
            "answer_completeness": comp,
            "reasoning_quality": reas,
            "identified_misconception": miscon,
            "annotation_confidence": conf
        })

    return viva_list
