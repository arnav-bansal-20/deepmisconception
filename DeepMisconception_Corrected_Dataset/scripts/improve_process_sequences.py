"""
DeepMisconception Process Event Sequence Generator (Rich Stochastic Trajectories)
Generates authentic, diverse chronological coding-process event sequences modeling
6 distinct behavioral trajectories with stochastic sub-action sequences, variable
chunking, interleaved edits, dynamic error-debug cycles, and varied testing runs.
"""

import random
from datetime import datetime, timedelta

TRAJECTORY_TYPES = [
    "planned",
    "exploratory",
    "debugging_oriented",
    "struggling",
    "mixed",
    "test_driven"
]

ERROR_TYPES = [
    "SyntaxError",
    "IndentationError",
    "IndexError",
    "TypeError",
    "KeyError",
    "NameError",
    "ValueError"
]

DEBUG_ACTIONS = [
    "inspected_error_traceback",
    "added_print_diagnostics",
    "corrected_boundary_condition",
    "fixed_indentation_level",
    "refactored_loop_condition",
    "added_type_assertion",
    "removed_diagnostic_prints"
]

def generate_diversified_process_events(session_id: str, 
                                        code_record: dict, 
                                        behavioral_archetype: str, 
                                        final_pass: int, 
                                        seed: int):
    random.seed(seed)
    
    start_time = datetime(2026, 11, 20, 8, 30, 0) + timedelta(minutes=random.randint(0, 1440))
    current_time = start_time
    
    events = []
    event_seq = 1
    
    def emit(ev_type, desc, c_ins=0, c_del=0, exec_res="none", err="none", dbg="none", test_res="none", gap_s=5):
        nonlocal current_time, event_seq
        current_time += timedelta(seconds=max(1, int(gap_s)))
        snap_ref = f"SNAP_{session_id}_{event_seq:03d}"
        ev = {
            "session_id": session_id,
            "event_id": f"EVT_{session_id}_{event_seq:03d}",
            "timestamp": current_time.isoformat() + "Z",
            "event_type": ev_type,
            "event_description": desc,
            "characters_inserted": int(c_ins),
            "characters_deleted": int(c_del),
            "code_snapshot_reference": snap_ref,
            "execution_result": exec_res,
            "error_type": err,
            "debug_action": dbg,
            "test_result": test_res
        }
        events.append(ev)
        event_seq += 1

    # Trajectory 1: Planned Coding
    if behavioral_archetype == "planned":
        emit("idle", "Problem analysis and solution design", gap_s=random.randint(35, 95))
        # Stochastic chunked insertion
        n_chunks = random.choice([2, 3, 4])
        for c_i in range(n_chunks):
            emit("insert", f"Authored module component {c_i+1}/{n_chunks}", c_ins=random.randint(25, 60), gap_s=random.randint(10, 25))
            if random.random() < 0.25:
                emit("edit", "Minor parameter adjustment", c_ins=random.randint(5, 15), c_del=random.randint(2, 6), gap_s=random.randint(8, 18))
        if random.random() < 0.35:
            emit("paste", "Pasted test fixtures or helper array", c_ins=random.randint(20, 45), gap_s=random.randint(5, 12))
        emit("run", "Validation run of assembled code", exec_res="success" if final_pass else "runtime_error", gap_s=random.randint(12, 30))
        if random.random() < 0.45:
            emit("edit", "Code refactoring and cleanup", c_ins=random.randint(10, 25), c_del=random.randint(3, 8), gap_s=random.randint(10, 22))
        if random.random() < 0.50:
            emit("save", "Checkpoint save", gap_s=random.randint(3, 8))
        # Testing loop
        n_tests = random.randint(2, 4)
        for t_idx in range(n_tests):
            t_pass = "pass" if (final_pass or t_idx < n_tests - 1) else "fail"
            emit("test", f"Ran acceptance test #{t_idx+1}", exec_res="test_pass" if t_pass=="pass" else "test_fail", test_res=t_pass, gap_s=random.randint(8, 20))
            if random.random() < 0.20:
                emit("edit", "Quick test assertion adjustment", c_ins=random.randint(5, 12), gap_s=random.randint(6, 14))
        emit("save", "Final submission save", gap_s=random.randint(4, 10))

    # Trajectory 2: Exploratory / Incremental Coding
    elif behavioral_archetype == "exploratory":
        emit("insert", "Initial stub and basic signature", c_ins=random.randint(20, 45), gap_s=random.randint(8, 18))
        emit("run", "Check basic interpreter environment", exec_res="success", gap_s=random.randint(8, 18))
        n_increments = random.choice([2, 3])
        for inc in range(n_increments):
            emit("insert", f"Incremental feature implementation step {inc+1}", c_ins=random.randint(25, 55), gap_s=random.randint(12, 28))
            if random.random() < 0.45:
                emit("edit", "Refined condition logic in loop", c_ins=random.randint(10, 25), c_del=random.randint(5, 12), gap_s=random.randint(10, 22))
            emit("run", f"Intermediate execution check {inc+1}", exec_res="success", gap_s=random.randint(10, 22))
        if random.random() < 0.40:
            emit("paste", "Pasted auxiliary utility function", c_ins=random.randint(25, 50), gap_s=random.randint(5, 12))
        if random.random() < 0.50:
            emit("revision", "Restructured iteration structure", c_ins=random.randint(15, 30), c_del=random.randint(10, 20), gap_s=random.randint(18, 40))
        if random.random() < 0.35:
            emit("delete", "Removed unused exploratory variables", c_del=random.randint(12, 25), gap_s=random.randint(8, 16))
        n_tests = random.randint(1, 3)
        for t in range(n_tests):
            t_res = "pass" if final_pass else "fail"
            emit("test", f"Exploratory unit verification #{t+1}", exec_res="test_pass" if t_res=="pass" else "test_fail", test_res=t_res, gap_s=random.randint(10, 22))
        emit("save", "Saved exploratory codebase", gap_s=random.randint(4, 10))

    # Trajectory 3: Debugging-Oriented Coding
    elif behavioral_archetype == "debugging_oriented":
        emit("insert", "Drafted complete initial implementation", c_ins=random.randint(50, 95), gap_s=random.randint(15, 35))
        n_debug_cycles = random.choice([1, 2, 3])
        for cyc in range(n_debug_cycles):
            err_type = random.choice(ERROR_TYPES[:4])
            emit("run", f"Execution attempt {cyc+1} triggered exception", exec_res=err_type.lower(), err=err_type, gap_s=random.randint(10, 25))
            ev_err_type = "syntax_error" if "syntax" in err_type.lower() or "indent" in err_type.lower() else "runtime_error"
            emit(ev_err_type, f"Encountered {err_type}", err=err_type, gap_s=random.randint(5, 12))
            emit("debug", random.choice(DEBUG_ACTIONS), dbg="inspected_traceback", c_ins=random.randint(5, 15), c_del=random.randint(2, 6), gap_s=random.randint(15, 35))
            emit("edit", f"Patched fault condition {cyc+1}", c_ins=random.randint(12, 30), c_del=random.randint(6, 18), gap_s=random.randint(12, 28))
            if random.random() < 0.40:
                emit("revision", "Refactored faulty branch", c_ins=random.randint(10, 25), c_del=random.randint(5, 12), gap_s=random.randint(15, 30))
        emit("run", "Post-debugging confirmation run", exec_res="success" if final_pass else "runtime_error", gap_s=random.randint(10, 22))
        if random.random() < 0.50:
            emit("delete", "Cleaned up temporary debug print statements", c_del=random.randint(15, 35), gap_s=random.randint(8, 16))
        n_tests = random.randint(1, 3)
        for t in range(n_tests):
            t_res = "pass" if final_pass else "fail"
            emit("test", f"Regression test #{t+1}", exec_res="test_pass" if t_res=="pass" else "test_fail", test_res=t_res, gap_s=random.randint(10, 22))
        emit("save", "Saved post-debug clean state", gap_s=random.randint(4, 10))

    # Trajectory 4: Struggling / Thrashing Coding
    elif behavioral_archetype == "struggling":
        emit("insert", "Initial partial logic attempt", c_ins=random.randint(35, 70), gap_s=random.randint(12, 30))
        n_fails = random.choice([2, 3, 4])
        for f_idx in range(n_fails):
            err = random.choice(["IndexError", "TypeError", "KeyError", "ValueError"])
            emit("run", f"Failed execution attempt {f_idx+1}", exec_res="runtime_error", err=err, gap_s=random.randint(10, 22))
            emit("runtime_error", f"Unresolved {err} in execution", err=err, gap_s=random.randint(4, 10))
            if random.random() < 0.65:
                emit("debug", "Added ad-hoc print logging", dbg="added_print", c_ins=random.randint(10, 20), gap_s=random.randint(15, 35))
            if random.random() < 0.55:
                emit("delete", "Deleted confused loop portion", c_del=random.randint(15, 35), gap_s=random.randint(10, 25))
            emit("edit", f"Trial-and-error edit {f_idx+1}", c_ins=random.randint(10, 25), c_del=random.randint(5, 15), gap_s=random.randint(15, 35))
            if random.random() < 0.40:
                emit("idle", "Extended hesitation trying to understand error", gap_s=random.randint(60, 150))
        if random.random() < 0.60:
            emit("revision", "Heuristic rewrite of loop bounds", c_ins=random.randint(12, 28), c_del=random.randint(8, 20), gap_s=random.randint(25, 55))
        emit("test", "Final test attempt", exec_res="test_fail", test_res="fail", gap_s=random.randint(12, 28))
        emit("save", "Saved code with unresolved issues", gap_s=random.randint(4, 12))

    # Trajectory 5: Mixed Coding Behavior
    elif behavioral_archetype == "mixed":
        if random.random() < 0.70:
            emit("idle", "Pre-coding planning pause", gap_s=random.randint(30, 75))
        emit("insert", "Initial prototype code block", c_ins=random.randint(45, 85), gap_s=random.randint(15, 35))
        if random.random() < 0.35:
            emit("paste", "Pasted snippet from reference", c_ins=random.randint(20, 45), gap_s=random.randint(5, 12))
        emit("run", "First trial run", exec_res="success" if random.random() < 0.5 else "runtime_error", gap_s=random.randint(10, 22))
        # Iterative refactoring
        n_edits = random.choice([1, 2, 3])
        for e_i in range(n_edits):
            emit("edit", f"Refactoring iteration {e_i+1}", c_ins=random.randint(15, 35), c_del=random.randint(8, 22), gap_s=random.randint(15, 35))
            if random.random() < 0.40:
                emit("run", "Intermediate check", exec_res="success", gap_s=random.randint(8, 18))
        emit("revision", "Substantial algorithmic restructuring", c_ins=random.randint(18, 38), c_del=random.randint(10, 25), gap_s=random.randint(22, 50))
        if random.random() < 0.40:
            emit("delete", "Removed deprecated helper logic", c_del=random.randint(10, 25), gap_s=random.randint(8, 16))
        n_tests = random.randint(1, 3)
        for t in range(n_tests):
            t_res = "pass" if (final_pass or t < n_tests - 1) else "fail"
            emit("test", f"Mixed suite verification #{t+1}", exec_res="test_pass" if t_res=="pass" else "test_fail", test_res=t_res, gap_s=random.randint(10, 25))
        emit("save", "Saved final mixed submission", gap_s=random.randint(4, 10))

    # Trajectory 6: Alternative / Test-Driven Workflows
    else:  # test_driven
        emit("insert", "Authored unit test assertion suite first", c_ins=random.randint(35, 65), gap_s=random.randint(12, 28))
        emit("test", "Ran test suite before implementation (baseline failure)", exec_res="test_fail", test_res="fail", gap_s=random.randint(8, 18))
        n_steps = random.choice([2, 3])
        for s in range(n_steps):
            emit("insert", f"Implemented logic to satisfy assertion step {s+1}", c_ins=random.randint(30, 60), gap_s=random.randint(15, 35))
            emit("run", f"Incremental execution check {s+1}", exec_res="success", gap_s=random.randint(8, 18))
            emit("test", f"Ran unit test assertions step {s+1}", exec_res="test_pass" if (final_pass or s < n_steps - 1) else "test_fail", test_res="pass" if (final_pass or s < n_steps - 1) else "fail", gap_s=random.randint(8, 18))
            if random.random() < 0.40:
                emit("edit", "Refactored implementation for readability", c_ins=random.randint(10, 22), c_del=random.randint(5, 12), gap_s=random.randint(10, 22))
        if random.random() < 0.35:
            emit("revision", "Final parameter refinement", c_ins=random.randint(8, 18), c_del=random.randint(4, 10), gap_s=random.randint(12, 25))
        emit("save", "Saved test-driven submission", gap_s=random.randint(4, 10))

    return events
