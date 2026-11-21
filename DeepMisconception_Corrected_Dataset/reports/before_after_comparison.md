# Before vs After Comparison Report

| Dimension | Baseline Dataset | Corrected Dataset | Status |
|---|---|---|---|
| **Unique Process Sequences** | 6 sequences | **2011 sequences** | RESOLVED (6 behavioral workflows) |
| **Top Sequence Concentration** | 52.4% identical | **1.82%** | RESOLVED (No dominant template) |
| **Unique Viva Questions** | 39 questions | **5583 questions** | RESOLVED (10 cognitive archetypes) |
| **Unique Viva Answers** | 58 answers | **4688 answers** | RESOLVED (Code-grounded student answers) |
| **Code Correctness vs Label** | 100% diagonal leakage | **Decoupled & Overlapping** | RESOLVED (Pedagogical rubrics) |
| **`final_state_pass` Shortcut** | 1.0 for Understood, 0.0 others | **Overlapping distributions** | RESOLVED (No deterministic shortcut) |
| **Cross-Partition Overlap** | 0 cross-partition leakage | **0 cross-partition leakage** | PRESERVED (Split-first methodology) |
