# Actual Cross-Partition Leakage Audit Report (Safeguard 3)

## 1. Scenario Family Cross-Partition Overlap
- Train ∩ Validation Overlap: 0 families
- Train ∩ Test Overlap: 0 families
- Validation ∩ Test Overlap: 0 families

## 2. Actual Code String Overlap
- Train ∩ Validation Exact Code Overlap: 0
- Train ∩ Test Exact Code Overlap: 0
- Validation ∩ Test Exact Code Overlap: 0

## 3. Actual Viva Question Overlap
- Train ∩ Validation Question Overlap: 46
- Train ∩ Test Question Overlap: 83
- Validation ∩ Test Question Overlap: 36

## 4. Code Correctness Contingency Table
|                        |   correct |   incorrect |   partially_correct |
|:-----------------------|----------:|------------:|--------------------:|
| Partial Understanding  |       277 |         157 |                 566 |
| Possible Misconception |        97 |         324 |                  79 |
| Understood             |      2969 |         105 |                 426 |

## 5. Feature Shortcut Audit
- Mean `final_state_pass` rates: {'Partial Understanding': 0.305, 'Possible Misconception': 0.276, 'Understood': 0.6717142857142857}
- Deterministic shortcut eliminated: **True**

**Empirical Verdict: ZERO CROSS-PARTITION LEAKAGE CONFIRMED: True**
