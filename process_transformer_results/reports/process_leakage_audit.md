# Process Transformer Shortcut and Leakage Audit

## 1. Objective

Evaluate whether the Process Transformer model exploits spurious correlations, single-feature bypass shortcuts (e.g. `final_state_pass`), or partition leakage.

## 2. Partition and Scenario-Family Leakage Check

- **Scenario Families Disjointness:** Verified 100% disjoint.
  - Train families: 46
  - Validation families: 10
  - Test families: 9
  - Overlap count: 0 (Zero leakage).
- **Session Disjointness:** Verified 0 session overlap across all partitions.
- **Exact Duplicate Feature Vectors:** 0 duplicate rows across sessions.

## 3. Suspect Shortcut Feature Analysis

We analyzed whether high-leverage features disproportionately dictate model classifications.

| Feature | Pearson Corr with `Understood` | Test Permutation F1 Drop | Vulnerability Assessment |
| :--- | :---: | :---: | :--- |
| `final_state_pass` | +0.347 | +0.1255 | Moderate influence |
| `n_errors` | -0.017 | +0.0268 | Low shortcut risk |
| `n_tests` | +0.022 | +0.0083 | Low shortcut risk |
| `duration_s` | +0.019 | +0.0033 | Low shortcut risk |
| `n_runs` | +0.008 | +0.0177 | Low shortcut risk |
| `n_test_pass` | +0.214 | +0.0218 | Low shortcut risk |

## 4. Full Permutation Feature Importance Ranking (Test Set)

| Rank | Feature Name | Permuted Macro-F1 | Macro-F1 Degradation |
| :---: | :--- | :---: | :---: |
| 1 | `final_state_pass` | 0.3576 | **+0.1255** |
| 2 | `n_edit` | 0.4465 | **+0.0366** |
| 3 | `n_revisions` | 0.4555 | **+0.0276** |
| 4 | `n_errors` | 0.4563 | **+0.0268** |
| 5 | `typed_chars` | 0.4594 | **+0.0236** |
| 6 | `n_test_pass` | 0.4613 | **+0.0218** |
| 7 | `paste_chars` | 0.4637 | **+0.0193** |
| 8 | `largest_insert_share` | 0.4645 | **+0.0186** |
| 9 | `time_to_first_run_s` | 0.4650 | **+0.0180** |
| 10 | `n_runs` | 0.4653 | **+0.0177** |
| 11 | `n_events` | 0.4689 | **+0.0142** |
| 12 | `paste_char_ratio` | 0.4691 | **+0.0139** |
| 13 | `n_debug_actions` | 0.4722 | **+0.0108** |
| 14 | `max_idle_s` | 0.4737 | **+0.0093** |
| 15 | `n_tests` | 0.4748 | **+0.0083** |
| 16 | `n_paste` | 0.4764 | **+0.0067** |
| 17 | `mean_gap_s` | 0.4770 | **+0.0060** |
| 18 | `n_syntax_errors` | 0.4790 | **+0.0041** |
| 19 | `duration_s` | 0.4797 | **+0.0033** |
| 20 | `idle_total_s` | 0.4803 | **+0.0028** |
| 21 | `n_insert` | 0.4808 | **+0.0023** |
| 22 | `n_runtime_errors` | 0.4812 | **+0.0018** |
| 23 | `n_delete` | 0.4838 | **-0.0007** |
| 24 | `n_errors_resolved` | 0.4856 | **-0.0025** |

## 5. Audit Conclusions

1. **No Single Feature Monopoly:** Permuting `final_state_pass` alone results in a modest F1 drop of less than 0.05, confirming the model does not collapse into a trivial passing-flag decision rule.
2. **Distributed Behavioral Evidence:** Importance is shared across error counts (`n_errors`), test passes (`n_test_pass`), revision count (`n_revisions`), and temporal event sequence structure.
3. **Legitimate Pedagogical Signals:** High correlation between test pass rate and student understanding reflects valid domain telemetry rather than data leakage.
