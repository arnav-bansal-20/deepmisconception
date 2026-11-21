# Process Transformer Dataset and Session Alignment Audit

- **Total Multimodal Sessions:** 5000
- **Dataset Source:** `DeepMisconception_Corrected_Dataset/`
- **Approved Partitions Used:** `train/` (N=3537), `validation/` (N=770), `test/` (N=693)
- **Disjointness:** Zero session overlap, Zero scenario-family overlap across partitions.

## Partition Breakdown

| Partition | Total Sessions | Insufficient Evidence (Excluded) | Usable Supervised Sessions | Understood | Partial Understanding | Possible Misconception |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 3537 | 400 | 3137 | 2251 | 587 | 299 |
| **Validation** | 770 | 77 | 693 | 484 | 143 | 66 |
| **Test** | 693 | 66 | 627 | 417 | 140 | 70 |

## Numerical Process Features (Exact 24 Columns)

Features: `n_events, duration_s, n_insert, n_edit, n_delete, n_paste, typed_chars, paste_chars, paste_char_ratio, largest_insert_share, n_runs, n_syntax_errors, n_runtime_errors, n_errors, n_debug_actions, n_tests, n_test_pass, n_revisions, n_errors_resolved, idle_total_s, max_idle_s, mean_gap_s, time_to_first_run_s, final_state_pass`

## Event Vocabulary

Observed event types (12): `debug, delete, edit, idle, insert, paste, revision, run, runtime_error, save, syntax_error, test`

- Chronological timestamps: Strictly monotonically increasing.
- Event sequence length per session: Min=6, Max=26, Mean=12.90

## Target Label Provenance

- `human_validated_label`: Currently null across all sessions.
- **Active Training Target:** `ground_truth_label` / `synthetic_annotator_label` (explicitly documented as synthetic pedagogical proxy labels).
- `evidence_status`: Used to filter out insufficient-evidence sessions from 3-class supervised classification.
