# Feature Validation Report (Exact 24 Numerical Features)

Identity and Mathematical Constraints:
- `n_errors == n_syntax_errors + n_runtime_errors`: Verified 5,000 / 5,000 sessions.
- `n_test_pass <= n_tests`: Verified 5,000 / 5,000 sessions.
- `n_errors_resolved <= n_errors`: Verified 5,000 / 5,000 sessions.
- `paste_char_ratio in [0.0, 1.0]`: Verified 5,000 / 5,000 sessions.
- `largest_insert_share in [0.0, 1.0]`: Verified 5,000 / 5,000 sessions.
- `final_state_pass in {0, 1}`: Verified 5,000 / 5,000 sessions.
