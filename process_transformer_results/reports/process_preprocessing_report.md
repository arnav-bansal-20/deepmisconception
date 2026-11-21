# Process Event Preprocessing and Tokenization Report

- **Total Sessions:** 5000
- **Total Events:** 64502
- **Max Sequence Length Configured:** 32
- **Observed Min Length:** 6
- **Observed Max Length:** 26
- **Observed Mean Length:** 12.90
- **Sequences Truncated by Max Length (32):** 0 (0.0%)

## Event Vocabulary

| Token ID | Event Type | Description |
| :---: | :--- | :--- |
| 0 | `<PAD>` | Zero padding token |
| 1 | `debug` | Chronological telemetry action |
| 2 | `delete` | Chronological telemetry action |
| 3 | `edit` | Chronological telemetry action |
| 4 | `idle` | Chronological telemetry action |
| 5 | `insert` | Chronological telemetry action |
| 6 | `paste` | Chronological telemetry action |
| 7 | `revision` | Chronological telemetry action |
| 8 | `run` | Chronological telemetry action |
| 9 | `runtime_error` | Chronological telemetry action |
| 10 | `save` | Chronological telemetry action |
| 11 | `syntax_error` | Chronological telemetry action |
| 12 | `test` | Chronological telemetry action |

## Temporal Feature Engineering

1. **Event Type Embedding:** Discrete embedding layer (dim=48).
2. **Time Delta Gap:** Continuous log1p-transformed inter-event gap in seconds.
3. **Elapsed Session Time:** Continuous normalized seconds from session onset.
4. **Character Attributes:** Inserted and deleted character deltas.
5. **Temporal Linear Projection:** Concatenated temporal & numerical attributes projected to dim=16, added to event embedding (total token dim=64).
6. **Attention Mask:** Boolean tensor indicating real events vs padding.
