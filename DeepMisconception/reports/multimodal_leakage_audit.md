# Multimodal Dataset & Partition Leakage Audit

## 1. Partition Verification

- **Scenario-Family Disjointness:** 100% disjoint across train (46), validation (10), and test (9). Zero cross-partition scenario leakage.
- **Session Isolation:** All session IDs strictly partitioned without duplication.
- **Preprocessing Isolation:** CodeBERT and SentenceTransformer encoders kept frozen without training-test leakage; 24 numerical process features normalized strictly using training set statistics.
- **Evidence Filtering:** Borderline / insufficient evidence sessions (N=543 total; 400 train, 77 val, 66 test) systematically excluded from supervised loss and evaluation.
