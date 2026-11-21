# Synthetic Data Limitations & Training Provenance Disclosure

## CRITICAL SCIENTIFIC INTEGRITY DISCLOSURE
- **Synthetic Proxy Labels:** All target labels in this dataset (`synthetic_annotator_label` and `ground_truth_label`) are generated via rule-based pedagogical rubrics.
- **Human Verification Status:** `human_validated_label` remains strictly null. These labels DO NOT represent verified student understanding or genuine human ground truth.
- **Training Pipeline Notice:** Any downstream Transformer model (CodeBERT, Sentence Transformer, Process Transformer, Multimodal Fusion Transformer) trained on this dataset is explicitly trained using synthetic proxy annotations.
- **Augmentation Notice:** The test and validation partitions have NOT been subjected to SMOTE, GAN, or any data augmentation.
