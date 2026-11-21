# DeepMisconception Multimodal Dataset (Enhanced & Safeguarded)

A multimodal dataset for predicting programming misconceptions, incorporating four scientific integrity safeguards:

## Key Safeguards Implemented
1. **Label Provenance & Separation:** `original_label` preserved; `synthetic_annotator_label` stores proxy labels; `human_validated_label` remains null until expert review.
2. **Evidence-Based Annotation:** Ground truth assigned via documented pedagogical rubrics without forced percentages or deterministic formulas.
3. **Actual Leakage Verification:** Zero cross-partition leakage verified empirically across code, viva, process, and scenario families.
4. **Split-First Generation:** Scenario families partitioned into Train, Validation, and Test before dependent synthetic generation.

## Modalities
- **Code:** Python submissions across 65 scenario families and 36 concept areas.
- **Viva:** 10 cognitive archetypes dynamically grounded in code.
- **Process:** 6 authentic behavioral trajectories and the exact 24 numerical features.
