# Synthetic Data and Label Limitations

## 1. Methodological Scope

All experiments in this study were conducted on the synthetic `DeepMisconception_Corrected_Dataset`. While synthetic generation enables complete multimodal behavioral alignment across code, viva explanations, and process telemetry, it carries distinct limitations:

1. **Absence of Real Human Adjudication:** The `human_validated_label` field remains unpopulated. Evaluated labels reflect rule-based synthetic pedagogical annotations.
2. **Generalization Boundary:** Findings demonstrate that multimodal fusion improves synthetic label prediction over unimodal baselines, but do not establish real-world classroom generalization.
3. **Process Regularity:** Synthetic process events exhibit less erratic hesitation, external browser lookups, and IDE switching than live student cohorts.
