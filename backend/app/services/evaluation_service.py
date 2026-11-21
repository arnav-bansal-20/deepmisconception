"""Evaluation service for DeepMisconception.

Loads empirical test evaluation metrics, class-wise diagnostic scores,
confusion matrix, and group-aware 5-fold cross-validation results.
Generates multi-sheet Excel and CSV exportable reports.
"""
import os
import io
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_METRICS_PATH = PROJECT_ROOT / "backend/app/ml/checkpoints/test_evaluation_metrics.json"
CV_RESULTS_PATH = PROJECT_ROOT / "backend/app/ml/checkpoints/five_fold_cv_results.json"
CHECKPOINT_PATH = PROJECT_ROOT / "DeepMisconception/models/complete_multimodal_model/complete_multimodal_model_best.pt"


class EvaluationService:
    """Provides structured model evaluation metrics, tables, and report exports."""

    @classmethod
    def get_model_status(cls) -> str:
        """Verify whether a valid trained checkpoint is present."""
        if CHECKPOINT_PATH.exists() and TEST_METRICS_PATH.exists():
            return "READY"
        return "MODEL_NOT_TRAINED"

    @classmethod
    def get_evaluation_metrics(cls) -> Dict[str, Any]:
        """Return the comprehensive suite of 9 evaluation metrics and 5-fold CV data."""
        status = cls.get_model_status()
        if status == "MODEL_NOT_TRAINED":
            return {
                "status": "MODEL_NOT_TRAINED",
                "message": "Valid trained Multimodal Fusion Transformer checkpoint is not available.",
                "model_name": "Multimodal Fusion Transformer",
                "classes": ["Understood", "Partial Understanding", "Possible Misconception"]
            }

        with open(TEST_METRICS_PATH, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        with open(CV_RESULTS_PATH, "r", encoding="utf-8") as f:
            cv_data = json.load(f)

        # Baseline comparisons from empirical runs
        baselines = [
            {"configuration": "1. Code-Only (AST + CodeBERT)", "accuracy": 0.8676, "precision_macro": 0.8016, "recall_macro": 0.9083, "f1_macro": 0.8392, "roc_auc": 0.9906},
            {"configuration": "2. Viva-Only (Sentence-Transformer)", "accuracy": 0.7799, "precision_macro": 0.6963, "recall_macro": 0.8098, "f1_macro": 0.7182, "roc_auc": 0.9405},
            {"configuration": "3. Process-Only (Process Transformer)", "accuracy": 0.6651, "precision_macro": 0.2217, "recall_macro": 0.3333, "f1_macro": 0.2663, "roc_auc": 0.5907},
            {"configuration": "4. Code + Viva (Bimodal)", "accuracy": 0.8278, "precision_macro": 0.7495, "recall_macro": 0.8733, "f1_macro": 0.7864, "roc_auc": 0.9792},
            {"configuration": "5. Code + Process (Bimodal)", "accuracy": 0.9123, "precision_macro": 0.9171, "recall_macro": 0.8737, "f1_macro": 0.8926, "roc_auc": 0.9604},
            {"configuration": "6. Viva + Process (Bimodal)", "accuracy": 0.9027, "precision_macro": 0.8342, "recall_macro": 0.9140, "f1_macro": 0.8653, "roc_auc": 0.9866},
            {"configuration": "7. Complete Multimodal Fusion (Proposed)", "accuracy": 0.9075, "precision_macro": 0.8582, "recall_macro": 0.9227, "f1_macro": 0.8850, "roc_auc": 0.9888}
        ]

        return {
            "status": "READY",
            "evaluated_at": datetime.now().isoformat(),
            "model_info": {
                "name": "Multimodal Fusion Transformer",
                "checkpoint": str(CHECKPOINT_PATH.name),
                "architecture": "2-Layer Transformer Encoder, 4 Attention Heads, d_model=64",
                "input_modalities": ["Python Code (768-D)", "Viva Explanations (384-D)", "Process Events (64-D)", "Numerical Features (24-D)"],
                "alignment_module": "Bidirectional Cross-Attention InfoNCE Alignment (64-D)"
            },
            "dataset_info": {
                "dataset_name": "DeepMisconception_Corrected_Dataset",
                "partition_evaluated": "test (Untouched Held-Out Split)",
                "total_test_sessions": test_data.get("total_test_sessions", 693),
                "evaluated_supervised_sessions": test_data.get("evaluated_supervised_sessions", 627),
                "excluded_insufficient_evidence_sessions": test_data.get("excluded_insufficient_evidence_sessions", 66),
                "disjoint_scenario_families": 9,
                "label_provenance": "Synthetic pedagogical proxy label (human_validated_label unpopulated)"
            },
            "overall_test_metrics": test_data["metrics"],
            "classwise_metrics": test_data["per_class"],
            "confusion_matrix": test_data["confusion_matrix"],
            "five_fold_cv": cv_data,
            "baselines_comparison": baselines
        }

    @classmethod
    def generate_excel_report(cls) -> bytes:
        """Generate a structured, professional Excel XML spreadsheet workbook."""
        data = cls.get_evaluation_metrics()
        if data["status"] != "READY":
            # Return simple text
            return b"MODEL_NOT_TRAINED"

        m = data["overall_test_metrics"]
        pc = data["classwise_metrics"]
        cm_raw = data["confusion_matrix"]["raw_counts"]
        cm_pct = data["confusion_matrix"]["normalized_percentages"]
        cv = data["five_fold_cv"]
        agg_cv = cv["aggregate_macro_metrics"]

        # Build Microsoft Excel XML 2003 Workbook format (natively supported by MS Excel, LibreOffice, Google Sheets)
        xml_lines = [
            '<?xml version="1.0"?>',
            '<?mso-application progid="Excel.Sheet"?>',
            '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"',
            ' xmlns:o="urn:schemas-microsoft-com:office:office"',
            ' xmlns:x="urn:schemas-microsoft-com:office:excel"',
            ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"',
            ' xmlns:html="http://www.w3.org/TR/REC-html40">',
            ' <Styles>',
            '  <Style ss:ID="Default" ss:Name="Normal">',
            '   <Alignment ss:Vertical="Center"/>',
            '   <Font ss:FontName="Calibri" ss:Size="11" ss:Color="#1E293B"/>',
            '  </Style>',
            '  <Style ss:ID="Header">',
            '   <Font ss:FontName="Calibri" ss:Size="11" ss:Color="#FFFFFF" ss:Bold="1"/>',
            '   <Interior ss:Color="#1E40AF" ss:Pattern="Solid"/>',
            '   <Alignment ss:Horizontal="Center" ss:Vertical="Center"/>',
            '  </Style>',
            '  <Style ss:ID="SubHeader">',
            '   <Font ss:FontName="Calibri" ss:Size="12" ss:Color="#0F172A" ss:Bold="1"/>',
            '   <Interior ss:Color="#E2E8F0" ss:Pattern="Solid"/>',
            '  </Style>',
            '  <Style ss:ID="BoldCell">',
            '   <Font ss:FontName="Calibri" ss:Size="11" ss:Color="#0F172A" ss:Bold="1"/>',
            '  </Style>',
            '  <Style ss:ID="PercentCell">',
            '   <NumberFormat ss:Format="0.00%"/>',
            '  </Style>',
            ' </Styles>'
        ]

        # Sheet 1: Overall Performance
        xml_lines.extend([
            ' <Worksheet ss:Name="Table 1 - Overall Performance">',
            '  <Table ss:DefaultColumnWidth="180">',
            '   <Row ss:Height="24">',
            '    <Cell ss:StyleID="SubHeader" ss:MergeAcross="1"><Data ss:Type="String">DeepMisconception - Overall Test Performance (N=627)</Data></Cell>',
            '   </Row>',
            '   <Row ss:Height="20">',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Metric</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Score</Data></Cell>',
            '   </Row>'
        ])

        rows_t1 = [
            ("Accuracy", f"{m['accuracy'] * 100:.2f}%"),
            ("Precision (Macro)", f"{m['precision_macro'] * 100:.2f}%"),
            ("Precision (Weighted)", f"{m['precision_weighted'] * 100:.2f}%"),
            ("Recall (Macro)", f"{m['recall_macro'] * 100:.2f}%"),
            ("Recall (Weighted)", f"{m['recall_weighted'] * 100:.2f}%"),
            ("F1 Score (Macro)", f"{m['f1_macro']:.4f}"),
            ("F1 Score (Weighted)", f"{m['f1_weighted']:.4f}"),
            ("Classification Error", f"{m['classification_error_pct']:.2f}%"),
            ("Sensitivity (Macro)", f"{m['sensitivity_macro'] * 100:.2f}%"),
            ("Specificity (Macro)", f"{m['specificity_macro'] * 100:.2f}%"),
            ("Multiclass ROC-AUC (OVR)", f"{m['roc_auc']:.4f}"),
            ("Multiclass PR-AUC (OVR)", f"{m['pr_auc']:.4f}"),
        ]
        for name, val in rows_t1:
            xml_lines.append(f'   <Row><Cell ss:StyleID="BoldCell"><Data ss:Type="String">{name}</Data></Cell><Cell><Data ss:Type="String">{val}</Data></Cell></Row>')

        xml_lines.extend(['  </Table>', ' </Worksheet>'])

        # Sheet 2: Class-wise Performance
        xml_lines.extend([
            ' <Worksheet ss:Name="Table 2 - Class-wise">',
            '  <Table ss:DefaultColumnWidth="140">',
            '   <Row ss:Height="24">',
            '    <Cell ss:StyleID="SubHeader" ss:MergeAcross="6"><Data ss:Type="String">Class-wise Diagnostic Performance Breakdown</Data></Cell>',
            '   </Row>',
            '   <Row ss:Height="20">',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Class</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Support</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Precision</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Recall</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">F1 Score</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Sensitivity</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Specificity</Data></Cell>',
            '   </Row>'
        ])
        for cname in ["Understood", "Partial Understanding", "Possible Misconception"]:
            cdata = pc[cname]
            xml_lines.append(
                f'   <Row>'
                f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{cname}</Data></Cell>'
                f'<Cell><Data ss:Type="Number">{cdata["support"]}</Data></Cell>'
                f'<Cell><Data ss:Type="String">{cdata["precision"]*100:.2f}%</Data></Cell>'
                f'<Cell><Data ss:Type="String">{cdata["recall"]*100:.2f}%</Data></Cell>'
                f'<Cell><Data ss:Type="String">{cdata["f1"]:.4f}</Data></Cell>'
                f'<Cell><Data ss:Type="String">{cdata["sensitivity"]*100:.2f}%</Data></Cell>'
                f'<Cell><Data ss:Type="String">{cdata["specificity"]*100:.2f}%</Data></Cell>'
                f'</Row>'
            )
        xml_lines.extend(['  </Table>', ' </Worksheet>'])

        # Sheet 3: 5-Fold Cross-Validation
        xml_lines.extend([
            ' <Worksheet ss:Name="Table 3 - 5-Fold CV">',
            '  <Table ss:DefaultColumnWidth="130">',
            '   <Row ss:Height="24">',
            '    <Cell ss:StyleID="SubHeader" ss:MergeAcross="7"><Data ss:Type="String">Stratified Group-Aware 5-Fold Cross-Validation (Train N=3,137)</Data></Cell>',
            '   </Row>',
            '   <Row ss:Height="20">',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Fold</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Accuracy</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Precision</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Recall</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">F1 Score</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Classification Error</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Sensitivity</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Specificity</Data></Cell>',
            '   </Row>'
        ])
        for f in cv["individual_folds"]:
            fm = f["metrics"]
            xml_lines.append(
                f'   <Row>'
                f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">Fold {f["fold"]}</Data></Cell>'
                f'<Cell><Data ss:Type="String">{fm["accuracy"]*100:.2f}%</Data></Cell>'
                f'<Cell><Data ss:Type="String">{fm["precision_macro"]*100:.2f}%</Data></Cell>'
                f'<Cell><Data ss:Type="String">{fm["recall_macro"]*100:.2f}%</Data></Cell>'
                f'<Cell><Data ss:Type="String">{fm["f1_macro"]:.4f}</Data></Cell>'
                f'<Cell><Data ss:Type="String">{fm["classification_error_pct"]:.2f}%</Data></Cell>'
                f'<Cell><Data ss:Type="String">{fm["sensitivity_macro"]*100:.2f}%</Data></Cell>'
                f'<Cell><Data ss:Type="String">{fm["specificity_macro"]*100:.2f}%</Data></Cell>'
                f'</Row>'
            )
        # Mean +- Std row
        xml_lines.append(
            f'   <Row>'
            f'<Cell ss:StyleID="Header"><Data ss:Type="String">Mean ± Std</Data></Cell>'
            f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{agg_cv["accuracy"]["mean"]*100:.2f} ± {agg_cv["accuracy"]["std"]*100:.2f}%</Data></Cell>'
            f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{agg_cv["precision_macro"]["mean"]*100:.2f} ± {agg_cv["precision_macro"]["std"]*100:.2f}%</Data></Cell>'
            f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{agg_cv["recall_macro"]["mean"]*100:.2f} ± {agg_cv["recall_macro"]["std"]*100:.2f}%</Data></Cell>'
            f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{agg_cv["f1_macro"]["mean"]:.4f} ± {agg_cv["f1_macro"]["std"]:.4f}</Data></Cell>'
            f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{agg_cv["classification_error_pct"]["mean"]:.2f} ± {agg_cv["classification_error_pct"]["std"]:.2f}%</Data></Cell>'
            f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{agg_cv["sensitivity_macro"]["mean"]*100:.2f} ± {agg_cv["sensitivity_macro"]["std"]*100:.2f}%</Data></Cell>'
            f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{agg_cv["specificity_macro"]["mean"]*100:.2f} ± {agg_cv["specificity_macro"]["std"]*100:.2f}%</Data></Cell>'
            f'</Row>'
        )
        xml_lines.extend(['  </Table>', ' </Worksheet>'])

        # Sheet 4: Confusion Matrix
        xml_lines.extend([
            ' <Worksheet ss:Name="Confusion Matrix">',
            '  <Table ss:DefaultColumnWidth="150">',
            '   <Row ss:Height="24">',
            '    <Cell ss:StyleID="SubHeader" ss:MergeAcross="3"><Data ss:Type="String">Test Confusion Matrix (Rows=True, Cols=Predicted)</Data></Cell>',
            '   </Row>',
            '   <Row ss:Height="20">',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Actual \\ Predicted</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Understood</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Partial Understanding</Data></Cell>',
            '    <Cell ss:StyleID="Header"><Data ss:Type="String">Possible Misconception</Data></Cell>',
            '   </Row>'
        ])
        classes_short = ["Understood", "Partial Understanding", "Possible Misconception"]
        for r_idx, rname in enumerate(classes_short):
            xml_lines.append(
                f'   <Row>'
                f'<Cell ss:StyleID="BoldCell"><Data ss:Type="String">{rname}</Data></Cell>'
                f'<Cell><Data ss:Type="String">{cm_raw[r_idx][0]} ({cm_pct[r_idx][0]:.1f}%)</Data></Cell>'
                f'<Cell><Data ss:Type="String">{cm_raw[r_idx][1]} ({cm_pct[r_idx][1]:.1f}%)</Data></Cell>'
                f'<Cell><Data ss:Type="String">{cm_raw[r_idx][2]} ({cm_pct[r_idx][2]:.1f}%)</Data></Cell>'
                f'</Row>'
            )
        xml_lines.extend(['  </Table>', ' </Worksheet>'])

        xml_lines.append('</Workbook>')
        return "\n".join(xml_lines).encode("utf-8")

    @classmethod
    def generate_csv_summary(cls) -> str:
        """Generate a clean CSV summary string containing all primary tables."""
        data = cls.get_evaluation_metrics()
        if data["status"] != "READY":
            return "Status,MODEL_NOT_TRAINED\n"

        out = io.StringIO()
        writer = csv.writer(out)

        writer.writerow(["# TABLE 1: OVERALL TEST PERFORMANCE (N=627)"])
        writer.writerow(["Metric", "Score"])
        m = data["overall_test_metrics"]
        writer.writerow(["Accuracy", f"{m['accuracy']*100:.2f}%"])
        writer.writerow(["Precision (Macro)", f"{m['precision_macro']*100:.2f}%"])
        writer.writerow(["Precision (Weighted)", f"{m['precision_weighted']*100:.2f}%"])
        writer.writerow(["Recall (Macro)", f"{m['recall_macro']*100:.2f}%"])
        writer.writerow(["Recall (Weighted)", f"{m['recall_weighted']*100:.2f}%"])
        writer.writerow(["F1 Score (Macro)", f"{m['f1_macro']:.4f}"])
        writer.writerow(["F1 Score (Weighted)", f"{m['f1_weighted']:.4f}"])
        writer.writerow(["Classification Error", f"{m['classification_error_pct']:.2f}%"])
        writer.writerow(["Sensitivity (Macro)", f"{m['sensitivity_macro']*100:.2f}%"])
        writer.writerow(["Specificity (Macro)", f"{m['specificity_macro']*100:.2f}%"])
        writer.writerow([])

        writer.writerow(["# TABLE 2: CLASS-WISE DIAGNOSTIC PERFORMANCE"])
        writer.writerow(["Class", "Support", "Precision", "Recall", "F1 Score", "Sensitivity", "Specificity"])
        pc = data["classwise_metrics"]
        for cname in ["Understood", "Partial Understanding", "Possible Misconception"]:
            c = pc[cname]
            writer.writerow([
                cname, c["support"],
                f"{c['precision']*100:.2f}%", f"{c['recall']*100:.2f}%",
                f"{c['f1']:.4f}", f"{c['sensitivity']*100:.2f}%", f"{c['specificity']*100:.2f}%"
            ])
        writer.writerow([])

        writer.writerow(["# TABLE 3: FIVE-FOLD STRATIFIED GROUP CROSS-VALIDATION"])
        writer.writerow(["Fold", "Accuracy", "Precision", "Recall", "F1 Score", "Classification Error", "Sensitivity", "Specificity"])
        cv = data["five_fold_cv"]
        for f in cv["individual_folds"]:
            fm = f["metrics"]
            writer.writerow([
                f"Fold {f['fold']}",
                f"{fm['accuracy']*100:.2f}%", f"{fm['precision_macro']*100:.2f}%", f"{fm['recall_macro']*100:.2f}%",
                f"{fm['f1_macro']:.4f}", f"{fm['classification_error_pct']:.2f}%",
                f"{fm['sensitivity_macro']*100:.2f}%", f"{fm['specificity_macro']*100:.2f}%"
            ])
        agg = cv["aggregate_macro_metrics"]
        writer.writerow([
            "Mean +/- Std",
            f"{agg['accuracy']['mean']*100:.2f} +/- {agg['accuracy']['std']*100:.2f}%",
            f"{agg['precision_macro']['mean']*100:.2f} +/- {agg['precision_macro']['std']*100:.2f}%",
            f"{agg['recall_macro']['mean']*100:.2f} +/- {agg['recall_macro']['std']*100:.2f}%",
            f"{agg['f1_macro']['mean']:.4f} +/- {agg['f1_macro']['std']:.4f}",
            f"{agg['classification_error_pct']['mean']:.2f} +/- {agg['classification_error_pct']['std']:.2f}%",
            f"{agg['sensitivity_macro']['mean']*100:.2f} +/- {agg['sensitivity_macro']['std']*100:.2f}%",
            f"{agg['specificity_macro']['mean']*100:.2f} +/- {agg['specificity_macro']['std']*100:.2f}%"
        ])
        writer.writerow([])

        writer.writerow(["# CONFUSION MATRIX"])
        writer.writerow(["Actual \\ Predicted", "Understood", "Partial Understanding", "Possible Misconception"])
        cm_raw = data["confusion_matrix"]["raw_counts"]
        cm_pct = data["confusion_matrix"]["normalized_percentages"]
        for r_idx, rname in enumerate(["Understood", "Partial Understanding", "Possible Misconception"]):
            writer.writerow([
                rname,
                f"{cm_raw[r_idx][0]} ({cm_pct[r_idx][0]:.1f}%)",
                f"{cm_raw[r_idx][1]} ({cm_pct[r_idx][1]:.1f}%)",
                f"{cm_raw[r_idx][2]} ({cm_pct[r_idx][2]:.1f}%)"
            ])

        return out.getvalue()
