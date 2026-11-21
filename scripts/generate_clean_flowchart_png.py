import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart(output_path):
    # Canvas size: width 10 inches, height 14 inches
    fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')

    # Styling constants
    BOX_EDGE = '#000000'
    TEXT_COLOR = '#000000'

    # Helper: draw labeled box
    def draw_box(x, y, w, h, title, subtitle_lines=[], boxstyle="square", linestyle="-", linewidth=1.5):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2" if boxstyle == "round" else "square,pad=0.0",
            facecolor="white",
            edgecolor=BOX_EDGE,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=3
        )
        ax.add_patch(rect)
        cx = x + w / 2.0
        
        if not subtitle_lines:
            ax.text(cx, y + h / 2.0, title, ha='center', va='center',
                    fontsize=10, fontweight='bold', color=TEXT_COLOR, zorder=4)
        else:
            ax.text(cx, y + h - 1.8, title, ha='center', va='top',
                    fontsize=10, fontweight='bold', color=TEXT_COLOR, zorder=4)
            # Divider line below title
            ax.plot([x + 1.5, x + w - 1.5], [y + h - 3.2, y + h - 3.2],
                    color=BOX_EDGE, linewidth=0.8, zorder=4)
            start_y = y + h - 4.5
            for i, line in enumerate(subtitle_lines):
                fontweight = 'bold' if line.startswith('•') else 'normal'
                fontsize = 8.5 if line.startswith('•') else 8.0
                ax.text(cx, start_y - (i * 2.0), line, ha='center', va='top',
                        fontsize=fontsize, fontweight=fontweight, color=TEXT_COLOR, zorder=4)

    # Helper: draw arrow
    def draw_arrow(x1, y1, x2, y2, linewidth=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=BOX_EDGE, lw=linewidth),
                    zorder=2)

    # ==================== PHASE 1 ====================
    ax.add_patch(patches.Rectangle((3, 116), 94, 21, facecolor='white', edgecolor=BOX_EDGE, linestyle='--', linewidth=1.2, zorder=1))
    ax.text(5, 134.5, "PHASE 1: MULTIMODAL DATA INPUT & REPRESENTATION", fontsize=9.5, fontweight='bold', color=TEXT_COLOR)

    # Stream 1: Code
    draw_box(6, 118, 28, 15, "Source Code Modality",
             ["• Python AST Parsing", "16 Structural Metrics (Complexity, Depth)",
              "• Pretrained CodeBERT", "768-D Contextual Vector"], linewidth=1.8)

    # Stream 2: Viva
    draw_box(36, 118, 28, 15, "Oral Viva Voce Modality",
             ["• Grounded Conceptual Q&A", "Testing Invariants & State Mutations",
              "• Pretrained Sentence-BERT", "384-D Frozen Semantic Vector"], linewidth=1.8)

    # Stream 3: Process
    draw_box(66, 118, 28, 15, "Process Telemetry Modality",
             ["• Chronological IDE Keystrokes", "128-Step Action Sequence Stream",
              "• Process Transformer (64-D)", "+ 24 Z-Score Normalized Features"], linewidth=1.8)

    # Arrows Phase 1 -> Phase 2
    draw_arrow(20, 118, 20, 107)
    draw_arrow(50, 118, 50, 112)
    draw_arrow(80, 118, 80, 89)

    # ==================== PHASE 2 ====================
    ax.add_patch(patches.Rectangle((3, 93), 94, 20, facecolor='white', edgecolor=BOX_EDGE, linestyle='--', linewidth=1.2, zorder=1))
    ax.text(5, 110.5, "PHASE 2: EVIDENCE STATUS FILTERING & PRE-PROCESSING", fontsize=9.5, fontweight='bold', color=TEXT_COLOR)

    # Code Preprocessing Box
    draw_box(6, 96, 28, 9, "Code Tokenization & AST",
             ["Fixed Length & Attention Masks", "16 AST Complexity Features"])
    draw_arrow(20, 96, 20, 89)

    # Diamond for Viva
    diamond = patches.Polygon([[50, 108], [61, 102], [50, 96], [39, 102]],
                              closed=True, facecolor='white', edgecolor=BOX_EDGE, linewidth=1.8, zorder=3)
    ax.add_patch(diamond)
    ax.text(50, 102.8, "Viva Oral", ha='center', va='center', fontsize=8.5, fontweight='bold', color=TEXT_COLOR, zorder=4)
    ax.text(50, 100.8, "Evidence Valid?", ha='center', va='center', fontsize=8.5, fontweight='bold', color=TEXT_COLOR, zorder=4)

    # Valid -> Down
    draw_arrow(50, 96, 50, 89)
    ax.text(51.5, 92.5, "YES", fontsize=7.5, fontweight='bold', color=TEXT_COLOR)

    # Invalid -> Right
    draw_arrow(61, 102, 69, 102)
    ax.text(65, 103.5, "NO (Evasive)", ha='center', fontsize=7.5, fontweight='bold', color=TEXT_COLOR)
    draw_box(69, 97, 25, 10, "Status Gate Activated",
             ["Pending / Insufficient Evidence", "(Routed to Instructor Review)"], linestyle="--")

    # ==================== PHASE 3: STAGE A ALIGNMENT ====================
    ax.add_patch(patches.Rectangle((3, 67), 94, 23, facecolor='white', edgecolor=BOX_EDGE, linestyle='--', linewidth=1.2, zorder=1))
    ax.text(5, 87.5, "PHASE 3: STAGE A - CROSS-MODAL LATENT CONTRASTIVE ALIGNMENT", fontsize=9.5, fontweight='bold', color=TEXT_COLOR)

    draw_box(6, 76, 28, 8, "Projection W_code", ["Linear + LayerNorm (768 -> 128)"])
    draw_box(36, 76, 28, 8, "Projection W_viva", ["Linear + LayerNorm (384 -> 128)"])
    draw_box(66, 76, 28, 8, "Projection W_proc", ["Linear + LayerNorm (64 -> 128)"])

    # Converging to Loss
    draw_arrow(20, 76, 38, 73.5)
    draw_arrow(50, 76, 50, 73.5)
    draw_arrow(80, 76, 62, 73.5)

    draw_box(25, 68.5, 50, 5, "InfoNCE Contrastive & Cosine Alignment Loss",
             ["L_align = InfoNCE(z_c, z_v) + λ_1 InfoNCE(z_c, z_p) + λ_2 Cosine(z_v, z_p)"], linewidth=1.8)
    draw_arrow(50, 68.5, 50, 63)

    # ==================== PHASE 4: STAGE B FUSION TRANSFORMER ====================
    ax.add_patch(patches.Rectangle((3, 37), 94, 27, facecolor='white', edgecolor=BOX_EDGE, linestyle='--', linewidth=1.2, zorder=1))
    ax.text(5, 61.5, "PHASE 4: STAGE B - MULTIMODAL FUSION TRANSFORMER", fontsize=9.5, fontweight='bold', color=TEXT_COLOR)

    # Token Assembly Box
    draw_box(16, 54, 68, 5, "Multimodal Sequence: H_0 = [ z_code + e_c ; z_viva + e_v ; z_proc + e_p ] ∈ R^(3 × 128)")
    draw_arrow(50, 54, 50, 50.5)

    # 4-Layer Transformer Block
    draw_box(16, 42.5, 68, 8, "4-Layer Bidirectional Transformer Encoder",
             ["• 8 Multi-Head Attention Heads (d_k = 16) | Cross-Modal Self-Attention",
              "• Feedforward FFN = 512 | Residual LayerNorm Connections | Dropout = 0.1"], linewidth=2.0)
    draw_arrow(50, 42.5, 50, 39.5)

    # Mean Pooling
    draw_box(25, 38, 50, 3.5, "Sequence Mean Pooling: z_fused = Mean(H_4, axis=tokens) ∈ R^128")
    draw_arrow(50, 38, 50, 33)

    # ==================== PHASE 5: STAGE C CLASSIFICATION ====================
    ax.add_patch(patches.Rectangle((3, 16), 94, 18, facecolor='white', edgecolor=BOX_EDGE, linestyle='--', linewidth=1.2, zorder=1))
    ax.text(5, 31.5, "PHASE 5: STAGE C - CLASSIFICATION HEAD & DIAGNOSIS", fontsize=9.5, fontweight='bold', color=TEXT_COLOR)

    draw_box(25, 25.5, 50, 5, "Linear Classification Head + GELU + Dropout(0.1)",
             ["y_hat = Softmax( W_2 · GELU(LayerNorm(W_1 · z_fused + b_1)) + b_2 )"])

    # Branching arrows
    draw_arrow(38, 25.5, 20, 23.5)
    draw_arrow(50, 25.5, 50, 23.5)
    draw_arrow(62, 25.5, 80, 23.5)

    # 3 Diagnostic Classes
    draw_box(6, 17.5, 28, 6, "Understood", ["Accurate Syntax & Trace", "Sound Conceptual Explanation"], linewidth=1.8)
    draw_box(36, 17.5, 28, 6, "Partial Understanding", ["Functional Code / Fragile Logic", "Edge-Case Uncertainty"], linewidth=1.8)
    draw_box(66, 17.5, 28, 6, "Possible Misconception", ["Verified Contradictions", "Oral Claims Clash with Traces"], linewidth=1.8)

    # ==================== PHASE 6: TESTING PROTOCOL ====================
    ax.add_patch(patches.Rectangle((3, 2), 94, 11.5, facecolor='white', edgecolor=BOX_EDGE, linestyle='--', linewidth=1.2, zorder=1))
    ax.text(5, 11.2, "PHASE 6: VERIFICATION & TESTING PROTOCOL", fontsize=9.5, fontweight='bold', color=TEXT_COLOR)

    draw_box(5, 3.5, 43, 7, "Held-Out Test Set (N = 627)",
             ["• Accuracy: 90.75% | Error Rate: 9.25%",
              "• Macro Precision: 85.82% | Macro Recall: 92.27%",
              "• Macro Specificity: 95.63% | Macro F1: 0.8850 | AUROC: 0.9888"])

    draw_box(52, 3.5, 43, 7, "5-Fold Group Cross-Validation (N = 3,137)",
             ["• 46 Independent Scenario Families (Zero Leakage)",
              "• Mean Accuracy: 87.23% ± 4.12% | Mean Error: 12.77% ± 4.12%",
              "• Mean Macro F1: 0.8284 ± 0.0336 | AUROC: 0.9619 ± 0.0088"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Flowchart generated: {output_path}")

if __name__ == "__main__":
    draw_flowchart("figures/flowchart.png")
    draw_flowchart("DeepMisconception/plots/flowchart.png")
