import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_backbone_comparison(output_path="Results/confronto_backbone.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig_w, fig_h = 15.5, 4.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    font_main = "DejaVu Sans"

    # Titolo principale
    ax.text(
        fig_w / 2,
        fig_h - 0.55,
        "Confronto Backbone — Risultati Fused RGBT (Test Set FLAME3 Zero-Shot, 256x256)",
        fontsize=14.5,
        fontweight="bold",
        color="#111827",
        ha="center",
        va="center",
        family=font_main,
    )

    headers = [
        "Backbone (Fused RGBT)",
        "Parametri (FP16)",
        "GFLOPs",
        "Loss",
        "Accuracy",
        "Balanced Acc",
        "F1-Score",
        "Precision",
        "Recall",
        "FN",
        "FP",
    ]
    col_widths = [2.9, 1.8, 1.1, 1.0, 1.25, 1.4, 1.25, 1.25, 1.25, 0.65, 0.65]  # tot = 14.5

    rows = [
        [
            ["resnet50 (Baseline)", col_widths[0], "left"],
            ["47.0M (~94 MB)", col_widths[1], "center"],
            ["8.18", col_widths[2], "center"],
            ["0.1172", col_widths[3], "center"],
            ["95.26%", col_widths[4], "center"],
            ["95.78%", col_widths[5], "center"],
            ["97.12%", col_widths[6], "center"],
            ["99.33%", col_widths[7], "center"],
            ["95.02%", col_widths[8], "center"],
            ["31", col_widths[9], "center"],
            ["4", col_widths[10], "center"],
        ],
        [
            ["mobilenetv3_large_100", col_widths[0], "left"],
            ["8.4M (~17 MB)", col_widths[1], "center"],
            [("0.44", "#065F46"), col_widths[2], "center"],
            ["0.2337", col_widths[3], "center"],
            ["88.62%", col_widths[4], "center"],
            ["92.90%", col_widths[5], "center"],
            ["92.77%", col_widths[6], "center"],
            ["99.81%", col_widths[7], "center"],
            ["86.66%", col_widths[8], "center"],
            [("83", "#B91C1C"), col_widths[9], "center"],
            ["1", col_widths[10], "center"],
        ],
        [
            ["regnety_008", col_widths[0], "left"],
            ["11.0M (~22 MB)", col_widths[1], "center"],
            ["1.66", col_widths[2], "center"],
            ["0.1207", col_widths[3], "center"],
            ["94.44%", col_widths[4], "center"],
            ["93.90%", col_widths[5], "center"],
            ["96.64%", col_widths[6], "center"],
            ["98.66%", col_widths[7], "center"],
            ["94.69%", col_widths[8], "center"],
            ["33", col_widths[9], "center"],
            ["8", col_widths[10], "center"],
        ],
        [
            ["efficientnet_b0", col_widths[0], "left"],
            ["8.0M (~16 MB)", col_widths[1], "center"],
            ["0.78", col_widths[2], "center"],
            ["0.1766", col_widths[3], "center"],
            ["92.68%", col_widths[4], "center"],
            ["93.56%", col_widths[5], "center"],
            ["95.51%", col_widths[6], "center"],
            ["98.97%", col_widths[7], "center"],
            ["92.28%", col_widths[8], "center"],
            ["48", col_widths[9], "center"],
            ["6", col_widths[10], "center"],
        ],
        [
            [("resnet18 (Miglior Modello)", "#065F46"), col_widths[0], "left"],
            ["22.4M (~45 MB)", col_widths[1], "center"],
            ["3.62", col_widths[2], "center"],
            [("0.1111", "#065F46"), col_widths[3], "center"],
            [("97.70%", "#065F46"), col_widths[4], "center"],
            [("96.18%", "#065F46"), col_widths[5], "center"],
            [("98.63%", "#065F46"), col_widths[6], "center"],
            ["98.87%", col_widths[7], "center"],
            [("98.39%", "#065F46"), col_widths[8], "center"],
            [("10", "#065F46"), col_widths[9], "center"],
            ["7", col_widths[10], "center"],
        ],
        [
            ["regnety_004", col_widths[0], "left"],
            [("7.8M (~16 MB)", "#065F46"), col_widths[1], "center"],
            ["0.80", col_widths[2], "center"],
            ["0.1746", col_widths[3], "center"],
            ["92.01%", col_widths[4], "center"],
            ["95.26%", col_widths[5], "center"],
            ["95.02%", col_widths[6], "center"],
            [("100.00%", "#065F46"), col_widths[7], "center"],
            ["90.51%", col_widths[8], "center"],
            ["59", col_widths[9], "center"],
            [("0", "#065F46"), col_widths[10], "center"],
        ],
    ]

    x = 0.5
    curr_y = fig_h - 1.15
    row_height = 0.48
    tot_w = sum(col_widths)

    # Header
    hdr_rect = patches.Rectangle(
        (x, curr_y - row_height),
        tot_w,
        row_height,
        facecolor="#F3F4F6",
        edgecolor="#9CA3AF",
        linewidth=0.9,
    )
    ax.add_patch(hdr_rect)

    cur_x = x
    for h, w in zip(headers, col_widths):
        ax.text(
            cur_x + w / 2,
            curr_y - row_height / 2,
            h,
            fontsize=10.0,
            fontweight="bold",
            color="#111827",
            ha="center",
            va="center",
            family=font_main,
        )
        cur_x += w

    curr_y -= row_height

    # Data rows
    for r_idx, row in enumerate(rows):
        row_bg = "#ECFDF5" if r_idx == 4 else ("#F9FAFB" if r_idx % 2 == 1 else "#FFFFFF")
        r_rect = patches.Rectangle(
            (x, curr_y - row_height),
            tot_w,
            row_height,
            facecolor=row_bg,
            edgecolor="#D1D5DB",
            linewidth=0.7,
        )
        ax.add_patch(r_rect)

        cur_x = x
        for val, w, align in row:
            t_x = (
                cur_x + w / 2
                if align == "center"
                else (cur_x + 0.15 if align == "left" else cur_x + w - 0.15)
            )
            val_text = val[0] if isinstance(val, tuple) else str(val)
            font_weight = "bold" if isinstance(val, tuple) else "normal"
            color = val[1] if isinstance(val, tuple) and len(val) > 1 else "#1F2937"

            ax.text(
                t_x,
                curr_y - row_height / 2,
                val_text,
                fontsize=10.0,
                fontweight=font_weight,
                color=color,
                ha=align,
                va="center",
                family=font_main,
            )
            cur_x += w

        curr_y -= row_height

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor="white", edgecolor="none")
    plt.close()
    print(f"Tabella confronto backbone salvata in: {output_path}")


if __name__ == "__main__":
    create_backbone_comparison()
