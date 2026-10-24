import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_simple_test_results(output_path="Results/risultati_test.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig_w, fig_h = 15.0, 10.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    font_main = "DejaVu Sans"

    # Titolo principale
    ax.text(fig_w / 2, fig_h - 0.65, "Risultati di Test nei 3 Esperimenti: RGBT vs Solo RGB vs Solo Termica",
            fontsize=16, fontweight="bold", color="#111827", ha="center", va="center", family=font_main)

    def draw_table(x, y, col_widths, headers, rows, title, row_height=0.42):
        ax.text(x, y + 0.20, title, fontsize=11.5, fontweight="bold", color="#1F2937", va="bottom", family=font_main)

        curr_y = y
        tot_w = sum(col_widths)

        # Header
        hdr_rect = patches.Rectangle((x, curr_y - row_height), tot_w, row_height,
                                     facecolor="#F3F4F6", edgecolor="#9CA3AF", linewidth=0.8)
        ax.add_patch(hdr_rect)

        cur_x = x
        for h, w in zip(headers, col_widths):
            ax.text(cur_x + w / 2, curr_y - row_height / 2, h, fontsize=9.5, fontweight="bold",
                    color="#111827", ha="center", va="center", family=font_main)
            cur_x += w

        curr_y -= row_height

        # Data rows
        for r_idx, row in enumerate(rows):
            row_bg = "#F9FAFB" if r_idx % 2 == 1 else "#FFFFFF"
            r_rect = patches.Rectangle((x, curr_y - row_height), tot_w, row_height,
                                       facecolor=row_bg, edgecolor="#D1D5DB", linewidth=0.6)
            ax.add_patch(r_rect)

            cur_x = x
            for val, w, align in row:
                t_x = cur_x + w / 2 if align == "center" else (cur_x + 0.15 if align == "left" else cur_x + w - 0.15)
                val_text = val[0] if isinstance(val, tuple) else str(val)
                font_weight = "bold" if isinstance(val, tuple) else "normal"
                color = val[1] if isinstance(val, tuple) and len(val) > 1 else "#1F2937"

                ax.text(t_x, curr_y - row_height / 2, val_text, fontsize=9.5, fontweight=font_weight,
                        color=color, ha=align, va="center", family=font_main)
                cur_x += w

            curr_y -= row_height

        return curr_y

    headers = [
        "Modalità di Input", "Accuracy", "Balanced Acc", "F1-Score",
        "Precision", "Recall", "Falsi Negativi (FN)", "Falsi Positivi (FP)"
    ]
    widths = [3.0, 1.5, 1.6, 1.5, 1.6, 1.6, 1.6, 1.6]  # tot = 14.0

    # =========================================================================
    # 1. TEST DATASET MISTO (FIRST_TEST - 820 campioni)
    # =========================================================================
    y_1 = fig_h - 1.45
    rows_1 = [
        [
            [("RGBT (4 Canali)", "#1E3A8A"), widths[0], "left"],
            ["95.12%", widths[1], "center"],
            ["96.92%", widths[2], "center"],
            ["96.82%", widths[3], "center"],
            ["100.00%", widths[4], "center"],
            [("93.84%", "#1E3A8A"), widths[5], "center"],
            [("40 (Incendi mancati)", "#B91C1C"), widths[6], "center"],
            ["0", widths[7], "center"]
        ],
        [
            ["Solo RGB (Termico=0.5)", widths[0], "left"],
            ["89.15%", widths[1], "center"],
            ["76.13%", widths[2], "center"],
            ["93.49%", widths[3], "center"],
            ["89.00%", widths[4], "center"],
            ["98.46%", widths[5], "center"],
            ["10", widths[6], "center"],
            ["79", widths[7], "center"]
        ],
        [
            ["Solo Termica (RGB=ImageNet)", widths[0], "left"],
            ["74.02%", widths[1], "center"],
            ["49.56%", widths[2], "center"],
            ["84.80%", widths[3], "center"],
            ["78.99%", widths[4], "center"],
            ["91.53%", widths[5], "center"],
            [("55", "#B91C1C"), widths[6], "center"],
            ["158", widths[7], "center"]
        ],
    ]
    end_y1 = draw_table(0.5, y_1, widths, headers, rows_1,
                        title="1. Esperimento 1: Dataset Misto (First_test — Test Set Misto: 820 immagini)")

    # =========================================================================
    # 2. TEST CROSS DATASET FLAME3 (FLAME3CROSS - 738 campioni unseen)
    # =========================================================================
    y_2 = end_y1 - 0.65
    rows_2 = [
        [
            [("RGBT (4 Canali)", "#1E3A8A"), widths[0], "left"],
            ["92.01%", widths[1], "center"],
            ["91.75%", widths[2], "center"],
            ["95.10%", widths[3], "center"],
            ["98.28%", widths[4], "center"],
            [("92.12%", "#1E3A8A"), widths[5], "center"],
            [("49 (Incendi mancati)", "#B91C1C"), widths[6], "center"],
            ["10", widths[7], "center"]
        ],
        [
            ["Solo RGB (Termico=0.5)", widths[0], "left"],
            ["91.46%", widths[1], "center"],
            ["87.92%", widths[2], "center"],
            ["94.84%", widths[3], "center"],
            ["96.66%", widths[4], "center"],
            ["93.09%", widths[5], "center"],
            ["43", widths[6], "center"],
            ["20", widths[7], "center"]
        ],
        [
            ["Solo Termica (RGB=ImageNet)", widths[0], "left"],
            ["84.82%", widths[1], "center"],
            ["51.72%", widths[2], "center"],
            ["91.74%", widths[3], "center"],
            ["84.74%", widths[4], "center"],
            [("100.00%", "#065F46"), widths[5], "center"],
            [("0 (Nessun incendio perso)", "#065F46"), widths[6], "center"],
            ["112", widths[7], "center"]
        ],
    ]
    end_y2 = draw_table(0.5, y_2, widths, headers, rows_2,
                        title="2. Esperimento 2: Cross Dataset (Flame3Cross — Test Set FLAME3 100% Unseen: 738 immagini)")

    # =========================================================================
    # 3. TEST CROSS DATASET CON CART (FLAME3CROSS_CART - 738 campioni unseen)
    # =========================================================================
    y_3 = end_y2 - 0.65
    rows_3 = [
        [
            [("RGBT (4 Canali)", "#1E3A8A"), widths[0], "left"],
            ["89.57%", widths[1], "center"],
            ["92.76%", widths[2], "center"],
            ["93.44%", widths[3], "center"],
            ["99.46%", widths[4], "center"],
            [("88.10%", "#B91C1C"), widths[5], "center"],
            [("74 (Incendi mancati)", "#B91C1C"), widths[6], "center"],
            ["3", widths[7], "center"]
        ],
        [
            ["Solo RGB (Termico=0.5)", widths[0], "left"],
            ["88.21%", widths[1], "center"],
            ["92.31%", widths[2], "center"],
            ["92.51%", widths[3], "center"],
            ["99.63%", widths[4], "center"],
            ["86.33%", widths[5], "center"],
            [("85", "#B91C1C"), widths[6], "center"],
            ["2", widths[7], "center"]
        ],
        [
            ["Solo Termica (RGB=ImageNet)", widths[0], "left"],
            ["85.64%", widths[1], "center"],
            ["54.31%", widths[2], "center"],
            ["92.15%", widths[3], "center"],
            ["85.44%", widths[4], "center"],
            [("100.00%", "#065F46"), widths[5], "center"],
            [("0 (Nessun incendio perso)", "#065F46"), widths[6], "center"],
            ["106", widths[7], "center"]
        ],
    ]
    end_y3 = draw_table(0.5, y_3, widths, headers, rows_3,
                        title="3. Esperimento 3: Cross Dataset con CART (Flame3Cross_CART — Test Set FLAME3 100% Unseen: 738 immagini)")

    # =========================================================================
    # 4. TABELLA DI CONFRONTO DIRETTO DEI 3 MODELLI (MODALITÀ RGBT A 4 CANALI)
    # =========================================================================
    y_4 = end_y3 - 0.65
    headers_4 = [
        "Esperimento (RGBT)", "Test Set", "Accuracy", "Balanced Acc",
        "F1-Score", "Precision", "Recall", "Falsi Negativi (FN)", "Falsi Positivi (FP)"
    ]
    widths_4 = [2.8, 2.0, 1.3, 1.4, 1.3, 1.3, 1.3, 1.4, 1.2]  # tot = 14.0
    rows_4 = [
        [
            [("1. Dataset Misto (First_test)", "#111827"), widths_4[0], "left"],
            ["820 img (Misto)", widths_4[1], "center"],
            ["95.12%", widths_4[2], "center"],
            ["96.92%", widths_4[3], "center"],
            ["96.82%", widths_4[4], "center"],
            ["100.00%", widths_4[5], "center"],
            ["93.84%", widths_4[6], "center"],
            ["40", widths_4[7], "center"],
            ["0", widths_4[8], "center"]
        ],
        [
            [("2. Cross Dataset (Flame3Cross)", "#111827"), widths_4[0], "left"],
            ["738 img (FLAME3)", widths_4[1], "center"],
            ["92.01%", widths_4[2], "center"],
            ["91.75%", widths_4[3], "center"],
            ["95.10%", widths_4[4], "center"],
            ["98.28%", widths_4[5], "center"],
            [("92.12%", "#065F46"), widths_4[6], "center"],
            [("49", "#065F46"), widths_4[7], "center"],
            ["10", widths_4[8], "center"]
        ],
        [
            [("3. Cross Dataset + CART", "#111827"), widths_4[0], "left"],
            ["738 img (FLAME3)", widths_4[1], "center"],
            ["89.57%", widths_4[2], "center"],
            ["92.76%", widths_4[3], "center"],
            ["93.44%", widths_4[4], "center"],
            ["99.46%", widths_4[5], "center"],
            [("88.10%", "#B91C1C"), widths_4[6], "center"],
            [("74 (+25 mancati)", "#B91C1C"), widths_4[7], "center"],
            ["3", widths_4[8], "center"]
        ],
    ]
    draw_table(0.5, y_4, widths_4, headers_4, rows_4,
               title="4. Confronto Diretto RGBT tra i 3 Esperimenti (Evidenza su Falsi Negativi e Recall)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor="white", edgecolor="none")
    plt.savefig("Results/confronto_esperimenti.png", dpi=200, facecolor="white", edgecolor="none")
    plt.close()
    print(f"Immagine aggiornata con evidenza sui Falsi Negativi salvata in: {output_path}")

if __name__ == "__main__":
    create_simple_test_results()
