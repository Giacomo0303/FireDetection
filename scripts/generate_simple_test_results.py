import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_simple_test_results(output_path="Results/risultati_test.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Dimensioni compatte e proporzionate: 16 x 11.6 pollici a 200 DPI (3200 x 2320 px)
    fig_w, fig_h = 16.0, 11.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    font_main = "DejaVu Sans"

    # Titolo semplice e pulito
    ax.text(fig_w / 2, fig_h - 0.7, "Risultati dei Test nei 3 Esperimenti (ResNet-50 RGBT)",
            fontsize=18, fontweight="bold", color="#111827", ha="center", va="center", family=font_main)

    def draw_simple_table(x, y, col_widths, headers, rows, title, row_height=0.40):
        # Titolo sezione
        ax.text(x, y + 0.22, title, fontsize=12, fontweight="bold", color="#1F2937", va="bottom", family=font_main)

        curr_y = y
        tot_w = sum(col_widths)

        # Header
        hdr_rect = patches.Rectangle((x, curr_y - row_height), tot_w, row_height,
                                     facecolor="#E5E7EB", edgecolor="#9CA3AF", linewidth=0.8)
        ax.add_patch(hdr_rect)

        cur_x = x
        for h, w in zip(headers, col_widths):
            ax.text(cur_x + w / 2, curr_y - row_height / 2, h, fontsize=9.5, fontweight="bold",
                    color="#111827", ha="center", va="center", family=font_main)
            cur_x += w

        curr_y -= row_height

        # Righe dati
        for r_idx, row in enumerate(rows):
            is_last = (r_idx == len(rows) - 1)
            row_bg = "#F9FAFB" if r_idx % 2 == 1 else "#FFFFFF"
            if is_last and "TOTALE" in str(row[0][0]):
                row_bg = "#EEF2FF"  # Evidenzia leggermente la riga totale

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

    # =========================================================================
    # 1. TABELLA: TEST DATASET MISTO (FIRST_TEST)
    # =========================================================================
    y_t1 = fig_h - 1.5
    headers_1 = ["Dataset nel Test Set", "N. Immagini", "Accuracy", "Recall (Fire)", "Specificità (No_Fire)", "Falsi Positivi (FP)"]
    widths_1 = [3.8, 2.0, 2.0, 2.4, 2.4, 2.4]  # tot = 15.0
    rows_1 = [
        ["FLAME2", "308", "87.34%", "123 / 162 (75.9%)", "146 / 146 (100.0%)", "0"],
        ["FLAME3 (In-Domain)", "124", "100.00%", "110 / 110 (100.0%)", "14 / 14 (100.0%)", "0"],
        ["FireMan-UAV", "181", "100.00%", "170 / 170 (100.0%)", "11 / 11 (100.0%)", "0"],
        ["RGBT-3M", "207", "99.52%", "206 / 207 (99.5%)", "—", "0"],
        [[("TOTALE TEST SET MISTO", "#1E3A8A"), widths_1[0], "left"],
         [("820", "#1E3A8A"), widths_1[1], "center"],
         [("95.12%", "#1E3A8A"), widths_1[2], "center"],
         [("609 / 649 (93.8%)", "#1E3A8A"), widths_1[3], "center"],
         [("171 / 171 (100.0%)", "#065F46"), widths_1[4], "center"],
         [("0", "#065F46"), widths_1[5], "center"]]
    ]
    # Dati formattati
    fmt_rows_1 = [
        [[r[0], widths_1[0], "left"], [r[1], widths_1[1], "center"], [r[2], widths_1[2], "center"],
         [r[3], widths_1[3], "center"], [r[4], widths_1[4], "center"], [r[5], widths_1[5], "center"]]
        for r in rows_1[:-1]
    ]
    fmt_rows_1.append([
        [("TOTALE TEST SET MISTO", "#1E3A8A"), widths_1[0], "left"],
        [("820", "#1E3A8A"), widths_1[1], "center"],
        [("95.12%", "#1E3A8A"), widths_1[2], "center"],
        [("609 / 649 (93.8%)", "#1E3A8A"), widths_1[3], "center"],
        [("171 / 171 (100.0%)", "#065F46"), widths_1[4], "center"],
        [("0 / 171", "#065F46"), widths_1[5], "center"]
    ])
    end_y1 = draw_simple_table(0.5, y_t1, widths_1, headers_1, fmt_rows_1,
                               title="1. Test con Dataset Misto (First_test — Intra-Dataset: 820 immagini, Loss: 0.0837, BalAcc: 96.93%, F1: 96.84%)")

    # =========================================================================
    # 2. TABELLA: TEST CROSS DATASET FLAME3 (FLAME3CROSS)
    # =========================================================================
    y_t2 = end_y1 - 0.75
    headers_2 = ["Classe (FLAME3 100% Unseen)", "Campioni", "Predetti Corretti", "Errori", "Metrica Ottenuta", "Esito Operativo"]
    widths_2 = [3.8, 2.0, 2.4, 2.0, 2.4, 2.4]  # tot = 15.0
    fmt_rows_2 = [
        [[("Fire (1.0)", "#111827"), widths_2[0], "left"],
         ["622", widths_2[1], "center"],
         ["573 (Veri Positivi)", widths_2[2], "center"],
         ["49 (Falsi Negativi)", widths_2[3], "center"],
         ["Recall: 92.12%", widths_2[4], "center"],
         ["Focolai rilevati", widths_2[5], "center"]],
        [[("No_Fire (0.0)", "#111827"), widths_2[0], "left"],
         ["116", widths_2[1], "center"],
         ["106 (Veri Negativi)", widths_2[2], "center"],
         [("10 (Falsi Positivi)", "#DC2626"), widths_2[3], "center"],
         ["Specificità: 91.38%", widths_2[4], "center"],
         [("10 Falsi Allarmi su rocce/laghi", "#DC2626"), widths_2[5], "center"]],
        [[("TOTALE TEST FLAME3", "#1E3A8A"), widths_2[0], "left"],
         [("738", "#1E3A8A"), widths_2[1], "center"],
         [("679 corretti (92.01%)", "#1E3A8A"), widths_2[2], "center"],
         [("59 errori", "#1E3A8A"), widths_2[3], "center"],
         [("Balanced Acc: 91.75%", "#1E3A8A"), widths_2[4], "center"],
         [("F1-Score: 95.10%", "#1E3A8A"), widths_2[5], "center"]]
    ]
    end_y2 = draw_simple_table(0.5, y_t2, widths_2, headers_2, fmt_rows_2,
                               title="2. Test in Cross Dataset (Flame3Cross — FLAME3 100% Unseen: 738 immagini, Loss: 0.1767)")

    # =========================================================================
    # 3. TABELLA: TEST CROSS DATASET CON CART (FLAME3CROSS_CART)
    # =========================================================================
    y_t3 = end_y2 - 0.75
    headers_3 = ["Classe (FLAME3 100% Unseen)", "Campioni", "Predetti Corretti", "Errori", "Metrica Ottenuta", "Confronto vs Senza CART"]
    widths_3 = [3.8, 2.0, 2.4, 2.0, 2.4, 2.4]  # tot = 15.0
    fmt_rows_3 = [
        [[("Fire (1.0)", "#111827"), widths_3[0], "left"],
         ["622", widths_3[1], "center"],
         ["548 (Veri Positivi)", widths_3[2], "center"],
         ["74 (Falsi Negativi)", widths_3[3], "center"],
         ["Recall: 88.10%", widths_3[4], "center"],
         ["-4.02% rispetto a prima", widths_3[5], "center"]],
        [[("No_Fire (0.0)", "#111827"), widths_3[0], "left"],
         ["116", widths_3[1], "center"],
         [("113 (Veri Negativi)", "#065F46"), widths_3[2], "center"],
         [("SOLI 3 (Falsi Positivi)", "#065F46"), widths_3[3], "center"],
         [("Specificità: 97.41%", "#065F46"), widths_3[4], "center"],
         [("Falsi Allarmi crollati del -70%", "#065F46"), widths_3[5], "center"]],
        [[("TOTALE TEST FLAME3 (+CART)", "#065F46"), widths_3[0], "left"],
         [("738", "#065F46"), widths_3[1], "center"],
         [("661 corretti (89.57%)", "#065F46"), widths_3[2], "center"],
         [("77 errori", "#065F46"), widths_3[3], "center"],
         [("Balanced Acc: 92.76%", "#065F46"), widths_3[4], "center"],
         [("+1.01% Balanced Acc", "#065F46"), widths_3[5], "center"]]
    ]
    end_y3 = draw_simple_table(0.5, y_t3, widths_3, headers_3, fmt_rows_3,
                               title="3. Test Cross Dataset con CART (Flame3Cross_CART — FLAME3 100% Unseen: 738 immagini, Loss: 0.2508)")

    # =========================================================================
    # 4. TABELLA RIASSUNTIVA DIRETTA A CONFRONTO
    # =========================================================================
    y_t4 = end_y3 - 0.75
    headers_4 = ["Esperimento", "Test Set Valutato", "Loss", "Accuracy", "Balanced Acc", "F1-Score", "Recall (Fire)", "Specificità (No_Fire)", "Falsi Positivi"]
    widths_4 = [3.2, 2.4, 1.2, 1.3, 1.5, 1.3, 1.7, 1.8, 1.6]  # tot = 16.0 -> 15.0:
    # Ridistribuiamo su tot = 15.0
    widths_4 = [3.0, 2.2, 1.1, 1.3, 1.5, 1.2, 1.5, 1.7, 1.5]  # tot = 15.0
    fmt_rows_4 = [
        [[("1. Dataset Misto (Senza CART)", "#111827"), widths_4[0], "left"],
         ["820 img (Misto)", widths_4[1], "center"],
         ["0.0837", widths_4[2], "center"],
         ["95.12%", widths_4[3], "center"],
         ["96.93%", widths_4[4], "center"],
         ["96.84%", widths_4[5], "center"],
         ["93.84%", widths_4[6], "center"],
         ["100.00%", widths_4[7], "center"],
         ["0 / 171", widths_4[8], "center"]],
        [[("2. Cross Dataset (Flame3Cross)", "#111827"), widths_4[0], "left"],
         ["738 img (FLAME3)", widths_4[1], "center"],
         ["0.1767", widths_4[2], "center"],
         ["92.01%", widths_4[3], "center"],
         ["91.75%", widths_4[4], "center"],
         ["95.10%", widths_4[5], "center"],
         ["92.12%", widths_4[6], "center"],
         ["91.38%", widths_4[7], "center"],
         [("10 / 116", "#DC2626"), widths_4[8], "center"]],
        [[("3. Cross Dataset + CART", "#065F46"), widths_4[0], "left"],
         ["738 img (FLAME3)", widths_4[1], "center"],
         ["0.2508", widths_4[2], "center"],
         ["89.57%", widths_4[3], "center"],
         [("92.76%", "#065F46"), widths_4[4], "center"],
         ["93.44%", widths_4[5], "center"],
         ["88.10%", widths_4[6], "center"],
         [("97.41%", "#065F46"), widths_4[7], "center"],
         [("SOLI 3 / 116", "#065F46"), widths_4[8], "center"]]
    ]
    end_y4 = draw_simple_table(0.5, y_t4, widths_4, headers_4, fmt_rows_4,
                               title="4. Confronto Diretto delle Metriche di Test nei 3 Esperimenti")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor="white", edgecolor="none")
    # Copia anche come confronto_esperimenti.png
    plt.savefig("Results/confronto_esperimenti.png", dpi=200, facecolor="white", edgecolor="none")
    plt.close()
    print(f"Immagine semplice salvata in: {output_path}")

if __name__ == "__main__":
    create_simple_test_results()
