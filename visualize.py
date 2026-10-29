from src.models import DualBranchRGBTModel
from src.datasets import FireClassificationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from tqdm import tqdm
import config as cfg
import torch
import argparse


def main(checkpoint_path):
    model = DualBranchRGBTModel(model_name=cfg.MODEL_NAME, pretrained=False)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    rbg_backbone = model.rgb_backbone.to("cuda")
    thermal_backbone = model.thermal_backbone.to("cuda")
    rgb_norm = model.rgb_norm.to("cuda")
    thermal_norm = model.thermal_norm.to("cuda")

    test_set = FireClassificationDataset(path=cfg.DATA_PATH, split="test")
    test_loader = DataLoader(
        dataset=test_set, shuffle=False, batch_size=cfg.BATCH_SIZE * 2
    )

    rgb_features = []
    thermal_features = []
    labels = []

    for x, y in tqdm(test_loader):
        x = x.to("cuda")
        x_rgb = x[:, 0:3, :, :]
        x_thermal = x[:, 3:4, :, :]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            rgb_features.append(rgb_norm(rbg_backbone(x_rgb)).to("cpu"))
            thermal_features.append(thermal_norm(thermal_backbone(x_thermal)).to("cpu"))
        labels.append(y)

    rgb_features = torch.cat(rgb_features, dim=0)
    thermal_features = torch.cat(thermal_features, dim=0)
    fused_features = torch.cat([rgb_features, thermal_features], dim=1)

    labels = torch.cat(labels, dim=0)

    tsne = TSNE()
    rgb_tsne = tsne.fit_transform(rgb_features)
    thermal_tsne = tsne.fit_transform(thermal_features)
    fused_tsne = tsne.fit_transform(fused_features)

    umap = UMAP()
    rgb_umap = umap.fit_transform(rgb_features)
    thermal_umap = umap.fit_transform(thermal_features)
    fused_umap = umap.fit_transform(fused_features)

    # Griglia 2x3: 2 righe (t-SNE, UMAP) x 3 colonne (RGB, Termico, Fused)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 15), dpi=250)

    # --- RIGA 1: t-SNE ---
    # Subplot 1: Feature RGB
    for lab, color, name in [(1, "#EF4444", "Fire"), (0, "#3B82F6", "No Fire")]:
        mask = labels == lab
        ax1.scatter(
            rgb_tsne[mask, 0],
            rgb_tsne[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=25,
            edgecolors="none",
        )
    ax1.set_title("t-SNE: Feature RGB (2048-d)", fontsize=13, fontweight="bold")
    ax1.legend(title="Classe")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Feature Termiche
    for lab, color, name in [(1, "#EF4444", "Fire"), (0, "#3B82F6", "No Fire")]:
        mask = labels == lab
        ax2.scatter(
            thermal_tsne[mask, 0],
            thermal_tsne[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=25,
            edgecolors="none",
        )
    ax2.set_title("t-SNE: Feature Termiche (2048-d)", fontsize=13, fontweight="bold")
    ax2.legend(title="Classe")
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Subplot 3: Feature Fuse
    for lab, color, name in [(1, "#EF4444", "Fire"), (0, "#3B82F6", "No Fire")]:
        mask = labels == lab
        ax3.scatter(
            fused_tsne[mask, 0],
            fused_tsne[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=25,
            edgecolors="none",
        )
    ax3.set_title("t-SNE: Feature Fuse (4096-d)", fontsize=13, fontweight="bold")
    ax3.legend(title="Classe")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # --- RIGA 2: UMAP ---
    # Subplot 4: Feature RGB
    for lab, color, name in [(1, "#EF4444", "Fire"), (0, "#3B82F6", "No Fire")]:
        mask = labels == lab
        ax4.scatter(
            rgb_umap[mask, 0],
            rgb_umap[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=25,
            edgecolors="none",
        )
    ax4.set_title("UMAP: Feature RGB (2048-d)", fontsize=13, fontweight="bold")
    ax4.legend(title="Classe")
    ax4.grid(True, linestyle=":", alpha=0.5)

    # Subplot 5: Feature Termiche
    for lab, color, name in [(1, "#EF4444", "Fire"), (0, "#3B82F6", "No Fire")]:
        mask = labels == lab
        ax5.scatter(
            thermal_umap[mask, 0],
            thermal_umap[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=25,
            edgecolors="none",
        )
    ax5.set_title("UMAP: Feature Termiche (2048-d)", fontsize=13, fontweight="bold")
    ax5.legend(title="Classe")
    ax5.grid(True, linestyle=":", alpha=0.5)

    # Subplot 6: Feature Fuse
    for lab, color, name in [(1, "#EF4444", "Fire"), (0, "#3B82F6", "No Fire")]:
        mask = labels == lab
        ax6.scatter(
            fused_umap[mask, 0],
            fused_umap[mask, 1],
            c=color,
            label=name,
            alpha=0.7,
            s=25,
            edgecolors="none",
        )
    ax6.set_title("UMAP: Feature Fuse (4096-d)", fontsize=13, fontweight="bold")
    ax6.legend(title="Classe")
    ax6.grid(True, linestyle=":", alpha=0.5)

    # Spaziatura orizzontale maggiorata tra le colonne per migliorare la separazione visiva
    plt.tight_layout(w_pad=4.0, h_pad=2.5)
    plt.savefig("visualization.png")
    plt.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Visualizzazione delle feature")
    argparser.add_argument(
        "--checkpoint",
        type=str,
        default="Results/DualBranch_2Clf_ResNet50_CART/best_model.pth",
        help="Percorso al file .pth",
    )

    args = argparser.parse_args()

    main(checkpoint_path=args.checkpoint)
