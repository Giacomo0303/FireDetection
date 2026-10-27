import timm
import torch
import torch.nn as nn


class DualBranchRGBTModel(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        num_classes=1,
        drop_rate=0.4,
        drop_path_rate=0.2,
    ):
        super().__init__()

        # 1. Branch RGB (3 canali, pre-addestrato su ImageNet, senza classificatore finale)
        self.rgb_backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        # 2. Branch Termico (1 canale, pesi conv1 adattati da ImageNet, senza classificatore)
        self.thermal_backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        feat_dim = self.rgb_backbone.num_features

        # 3. Normalizzazione dei singoli branch per allineare le scale nello spazio 2048 (senza ReLU)
        self.rgb_norm = nn.LayerNorm(feat_dim)
        self.thermal_norm = nn.LayerNorm(feat_dim)

        # 4. Classificatore Unimodale Condiviso (per i singoli segnali RGB o Termico a 2048 dim)
        self.unimodal_classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

        # 5. Classificatore Fuso (lavora direttamente sulla concatenazione [f_rgb, f_th] a 4096 dim)
        self.fused_classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim * 2, num_classes),
        )

    def forward(self, x):
        # Slicing del tensore (B, 4, H, W) -> RGB (B, 3, H, W) e Termico (B, 1, H, W)
        x_rgb = x[:, :3, :, :]
        x_th = x[:, 3:4, :, :]

        # Estrazione feature indipendenti e normalizzazione (B, 2048)
        f_rgb = self.rgb_norm(self.rgb_backbone(x_rgb))
        f_th = self.thermal_norm(self.thermal_backbone(x_th))

        # Concatenazione diretta delle feature normalizzate (B, 4096) senza proiezione intermedia
        f_cat = torch.cat([f_rgb, f_th], dim=1)

        # Task 1: Classificazione unimodale tramite il classificatore condiviso 2048 -> 1
        logit_rgb = self.unimodal_classifier(f_rgb).squeeze(1)
        logit_th = self.unimodal_classifier(f_th).squeeze(1)

        # Task 2: Classificazione multimodale tramite il classificatore dedicato 4096 -> 1
        logit_fused = self.fused_classifier(f_cat).squeeze(1)

        return {
            "fused": logit_fused,
            "rgb": logit_rgb,
            "thermal": logit_th,
        }
