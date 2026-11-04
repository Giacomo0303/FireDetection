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

        # Dimensione reale del vettore di feature in uscita dalla backbone
        feat_dim = getattr(
            self.rgb_backbone, "head_hidden_size", self.rgb_backbone.num_features
        )

        # 3. Normalizzazione dei singoli branch per allineare le scale (senza ReLU)
        self.rgb_norm = nn.LayerNorm(feat_dim)
        self.thermal_norm = nn.LayerNorm(feat_dim)

        # 4. Classificatore Unimodale Condiviso (per i singoli segnali RGB o Termico)
        self.unimodal_classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

        # 5. Classificatore Fuso (lavora direttamente sulla concatenazione [f_rgb, f_th])
        self.fused_classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim * 2, num_classes),
        )

    def forward(self, x):
        # Slicing del tensore (B, 4, H, W) -> RGB (B, 3, H, W) e Termico (B, 1, H, W)
        x_rgb = x[:, :3, :, :]
        x_th = x[:, 3:4, :, :]

        # Estrazione feature indipendenti e normalizzazione (B, feat_dim)
        f_rgb = self.rgb_norm(self.rgb_backbone(x_rgb))
        f_th = self.thermal_norm(self.thermal_backbone(x_th))

        # Concatenazione diretta delle feature normalizzate (B, feat_dim * 2) senza proiezione intermedia
        f_cat = torch.cat([f_rgb, f_th], dim=1)

        # Task 1: Classificazione unimodale tramite il classificatore condiviso feat_dim -> 1
        logit_rgb = self.unimodal_classifier(f_rgb).squeeze(1)
        logit_th = self.unimodal_classifier(f_th).squeeze(1)

        # Task 2: Classificazione multimodale tramite il classificatore dedicato (feat_dim * 2) -> 1
        logit_fused = self.fused_classifier(f_cat).squeeze(1)

        return {
            "fused": logit_fused,
            "rgb": logit_rgb,
            "thermal": logit_th,
        }


class SingleBranchModel(nn.Module):
    def __init__(
        self,
        mode,
        model_name="resnet50",
        pretrained=True,
        num_classes=1,
        drop_rate=0.4,
        drop_path_rate=0.2,
    ):

        super().__init__()

        self.mode = mode
        if self.mode not in ["rgb", "thermal"]:
            raise ValueError("Wrong mode, chose between 'rgb' and 'thermal'")
        elif self.mode == "rgb":
            self.backbone = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=3,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
        else:
            self.backbone = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )

        feat_dim = getattr(
            self.backbone, "head_hidden_size", self.backbone.num_features
        )

        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        if self.mode == "rgb":
            x = x[:, :3, :, :]
        else:
            x = x[:, 3:4, :, :]

        feature = self.norm(self.backbone(x))
        logits = self.classifier(self.dropout(feature))

        return logits


class SingleClassifierModel(nn.Module):
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

        # Dimensione reale del vettore di feature in uscita dalla backbone
        feat_dim = getattr(
            self.rgb_backbone, "head_hidden_size", self.rgb_backbone.num_features
        )

        self.fuse_norm = nn.LayerNorm(feat_dim)

        self.rgb_norm = nn.LayerNorm(feat_dim)
        self.thermal_norm = nn.LayerNorm(feat_dim)

        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(in_features=feat_dim, out_features=num_classes) 

    def forward(self, x):
        x_rgb = x[:, :3, :, :]
        x_thermal = x[:, 3:4, :, :]

        rgb_features = self.rgb_norm(self.rgb_backbone(x_rgb))  # [b, feat_dim]
        thermal_features = self.thermal_norm(self.thermal_backbone(x_thermal))

        fused_features = self.fuse_norm((rgb_features + thermal_features) / 2.0)

        rgb_logits = self.classifier(self.dropout(rgb_features)).squeeze(1)
        thermal_logits = self.classifier(self.dropout(thermal_features)).squeeze(1)
        fused_logits = self.classifier(self.dropout(fused_features)).squeeze(1)

        return {
            "fused": fused_logits,
            "rgb": rgb_logits,
            "thermal": thermal_logits,
        }
