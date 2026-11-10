from src.models import SingleClassifierModel
import config as cfg
import torch
import pnnx
import argparse


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_rgb, x_thermal):
        f_rgb = self.model.rgb_norm(self.model.rgb_backbone(x_rgb))
        f_thermal = self.model.thermal_norm(self.model.thermal_backbone(x_thermal))
        f_fused = self.model.fuse_norm(torch.max(f_rgb, f_thermal))

        logit_rgb = self.model.classifier(f_rgb)
        logit_thermal = self.model.classifier(f_thermal)
        logit_fused = self.model.classifier(f_fused)

        return logit_fused, logit_rgb, logit_thermal


def main(path):
    model = SingleClassifierModel(
        model_name=cfg.MODEL_NAME, fusion_mode=cfg.FUSION_MODE
    )
    state_dict = torch.load(f=path)
    model.load_state_dict(state_dict)
    wrapped_model = ModelWrapper(model).eval()

    x_rgb = torch.randn(1, 3, 256, 256)
    x_thermal = torch.randn(1, 1, 256, 256)

    pnnx.export(
        model=wrapped_model,
        ptpath=f"Exported_models/{cfg.MODEL_NAME}_exported.pt",
        inputs=(x_rgb, x_thermal),
        fp16=True,
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--path", type=str, help="Percorso al file pth")
    args = arg_parser.parse_args()

    main(path=args.path)
