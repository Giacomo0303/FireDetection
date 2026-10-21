import os

import pandas as pd
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class FireClassificationDataset(Dataset):
    def __init__(self, path, resolution=256, split="train"):
        super().__init__()

        self.split = split
        if split == "train":
            self.data = pd.read_csv(os.path.join(path, "train_set.csv"))
        elif split == "val":
            self.data = pd.read_csv(os.path.join(path, "val_set.csv"))
        elif split == "test":
            self.data = pd.read_csv(os.path.join(path, "test_set.csv"))
        else:
            raise ValueError(
                f"Split '{split}' non valido. Usa 'train', 'val' o 'test'."
            )

        self.resolution = resolution

        self.to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]
        )

        if self.split == "train":
            self.transforms = T.Compose(
                [
                    T.Resize((self.resolution, self.resolution)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomRotation(degrees=15),
                    self.normalize,
                ]
            )
        else:
            self.transforms = T.Compose(
                [
                    T.Resize((self.resolution, self.resolution), antialias=True),
                    self.normalize,
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_info = self.data.iloc[index]
        rgb_path, thermal_path, label = (
            sample_info["rgb_path"],
            sample_info["thermal_path"],
            sample_info["label"],
        )

        raw_rgb = Image.open(rgb_path).convert("RGB")
        raw_thermal = Image.open(thermal_path).convert("L")

        rgb_tensor = self.to_tensor(raw_rgb)
        thermal_tensor = self.to_tensor(raw_thermal)

        rgbt_tensor = torch.cat([rgb_tensor, thermal_tensor], dim=0)

        rgbt_tensor = self.transforms(rgbt_tensor)

        return rgbt_tensor, torch.tensor(label, dtype=torch.float32)
