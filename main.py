import logging
import os
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from src.datasets import FireClassificationDataset
from src.train import eval_model, train_model


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


def main():
    setup_logging(cfg.LOG_FILE)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    train_set = FireClassificationDataset(
        path=cfg.DATA_PATH, resolution=cfg.RESOLUTION, split="train"
    )
    val_set = FireClassificationDataset(
        path=cfg.DATA_PATH, resolution=cfg.RESOLUTION, split="val"
    )
    test_set = FireClassificationDataset(
        path=cfg.DATA_PATH, resolution=cfg.RESOLUTION, split="test"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        pin_memory=True,
        num_workers=cfg.NUM_WORKERS // 2,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.BATCH_SIZE,
        pin_memory=True,
        num_workers=cfg.NUM_WORKERS // 2,
    )

    model = timm.create_model(
        model_name=cfg.MODEL_NAME,
        pretrained=cfg.PRETRAINED,
        in_chans=cfg.IN_CHANS,
        num_classes=cfg.NUM_CLASSES,
        drop_rate=cfg.DROP_RATE,
        drop_path_rate=cfg.DROP_PATH_RATE,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
    )

    model, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=cfg.NUM_EPOCHS,
        device=device,
        scheduler=scheduler,
        patience=cfg.PATIENCE,
        save_dir=cfg.SAVE_DIR,
    )

    # Carica il miglior modello e valuta sul Test Set
    best_model_path = os.path.join(cfg.SAVE_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = eval_model(model, test_loader, loss_fn, device)
    logging.info("-" * 80)
    logging.info(
        f"[Test Set Finale] Loss: {test_metrics['loss']:.4f} - "
        f"Acc: {test_metrics['acc']:.3f} - "
        f"BalAcc: {test_metrics['bal_acc']:.3f} - "
        f"F1: {test_metrics['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
