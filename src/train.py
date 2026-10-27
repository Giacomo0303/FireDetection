import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
)

import config as cfg


def _create_branch_metrics(device):
    return {
        "acc": BinaryAccuracy().to(device),
        "bal_acc": MulticlassAccuracy(num_classes=2, average="macro").to(device),
        "f1": BinaryF1Score().to(device),
        "precision": BinaryPrecision().to(device),
        "recall": BinaryRecall().to(device),
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }


def _update_branch_metrics(m_dict, logits, y_int):
    preds = (logits > 0.0).long()
    m_dict["acc"].update(preds, y_int)
    m_dict["bal_acc"].update(preds, y_int)
    m_dict["f1"].update(preds, y_int)
    m_dict["precision"].update(preds, y_int)
    m_dict["recall"].update(preds, y_int)

    m_dict["tp"] += ((preds == 1) & (y_int == 1)).sum().item()
    m_dict["fp"] += ((preds == 1) & (y_int == 0)).sum().item()
    m_dict["tn"] += ((preds == 0) & (y_int == 0)).sum().item()
    m_dict["fn"] += ((preds == 0) & (y_int == 1)).sum().item()


def _compute_branch_metrics(m_dict, avg_loss):
    return {
        "loss": avg_loss,
        "acc": m_dict["acc"].compute().item(),
        "bal_acc": m_dict["bal_acc"].compute().item(),
        "f1": m_dict["f1"].compute().item(),
        "precision": m_dict["precision"].compute().item(),
        "recall": m_dict["recall"].compute().item(),
        "tp": m_dict["tp"],
        "fp": m_dict["fp"],
        "tn": m_dict["tn"],
        "fn": m_dict["fn"],
    }


def train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch=1):
    model.train()
    epoch_loss_tot = 0.0
    epoch_loss_fused = 0.0
    epoch_loss_rgb = 0.0
    epoch_loss_th = 0.0
    is_dual = False

    w_fused = getattr(cfg, "LOSS_WEIGHT_FUSED", 1.0)
    w_rgb = getattr(cfg, "LOSS_WEIGHT_RGB", 0.5)
    w_th = getattr(cfg, "LOSS_WEIGHT_TH", 0.5)

    pbar = tqdm(train_loader, desc=f"Epoca {epoch:02d} [Train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_float = y.float()

        optimizer.zero_grad(set_to_none=True)

        device_type = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            output = model(x)
            if isinstance(output, dict):
                is_dual = True
                l_fused = loss_fn(output["fused"], y_float)
                l_rgb = loss_fn(output["rgb"], y_float)
                l_th = loss_fn(output["thermal"], y_float)
                loss = w_fused * l_fused + w_rgb * l_rgb + w_th * l_th
            else:
                logits = output.squeeze(1)
                l_fused = loss_fn(logits, y_float)
                loss = l_fused

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss_tot += loss.item()
        epoch_loss_fused += l_fused.item()
        if is_dual:
            epoch_loss_rgb += l_rgb.item()
            epoch_loss_th += l_th.item()
            pbar.set_postfix(
                tot=f"{loss.item():.3f}",
                fus=f"{l_fused.item():.3f}",
                rgb=f"{l_rgb.item():.3f}",
                th=f"{l_th.item():.3f}",
            )
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(train_loader)
    if is_dual:
        return {
            "loss": epoch_loss_fused / n,
            "loss_tot": epoch_loss_tot / n,
            "loss_fused": epoch_loss_fused / n,
            "loss_rgb": epoch_loss_rgb / n,
            "loss_th": epoch_loss_th / n,
        }
    return epoch_loss_tot / n


def eval_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss_tot = 0.0
    val_loss_fused = 0.0
    val_loss_rgb = 0.0
    val_loss_th = 0.0
    is_dual = False

    w_fused = getattr(cfg, "LOSS_WEIGHT_FUSED", 1.0)
    w_rgb = getattr(cfg, "LOSS_WEIGHT_RGB", 0.5)
    w_th = getattr(cfg, "LOSS_WEIGHT_TH", 0.5)

    m_fused = _create_branch_metrics(device)
    m_rgb = _create_branch_metrics(device)
    m_th = _create_branch_metrics(device)

    pbar = tqdm(val_loader, desc="[Validation]", leave=False)
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_float = y.float()
            y_int = y.long()

            device_type = "cuda" if "cuda" in str(device) else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                output = model(x)
                if isinstance(output, dict):
                    is_dual = True
                    l_fused = loss_fn(output["fused"], y_float)
                    l_rgb = loss_fn(output["rgb"], y_float)
                    l_th = loss_fn(output["thermal"], y_float)
                    loss = w_fused * l_fused + w_rgb * l_rgb + w_th * l_th
                else:
                    logits = output.squeeze(1)
                    l_fused = loss_fn(logits, y_float)
                    loss = l_fused

            val_loss_tot += loss.item()
            val_loss_fused += l_fused.item()

            if is_dual:
                val_loss_rgb += l_rgb.item()
                val_loss_th += l_th.item()
                _update_branch_metrics(m_fused, output["fused"], y_int)
                _update_branch_metrics(m_rgb, output["rgb"], y_int)
                _update_branch_metrics(m_th, output["thermal"], y_int)
            else:
                _update_branch_metrics(m_fused, logits, y_int)

    n = len(val_loader)
    res = _compute_branch_metrics(m_fused, val_loss_fused / n)
    if is_dual:
        res["loss_tot"] = val_loss_tot / n
        res["fused"] = dict(res)
        res["rgb"] = _compute_branch_metrics(m_rgb, val_loss_rgb / n)
        res["thermal"] = _compute_branch_metrics(m_th, val_loss_th / n)
    return res


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    num_epochs,
    device,
    scheduler=None,
    patience=5,
    save_dir="checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = torch.amp.GradScaler(device_type)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_loss_rgb": [],
        "val_loss_th": [],
    }
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    logging.info(
        f"Avvio addestramento su {device} per {num_epochs} epoche (Patience Early Stopping: {patience})..."
    )
    logging.info("-" * 80)

    for epoch in range(1, num_epochs + 1):
        train_res = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch=epoch
        )
        val_metrics = eval_model(model, val_loader, loss_fn, device)

        if isinstance(train_res, dict):
            train_loss_fused = train_res["loss_fused"]
            history["train_loss"].append(train_loss_fused)
            history["val_loss"].append(val_metrics["loss"])
            history["val_loss_rgb"].append(val_metrics["rgb"]["loss"])
            history["val_loss_th"].append(val_metrics["thermal"]["loss"])

            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoca [{epoch:02d}/{num_epochs:02d}] | LR: {current_lr:.2e} | "
                f"Train Loss [Tot: {train_res['loss_tot']:.4f} | Fus: {train_loss_fused:.4f} | "
                f"RGB: {train_res['loss_rgb']:.4f} | TH: {train_res['loss_th']:.4f}] | "
                f"Val Fus Loss: {val_metrics['loss']:.4f} - "
                f"Acc: {val_metrics['acc']:.3f} - "
                f"BalAcc: {val_metrics['bal_acc']:.3f} "
                f"(RGB: {val_metrics['rgb']['bal_acc']:.3f} | TH: {val_metrics['thermal']['bal_acc']:.3f}) - "
                f"F1: {val_metrics['f1']:.3f} - "
                f"Rec: {val_metrics['recall']:.3f} (FN: {val_metrics['fn']})"
            )
        else:
            train_loss = train_res
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])

            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoca [{epoch:02d}/{num_epochs:02d}] | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Acc: {val_metrics['acc']:.3f} - "
                f"BalAcc: {val_metrics['bal_acc']:.3f} - "
                f"F1: {val_metrics['f1']:.3f}"
            )

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        # Controllo miglioramento sulla Validation Loss (Fused) per checkpoint ed Early Stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  --> Checkpoint salvato con Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(
                f"  --> Nessun miglioramento sulla Val Loss per {epochs_no_improve}/{patience} epoche."
            )
            if epochs_no_improve >= patience:
                logging.info(
                    f"[Early Stopping] L'addestramento si interrompe all'epoca {epoch} per mancato miglioramento sulla Val Loss."
                )
                break

    # Salva il grafico delle loss
    actual_epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(9, 5.5))
    plt.plot(actual_epochs, history["train_loss"], label="Train Loss (Fused)", linewidth=2)
    plt.plot(
        actual_epochs,
        history["val_loss"],
        label="Val Loss (Fused)",
        linestyle="--",
        linewidth=2,
    )
    if history["val_loss_rgb"]:
        plt.plot(
            actual_epochs,
            history["val_loss_rgb"],
            label="Val Loss (Solo RGB)",
            linestyle=":",
            linewidth=1.5,
            alpha=0.85,
        )
        plt.plot(
            actual_epochs,
            history["val_loss_th"],
            label="Val Loss (Solo Termico)",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.85,
        )
    plt.xlabel("Epoca", fontsize=12)
    plt.ylabel("Loss (BCE)", fontsize=12)
    plt.title("Andamento della Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logging.info(f"Grafico delle loss salvato in: {plot_path}")

    return model, history
