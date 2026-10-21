import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, MulticlassAccuracy


def train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch=1):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoca {epoch:02d} [Train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        device_type = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            output = model(x).squeeze(1)
            loss = loss_fn(output, y.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(train_loader)


def eval_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0

    acc_metric = BinaryAccuracy().to(device)
    bal_acc_metric = MulticlassAccuracy(num_classes=2, average="macro").to(device)
    f1_metric = BinaryF1Score().to(device)

    pbar = tqdm(val_loader, desc="[Validation]", leave=False)
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            device_type = "cuda" if "cuda" in str(device) else "cpu"
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                output = model(x).squeeze(1)
                loss = loss_fn(output, y.float())

            val_loss += loss.item()
            preds = (output > 0.0).long()
            y_int = y.long()

            acc_metric.update(preds, y_int)
            bal_acc_metric.update(preds, y_int)
            f1_metric.update(preds, y_int)

    return {
        "loss": val_loss / len(val_loader),
        "acc": acc_metric.compute().item(),
        "bal_acc": bal_acc_metric.compute().item(),
        "f1": f1_metric.compute().item(),
    }


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

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    logging.info(f"Avvio addestramento su {device} per {num_epochs} epoche (Patience Early Stopping: {patience})...")
    logging.info("-" * 80)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch=epoch)
        val_metrics = eval_model(model, val_loader, loss_fn, device)

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

        # Controllo miglioramento sulla Validation Loss per checkpoint ed Early Stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  --> Checkpoint salvato con Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logging.info(f"  --> Nessun miglioramento sulla Val Loss per {epochs_no_improve}/{patience} epoche.")
            if epochs_no_improve >= patience:
                logging.info(f"[Early Stopping] L'addestramento si interrompe all'epoca {epoch} per mancato miglioramento sulla Val Loss.")
                break

    # Salva il grafico delle loss (usa la lunghezza effettiva delle epoche eseguite)
    actual_epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(actual_epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(actual_epochs, history["val_loss"], label="Val Loss", linestyle="--", linewidth=2)
    plt.xlabel("Epoca", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Andamento della Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logging.info(f"Grafico delle loss salvato in: {plot_path}")

    return model, history
