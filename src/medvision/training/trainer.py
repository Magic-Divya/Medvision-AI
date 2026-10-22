
"""Training utilities for MedVision AI."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm



def calculate_class_weights(train_loader, device):
    """Calculate class weights from the training dataset only."""
    labels = np.asarray(train_loader.dataset.labels, dtype=np.int64)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1], dtype=np.int64),
        y=labels,
    )

    return torch.tensor(weights, dtype=torch.float32, device=device)



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.inference_mode()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def fit(
    model,
    loaders,
    device,
    epochs=3,
    learning_rate=0.0001,
    checkpoint_path="artifacts/checkpoints/best_model.pth",
):
    """Train the model and save the best validation-loss checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    class_weights = calculate_class_weights(loaders["train"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0001,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, loaders["val"], criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_acc:.4f}"
        )
        print(
            f"Val loss:   {val_loss:.4f} | "
            f"Val accuracy:   {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "class_names": ["NORMAL", "PNEUMONIA"],
                },
                checkpoint,
            )
            print(f"Saved best checkpoint: {checkpoint}")

    return history
