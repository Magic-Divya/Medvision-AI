
"""Evaluate the trained MedVision-AI DenseNet121 model."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.medvision.data.dataset import create_dataloaders
from src.medvision.models.densenet import build_model


def main():
    root = Path(__file__).resolve().parent
    dataset_root = root / "data" / "raw" / "chest_xray"
    checkpoint_path = root / "artifacts" / "checkpoints" / "best_model.pth"
    results_dir = root / "reports" / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MedVision-AI: Test Evaluation")
    print("Device:", device)

    # Recreate the same data split and image preprocessing used in training.
    loaders = create_dataloaders(
        dataset_root=dataset_root,
        image_size=224,
        batch_size=8,
        validation_split=0.2,
        seed=42,
        num_workers=0,
    )
    test_loader = loaders["test"]
    print("Test images:", len(test_loader.dataset))

    # Load the saved best model without downloading pretrained weights.
    model = build_model(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    y_prob = np.asarray(all_probabilities)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except ValueError:
        auc = None

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_images": int(len(y_true)),
        "test_loss": float(total_loss / len(y_true)),
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "class_names": ["NORMAL", "PNEUMONIA"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print("\n=== TEST RESULTS ===")
    print(f"Test loss:          {metrics['test_loss']:.4f}")
    print(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall:    {recall:.4f}")
    print(f"Weighted F1-score:  {f1:.4f}")
    print("ROC-AUC:", f"{auc:.4f}" if auc is not None else "Unavailable")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)

    # Save machine-readable results.
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save a labelled confusion-matrix plot.
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="MedVision-AI Test Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    figure_path = results_dir / "confusion_matrix.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved evaluation metrics:", metrics_path)
    print("Saved confusion matrix:", figure_path)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
