
"""Compare classification thresholds on validation and test data."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from src.medvision.data.dataset import create_dataloaders
from src.medvision.models.densenet import build_model


def main():
    root = Path(__file__).resolve().parent
    dataset_root = root / "data" / "raw" / "chest_xray"
    checkpoint_path = root / "artifacts" / "checkpoints" / "best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = create_dataloaders(
        dataset_root=dataset_root,
        image_size=224,
        batch_size=8,
        validation_split=0.2,
        seed=42,
        num_workers=0,
    )

    model = build_model(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = {}

    with torch.inference_mode():
        for split in ("val", "test"):
            labels_all = []
            probabilities_all = []

            for images, labels in loaders[split]:
                logits = model(images.to(device))
                probabilities = torch.softmax(logits, dim=1)[:, 1]

                labels_all.extend(labels.numpy().tolist())
                probabilities_all.extend(probabilities.cpu().numpy().tolist())

            results[split] = (
                np.asarray(labels_all),
                np.asarray(probabilities_all),
            )

    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print("Class 0 = NORMAL; class 1 = PNEUMONIA")
    print("Threshold means predict PNEUMONIA when its probability >= threshold.")

    for split, (y_true, p_pneumonia) in results.items():
        print(f"\n{'=' * 18} {split.upper()} {'=' * 18}")

        auc = roc_auc_score(y_true, p_pneumonia)
        print(f"ROC-AUC: {auc:.4f}")

        for threshold in (0.5, 0.6, 0.7, 0.8):
            y_pred = (p_pneumonia >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            tn, fp, fn, tp = cm.ravel()
            normal_recall = tn / (tn + fp) if (tn + fp) else 0.0
            pneumonia_recall = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = normal_recall

            print(
                f"\nThreshold: {threshold:.1f}"
                f"\n  Accuracy:           {accuracy_score(y_true, y_pred):.4f}"
                f"\n  NORMAL recall:      {normal_recall:.4f}"
                f"\n  PNEUMONIA recall:   {pneumonia_recall:.4f}"
                f"\n  Specificity:        {specificity:.4f}"
                f"\n  TN={tn}, FP={fp}, FN={fn}, TP={tp}"
            )


if __name__ == "__main__":
    main()
