
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.medvision.data.dataset import create_dataloaders
from src.medvision.models.densenet import build_model


def main():
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = create_dataloaders(
        dataset_root=root / "data" / "raw" / "chest_xray",
        image_size=224,
        batch_size=8,
        validation_split=0.2,
        seed=42,
        num_workers=0,
    )

    model = build_model(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(
        root / "artifacts" / "checkpoints" / "best_model.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels_all, probs_all = [], []

    with torch.inference_mode():
        for images, labels in loaders["test"]:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            labels_all.extend(labels.numpy().tolist())
            probs_all.extend(probabilities.cpu().numpy().tolist())

    y_true = np.asarray(labels_all)
    probs = np.asarray(probs_all)
    y_pred = probs.argmax(axis=1)

    print("\nCheckpoint epoch:", checkpoint.get("epoch"))
    print("Class mapping: 0=NORMAL, 1=PNEUMONIA")
    print("ROC-AUC:", round(roc_auc_score(y_true, probs[:, 1]), 4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))

    for label, name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        scores = probs[y_true == label, 1]
        print(
            f"\nActual {name}: pneumonia probability "
            f"min={scores.min():.4f}, "
            f"median={np.median(scores):.4f}, "
            f"max={scores.max():.4f}"
        )

    print("\nThreshold check (predict PNEUMONIA if probability >= threshold):")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        predictions = (probs[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(y_true, predictions, labels=[0, 1])
        accuracy = (predictions == y_true).mean()
        print(
            f"Threshold={threshold:.1f} | "
            f"accuracy={accuracy:.4f} | "
            f"matrix={cm.tolist()}"
        )


if __name__ == "__main__":
    main()
