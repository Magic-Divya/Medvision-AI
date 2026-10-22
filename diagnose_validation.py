
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score

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

    checkpoint_path = (
        root / "artifacts" / "checkpoints" / "best_model.pth"
    )

    model = build_model(num_classes=2, pretrained=False).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.inference_mode():
        for images, labels in loaders["val"]:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy().tolist()
            )

    y_true = np.asarray(all_labels)
    pneumonia_probs = np.asarray(all_probabilities)

    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Validation images:", len(y_true))
    print("Class mapping: 0=NORMAL, 1=PNEUMONIA")
    print("\nValidation threshold comparison:")

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = (pneumonia_probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        accuracy = accuracy_score(y_true, y_pred)
        normal_recall = tn / (tn + fp) if tn + fp else 0
        pneumonia_recall = tp / (tp + fn) if tp + fn else 0

        print(
            f"Threshold={threshold:.1f} | "
            f"Accuracy={accuracy:.4f} | "
            f"Normal recall={normal_recall:.4f} | "
            f"Pneumonia recall={pneumonia_recall:.4f} | "
            f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        )


if __name__ == "__main__":
    main()
