
"""Inspect misclassified MedVision-AI test images."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.medvision.data.dataset import create_dataloaders
from src.medvision.models.densenet import build_model

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def main():
    root = Path(__file__).resolve().parent
    dataset_root = root / "data" / "raw" / "chest_xray"
    checkpoint_path = root / "artifacts" / "checkpoints" / "best_model.pth"
    output_dir = root / "reports" / "misclassified"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    loaders = create_dataloaders(
        dataset_root=dataset_root,
        image_size=224,
        batch_size=8,
        validation_split=0.2,
        seed=42,
        num_workers=0,
    )
    test_dataset = loaders["test"].dataset

    model = build_model(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    errors = []
    correct_count = 0

    with torch.inference_mode():
        for index in range(len(test_dataset)):
            image, label_tensor = test_dataset[index]
            label = int(label_tensor.item())

            logits = model(image.unsqueeze(0).to(device))
            probabilities = F.softmax(logits, dim=1)[0].cpu()
            prediction = int(probabilities.argmax().item())
            confidence = float(probabilities[prediction].item())

            if prediction == label:
                correct_count += 1
                continue

            errors.append({
                "index": index,
                "path": str(test_dataset.image_paths[index]),
                "actual": CLASS_NAMES[label],
                "predicted": CLASS_NAMES[prediction],
                "predicted_confidence": confidence,
                "normal_probability": float(probabilities[0].item()),
                "pneumonia_probability": float(probabilities[1].item()),
            })

    csv_path = output_dir / "misclassified_images.csv"
    fields = [
        "index", "path", "actual", "predicted",
        "predicted_confidence", "normal_probability",
        "pneumonia_probability",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    # Focus first on NORMAL images predicted as PNEUMONIA.
    false_positives = [
        item for item in errors
        if item["actual"] == "NORMAL"
        and item["predicted"] == "PNEUMONIA"
    ]

    print("\nTest images:", len(test_dataset))
    print("Correct predictions:", correct_count)
    print("Misclassified images:", len(errors))
    print("NORMAL predicted as PNEUMONIA:", len(false_positives))
    print("PNEUMONIA predicted as NORMAL:", sum(
        item["actual"] == "PNEUMONIA"
        and item["predicted"] == "NORMAL"
        for item in errors
    ))

    if false_positives:
        # Show up to 20 cases, prioritizing highest pneumonia confidence.
        selected = sorted(
            false_positives,
            key=lambda item: item["pneumonia_probability"],
            reverse=True,
        )[:20]

        columns = 4
        rows = (len(selected) + columns - 1) // columns
        fig, axes = plt.subplots(
            rows, columns, figsize=(14, 3.8 * rows), squeeze=False
        )

        for ax in axes.flat:
            ax.axis("off")

        for ax, item in zip(axes.flat, selected):
            # Display the original image, not its normalized tensor.
            from PIL import Image
            with Image.open(item["path"]) as original:
                ax.imshow(original.convert("RGB"))
            ax.set_title(
                f"Actual: NORMAL\n"
                f"Predicted: PNEUMONIA\n"
                f"Pneumonia probability: "
                f"{item['pneumonia_probability']:.1%}",
                fontsize=9,
            )
            ax.axis("off")

        fig.suptitle(
            "False Positives: NORMAL Test Images Predicted as PNEUMONIA",
            fontsize=13,
        )
        fig.tight_layout()
        image_path = output_dir / "normal_false_positives.png"
        fig.savefig(image_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("Contact sheet:", image_path)

    print("CSV report:", csv_path)
    print("\nInspection complete.")


if __name__ == "__main__":
    main()
