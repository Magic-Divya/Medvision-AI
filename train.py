
"""Train the MedVision AI chest X-ray classifier."""

from pathlib import Path

import torch

from src.medvision.data.dataset import create_dataloaders
from src.medvision.models.densenet import build_model
from src.medvision.training.trainer import fit


def main():
    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "data" / "raw" / "chest_xray"
    checkpoint_path = (
        project_root / "artifacts" / "checkpoints" / "best_model.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MedVision AI training")
    print("Device:", device)

    loaders = create_dataloaders(
        dataset_root=dataset_root,
        image_size=224,
        batch_size=8,
        validation_split=0.2,
        seed=42,
        num_workers=0,
    )

    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} images")

    model = build_model(num_classes=2, pretrained=True).to(device)

    history = fit(
        model=model,
        loaders=loaders,
        device=device,
        epochs=3,
        learning_rate=0.0001,
        checkpoint_path=checkpoint_path,
    )

    print("\nTraining finished.")
    print("Best checkpoint:", checkpoint_path)
    print("Training history:", history)


if __name__ == "__main__":
    main()
