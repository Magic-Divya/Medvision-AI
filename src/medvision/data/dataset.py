
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def collect_images(split_dir: Path):
    """Collect images and integer labels from class subfolders."""
    paths, labels = [], []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
                labels.append(label)

    if not paths:
        raise RuntimeError(f"No images found in {split_dir}")

    return paths, labels


def build_transforms(image_size=224):
    """Use augmentation only for training images."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return train_transform, eval_transform


class ChestXRayDataset(Dataset):
    """Dataset for NORMAL and PNEUMONIA chest X-ray images."""

    def __init__(self, image_paths, labels, transform=None):
        if len(image_paths) != len(labels):
            raise ValueError("Image paths and labels must have equal lengths.")

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[index], dtype=torch.long)


def create_dataloaders(
    dataset_root,
    image_size=224,
    batch_size=16,
    validation_split=0.2,
    seed=42,
    num_workers=0,
):
    """Split training data reproducibly; reserve the official test set."""
    root = Path(dataset_root)

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")

    train_paths, train_labels = collect_images(root / "train")
    test_paths, test_labels = collect_images(root / "test")

    train_indices, val_indices = train_test_split(
        list(range(len(train_paths))),
        test_size=validation_split,
        random_state=seed,
        stratify=train_labels,
    )

    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = ChestXRayDataset(
        [train_paths[i] for i in train_indices],
        [train_labels[i] for i in train_indices],
        train_transform,
    )
    val_dataset = ChestXRayDataset(
        [train_paths[i] for i in val_indices],
        [train_labels[i] for i in val_indices],
        eval_transform,
    )
    test_dataset = ChestXRayDataset(
        test_paths, test_labels, eval_transform
    )

    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return {
        "train": DataLoader(train_dataset, shuffle=True, **options),
        "val": DataLoader(val_dataset, shuffle=False, **options),
        "test": DataLoader(test_dataset, shuffle=False, **options),
    }
