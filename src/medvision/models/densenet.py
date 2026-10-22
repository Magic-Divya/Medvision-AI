
"""DenseNet121 model for MedVision AI."""

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build DenseNet121 for two-class chest X-ray classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet-pretrained weights when True.

    Returns:
        A DenseNet121 model producing raw class logits.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)

    input_features = model.classifier.in_features
    model.classifier = nn.Linear(input_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable model parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set pretrained=True for transfer learning.
    model = build_model(num_classes=2, pretrained=True).to(device)
    model.eval()

    # A small batch keeps this initial CPU test manageable.
    sample = torch.randn(2, 3, 224, 224, device=device)

    with torch.inference_mode():
        logits = model(sample)

    print("Model: DenseNet121")
    print("Device:", device)
    print("Output shape:", tuple(logits.shape))
    print("Parameters:", count_parameters(model))
    print("Model test passed.")
