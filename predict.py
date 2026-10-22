"""Image prediction utilities for MedVision-AI."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.medvision.models.densenet import build_model, CLASS_NAMES


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "artifacts" / "checkpoints" / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def load_model():
    """Load the trained DenseNet121 checkpoint."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = build_model(num_classes=2, pretrained=False).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict_image(image: Image.Image, model):
    """Return the predicted class and class probabilities."""
    if image is None:
        raise ValueError("Please provide an image.")

    image = image.convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i].item())
            for i in range(len(CLASS_NAMES))
        },
    }
