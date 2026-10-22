
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/raw/chest_xray")
OUTPUT = Path("reports/dataset_comparison")
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

groups = [
    ("Training NORMAL", ROOT / "train" / "NORMAL"),
    ("Test NORMAL", ROOT / "test" / "NORMAL"),
    ("Training PNEUMONIA", ROOT / "train" / "PNEUMONIA"),
    ("Test PNEUMONIA", ROOT / "test" / "PNEUMONIA"),
]

fig, axes = plt.subplots(4, 5, figsize=(15, 12))

for row, (title, folder) in enumerate(groups):
    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    samples = random.sample(paths, min(5, len(paths)))

    for col, ax in enumerate(axes[row]):
        ax.axis("off")

        if col < len(samples):
            with Image.open(samples[col]) as image:
                ax.imshow(image.convert("RGB"))

            if col == 0:
                ax.set_ylabel(title, fontsize=10)

fig.suptitle(
    "MedVision-AI: Random Training vs Test Samples",
    fontsize=15,
)
fig.tight_layout()

output_path = OUTPUT / "train_test_samples.png"
fig.savefig(output_path, dpi=130, bbox_inches="tight")
plt.close(fig)

print("Saved comparison image:", output_path.resolve())
