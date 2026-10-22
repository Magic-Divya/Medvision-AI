
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageStat

ROOT = Path("data/raw/chest_xray")
SPLITS = ["train", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    for split in SPLITS:
        print(f"\n{'=' * 55}")
        print(f"{split.upper()} IMAGE ANALYSIS")
        print("=" * 55)

        for class_name in CLASSES:
            folder = ROOT / split / class_name
            files = [
                p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            ]

            dimensions = Counter()
            formats = Counter()
            brightness_values = []
            contrast_values = []

            for path in files:
                try:
                    with Image.open(path) as image:
                        dimensions[image.size] += 1
                        formats[image.format or path.suffix.lower()] += 1

                        grayscale = image.convert("L")
                        stats = ImageStat.Stat(grayscale)
                        brightness_values.append(stats.mean[0])
                        contrast_values.append(stats.stddev[0])

                except Exception as exc:
                    print(f"Could not read {path}: {exc}")

            print(f"\nClass: {class_name}")
            print(f"Images: {len(files)}")
            print(f"Most common dimensions: {dimensions.most_common(5)}")
            print(f"File formats: {dict(formats)}")

            if brightness_values:
                print(
                    "Mean brightness (0-255): "
                    f"{np.mean(brightness_values):.2f}"
                )
                print(
                    "Mean image contrast: "
                    f"{np.mean(contrast_values):.2f}"
                )


if __name__ == "__main__":
    main()
