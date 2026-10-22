
from pathlib import Path

root = Path("data/raw/chest_xray")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "test"]:
    print(f"\n{split.upper()} DATASET")

    for class_name in ["NORMAL", "PNEUMONIA"]:
        folder = root / split / class_name

        if not folder.exists():
            print(f"{class_name}: Folder not found!")
            continue

        count = sum(
            1
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

        print(f"{class_name}: {count} images")
