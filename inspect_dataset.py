from pathlib import Path
from collections import Counter
from PIL import Image

base = Path("data/raw/chest_xray")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ("train", "val", "test"):
    print(f"\n--- ROOT COPY: {split} ---")
    for label in ("NORMAL", "PNEUMONIA"):
        folder = base / split / label
        if not folder.is_dir():
            print(f"{label}: MISSING")
            continue

        files = [p for p in folder.rglob("*")
                 if p.is_file() and p.suffix.lower() in extensions]
        valid = 0
        corrupt = []

        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()
                valid += 1
            except Exception:
                corrupt.append(path.name)

        print(f"{label}: files={len(files)}, valid={valid}, corrupt={len(corrupt)}")

    print(f"\n--- NESTED COPY: {split} ---")
    for label in ("NORMAL", "PNEUMONIA"):
        folder = base / "chest_xray" / split / label
        if not folder.is_dir():
            print(f"{label}: MISSING")
            continue

        files = [p for p in folder.rglob("*")
                 if p.is_file() and p.suffix.lower() in extensions]
        print(f"{label}: files={len(files)}")

print("\nInspection complete. No files were changed.")
