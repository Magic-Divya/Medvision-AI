from pathlib import Path

root = Path("data/raw/chest_xray")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")

    for class_name in ["NORMAL", "PNEUMONIA"]:
        folder = root / split / class_name

        if not folder.exists():
            print(f"  {class_name}: folder not found")
            continue

        count = sum(
            1 for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        print(f"  {class_name}: {count}")

nested = root / "chest_xray"
print("\nNested chest_xray folder exists:", nested.exists())
