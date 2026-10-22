from pathlib import Path
from hashlib import sha256

root = Path("data/raw/chest_xray")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        folder = root / split / cls
        seen = {}
        duplicates = []

        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue

            digest = sha256(path.read_bytes()).hexdigest()

            if digest in seen:
                duplicates.append((seen[digest], path.name))
            else:
                seen[digest] = path.name

        print(f"\n{split}/{cls}: {len(duplicates)} duplicate file(s)")
        for first, duplicate in duplicates[:10]:
            print(f"  {duplicate} matches {first}")
