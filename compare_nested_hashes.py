from pathlib import Path
from hashlib import sha256

outer = Path("data/raw/chest_xray")
nested = outer / "chest_xray"
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def image_hashes(folder):
    hashes = set()
    if not folder.exists():
        return hashes

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            digest = sha256(path.read_bytes()).hexdigest()
            hashes.add(digest)

    return hashes

for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        a = image_hashes(outer / split / cls)
        b = image_hashes(nested / split / cls)

        print(f"{split}/{cls}:")
        print(f"  Outer unique hashes:  {len(a)}")
        print(f"  Nested unique hashes: {len(b)}")
        print(f"  Exact hashes in both: {len(a & b)}")
        print(f"  Identical hash sets:  {a == b}")
