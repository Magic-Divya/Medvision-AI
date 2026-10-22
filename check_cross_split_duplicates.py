from pathlib import Path
from hashlib import sha256

root = Path("data/raw/chest_xray")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
splits = ["train", "val", "test"]

def collect(split):
    result = {}

    for path in (root / split).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        digest = sha256(path.read_bytes()).hexdigest()
        result.setdefault(digest, []).append(
            (split, path.parent.name, str(path.relative_to(root)))
        )

    return result

data = {split: collect(split) for split in splits}

for i, first_split in enumerate(splits):
    for second_split in splits[i + 1:]:
        first = data[first_split]
        second = data[second_split]
        shared = set(first) & set(second)

        print(f"\n{first_split.upper()} vs {second_split.upper()}")
        print("Shared exact image hashes:", len(shared))

        for digest in sorted(shared)[:10]:
            print("  ", first[digest][0], "<->", second[digest][0])

print("\nThis checks exact image contents, not patient identity.")
