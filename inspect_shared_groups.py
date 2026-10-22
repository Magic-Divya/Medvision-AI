from pathlib import Path
import re

ROOT = Path(r"data/raw/chest_xray")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def group_id(path):
    match = re.match(r"^(person\d+)_", path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else None

def grouped_files(folder):
    groups = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            group = group_id(path)
            if group:
                groups.setdefault(group, []).append(path.name)
    return groups

train = grouped_files(ROOT / "train")
test = grouped_files(ROOT / "test")
shared = sorted(set(train) & set(test))

print("Shared person-number groups:", len(shared))
for group in shared[:10]:
    print(f"\n{group}")
    print("  TRAIN:", train[group][:5])
    print("  TEST: ", test[group][:5])
