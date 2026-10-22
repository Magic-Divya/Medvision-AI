from pathlib import Path
import re

ROOT = Path(r"data/raw/chest_xray")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def group_id(path):
    name = path.stem

    match = re.match(r"^(person\d+)_", name, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    match = re.match(
        r"^((?:NORMAL2?-)?IM-\d+)-\d+$",
        name,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    return name.lower()

def collect_groups(folder):
    groups = set()
    count = 0

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            count += 1
            groups.add(group_id(path))

    return count, groups

train_count, train_groups = collect_groups(ROOT / "train")
test_count, test_groups = collect_groups(ROOT / "test")

overlap = train_groups & test_groups

print("Official training images:", train_count)
print("Official test images:", test_count)
print("Training candidate groups:", len(train_groups))
print("Test candidate groups:", len(test_groups))
print("Shared candidate groups:", len(overlap))
print("Examples:", sorted(overlap)[:20])
print()
print("NOTE: Filename-based groups are only a heuristic,")
print("not proof of patient identity.")
