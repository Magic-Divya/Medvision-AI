
from pathlib import Path
from collections import defaultdict
import re

from sklearn.model_selection import train_test_split

ROOT = Path("data/raw/chest_xray")
CLASSES = ["NORMAL", "PNEUMONIA"]


def possible_patient_id(path):
    """Extract a candidate grouping key from common dataset filenames."""
    name = path.stem

    # Example: IM-0093-0001 -> IM-0093
    # Example: NORMAL2-IM-0141-0001 -> NORMAL2-IM-0141
    if re.match(r"^(NORMAL2?-)?IM-\d+-\d+$", name, re.IGNORECASE):
        return re.sub(r"-\d+$", "", name)

    # Example: person1_bacteria_1 -> person1
    match = re.match(r"^(person\d+)_", name, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Unknown filename convention: use full filename stem.
    return name


def collect(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def main():
    for class_name in CLASSES:
        paths = collect(ROOT / "train" / class_name)

        train_indices, val_indices = train_test_split(
            list(range(len(paths))),
            test_size=0.2,
            random_state=42,
            stratify=[class_name] * len(paths),
        )

        train_ids = {
            possible_patient_id(paths[i]) for i in train_indices
        }
        val_ids = {
            possible_patient_id(paths[i]) for i in val_indices
        }

        overlap = train_ids & val_ids

        print(f"\n=== {class_name} ===")
        print("Training images:", len(train_indices))
        print("Validation images:", len(val_indices))
        print("Unique candidate IDs in training:", len(train_ids))
        print("Unique candidate IDs in validation:", len(val_ids))
        print("Candidate IDs shared across splits:", len(overlap))

        if overlap:
            print("Examples:", sorted(overlap)[:15])


if __name__ == "__main__":
    main()
