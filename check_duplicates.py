
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path("data/raw/chest_xray")
SPLITS = ["train", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    hashes_by_split = {
        split: defaultdict(list)
        for split in SPLITS
    }

    for split in SPLITS:
        for class_name in ["NORMAL", "PNEUMONIA"]:
            folder = ROOT / split / class_name

            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    file_hash = sha256_file(path)
                    hashes_by_split[split][file_hash].append(str(path))

    train_hashes = set(hashes_by_split["train"])
    test_hashes = set(hashes_by_split["test"])

    common_hashes = train_hashes & test_hashes

    print("Unique training image hashes:", len(train_hashes))
    print("Unique test image hashes:", len(test_hashes))
    print("Exact duplicate hashes across train and test:", len(common_hashes))

    for file_hash in list(common_hashes)[:10]:
        print("\nMatching files:")
        print("TRAIN:", hashes_by_split["train"][file_hash][0])
        print("TEST: ", hashes_by_split["test"][file_hash][0])


if __name__ == "__main__":
    main()
