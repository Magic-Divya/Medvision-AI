from pathlib import Path
import re

root = Path("data/raw/chest_xray")

def groups(split):
    result = {}

    folder = root / split / "PNEUMONIA"

    for path in folder.glob("*"):
        if not path.is_file():
            continue

        match = re.match(r"^(person\d+)_", path.stem, re.IGNORECASE)

        if match:
            group = match.group(1).lower()
            result.setdefault(group, []).append(path.name)

    return result

train = groups("train")
test = groups("test")
shared = set(train).intersection(set(test))

print("Shared candidate groups:", len(shared))
print(
    "Training pneumonia images in shared groups:",
    sum(len(train[group]) for group in shared)
)
print(
    "Test pneumonia images in shared groups:",
    sum(len(test[group]) for group in shared)
)

print("\nFirst 10 shared groups:")
for group in sorted(shared)[:10]:
    print(
        group,
        "| train:", len(train[group]),
        "| test:", len(test[group])
    )

print("\nReminder: shared filename prefixes do not prove shared patient identity.")
