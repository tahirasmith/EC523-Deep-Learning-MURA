import os
from collections import Counter


def collect_bodyparts(root_dir, split):
    path = os.path.join(root_dir, split)
    counter = Counter()

    for root, _, files in os.walk(path):
        for f in files:
            if not f.endswith(".png"):
                continue

            full_path = os.path.join(root, f)

            parts = full_path.split(os.sep)
            bodypart = next((p for p in parts if p.startswith("XR_")), None)

            if bodypart:
                counter[bodypart] += 1

    return counter


if __name__ == "__main__":
    root_dir = "MURA-v1.1"

    train_counts = collect_bodyparts(root_dir, "train")
    val_counts = collect_bodyparts(root_dir, "valid")

    print("\n-- training body parts --")
    for k, v in sorted(train_counts.items()):
        print(f"{k}: {v}")

    print("\n-- validation body parts --")
    for k, v in sorted(val_counts.items()):
        print(f"{k}: {v}")

