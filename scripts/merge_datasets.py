from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASETS = ["knife", "pistol", "rifle"]
SPLITS = ["train", "valid", "test"]
MERGED_DATASET = ROOT / "Dataset" / "weapon_dataset"


def ensure_clean_split_dirs() -> None:
    for split in SPLITS:
        for folder_name in ["images", "labels"]:
            target_dir = MERGED_DATASET / split / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for item in target_dir.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)


def copy_dataset(dataset_name: str, split: str, stats: dict[str, Counter], empty_labels: list[str]) -> None:
    source_images = ROOT / "Dataset" / dataset_name / split / "images"
    source_labels = ROOT / "Dataset" / dataset_name / split / "labels"
    target_images = MERGED_DATASET / split / "images"
    target_labels = MERGED_DATASET / split / "labels"

    if source_images.exists():
        for image_path in sorted(source_images.iterdir()):
            if image_path.is_file():
                shutil.copy2(image_path, target_images / f"{dataset_name}_{image_path.name}")
                stats[split]["images"] += 1

    if source_labels.exists():
        for label_path in sorted(source_labels.iterdir()):
            if not label_path.is_file() or label_path.suffix.lower() != ".txt":
                continue

            merged_label_path = target_labels / f"{dataset_name}_{label_path.name}"
            shutil.copy2(label_path, merged_label_path)
            stats[split]["labels"] += 1

            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                empty_labels.append(str(label_path))
                continue

            for line in content.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    stats[split]["malformed_lines"] += 1


def main() -> None:
    ensure_clean_split_dirs()

    stats: dict[str, Counter] = defaultdict(Counter)
    empty_labels: list[str] = []

    for dataset_name in SOURCE_DATASETS:
        for split in SPLITS:
            copy_dataset(dataset_name, split, stats, empty_labels)

    print("Datasets merged successfully!")
    print("\nMerge summary:")
    for split in SPLITS:
        print(
            f"{split}: {stats[split]['images']} images, {stats[split]['labels']} labels, "
            f"{stats[split]['malformed_lines']} malformed label lines"
        )

    print(f"\nEmpty label files: {len(empty_labels)}")
    for label_path in empty_labels[:20]:
        print(f"- {label_path}")


if __name__ == "__main__":
    main()