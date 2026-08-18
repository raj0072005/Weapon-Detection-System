from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def find_empty_label_files(folder: str | os.PathLike[str]) -> list[Path]:
    root = Path(folder)
    empty_files: list[Path] = []
    for path in root.rglob("*.txt"):
        if path.is_file() and not path.read_text(encoding="utf-8").strip():
            empty_files.append(path)
    return sorted(empty_files)


def remove_empty_label_files(folder: str | os.PathLike[str]) -> list[Path]:
    removed: list[Path] = []
    for path in find_empty_label_files(folder):
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def update_labels(folder: str | os.PathLike[str], class_id: int) -> None:
    root = Path(folder)
    for path in root.rglob("*.txt"):
        if not path.is_file():
            continue

        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                parts[0] = str(class_id)
            new_lines.append(" ".join(parts) + "\n")

        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(new_lines)


if __name__ == "__main__":
    # knife = 0
    update_labels("Dataset/knife", 0)

    # pistol = 1
    update_labels("Dataset/pistol", 1)

    # rifle = 2
    update_labels("Dataset/rifle", 2)

    removed = remove_empty_label_files("Dataset")
    print(f"Labels fixed successfully! Removed {len(removed)} empty label files.")