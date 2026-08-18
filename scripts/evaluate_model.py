from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

from inference_utils import DEFAULT_DATA, DEFAULT_WEIGHTS, make_output_dir, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 weapon detector.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to the trained YOLOv8 checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to the merged dataset YAML file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, or a CUDA device id like 0.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="outputs/eval",
        help="Directory for the saved evaluation report.",
    )
    parser.add_argument("--name", type=str, default=None, help="Optional run name.")
    return parser


def count_dataset_instances(data_root: Path) -> tuple[dict[str, dict[str, int]], list[str]]:
    class_names = ["knife", "pistol", "rifle"]
    class_counts: dict[str, dict[str, int]] = {split: {name: 0 for name in class_names} for split in ["train", "valid", "test"]}
    issues: list[str] = []

    for split in ["train", "valid", "test"]:
        label_dir = data_root / split / "labels"
        if not label_dir.exists():
            issues.append(f"Missing label directory: {label_dir}")
            continue

        for label_path in label_dir.glob("*.txt"):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                issues.append(f"Empty label file: {label_path}")
                continue

            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    issues.append(f"Malformed label line in {label_path}: {line}")
                    continue
                class_id = parts[0]
                if class_id not in {"0", "1", "2"}:
                    issues.append(f"Unexpected class id {class_id} in {label_path}")
                    continue
                class_counts[split][class_names[int(class_id)]] += 1

    return class_counts, issues


def write_report(
    report_path: Path,
    summary_lines: list[str],
    class_counts: dict[str, dict[str, int]],
    issues: list[str],
    metrics: dict[str, float],
    per_class_metrics: dict[str, dict[str, float]],
) -> None:
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Weapon Detector Evaluation Report\n\n")
        handle.write("## Summary\n")
        for line in summary_lines:
            handle.write(f"- {line}\n")

        handle.write("\n## Dataset Imbalance\n")
        handle.write("| split | knife | pistol | rifle | total |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for split in ["train", "valid", "test"]:
            knife = class_counts[split]["knife"]
            pistol = class_counts[split]["pistol"]
            rifle = class_counts[split]["rifle"]
            handle.write(f"| {split} | {knife} | {pistol} | {rifle} | {knife + pistol + rifle} |\n")

        handle.write("\n## Per-Class Metrics\n")
        handle.write("| class | precision | recall | mAP50 | mAP50-95 |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for class_name, values in per_class_metrics.items():
            handle.write(
                f"| {class_name} | {values['precision']:.4f} | {values['recall']:.4f} | "
                f"{values['mAP50']:.4f} | {values['mAP50-95']:.4f} |\n"
            )

        handle.write("\n## Label Issues\n")
        if issues:
            for issue in issues[:200]:
                handle.write(f"- {issue}\n")
        else:
            handle.write("- No malformed or empty labels found.\n")

        handle.write("\n## Validation Metrics\n")
        for key, value in metrics.items():
            handle.write(f"- {key}: {value:.4f}\n")


def main() -> None:
    args = build_parser().parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    data_root = data_path.parent
    class_counts, issues = count_dataset_instances(data_root)

    model = YOLO(str(weights_path))
    device = resolve_device(args.device)
    output_dir = make_output_dir("eval", args.name, args.project)

    metrics_result = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        device=device,
        project=str(output_dir.parent),
        name=output_dir.name,
        plots=True,
        save_json=True,
        verbose=True,
    )

    metric_map = {
        "precision": float(getattr(metrics_result.box, "mp", 0.0)),
        "recall": float(getattr(metrics_result.box, "mr", 0.0)),
        "mAP50": float(getattr(metrics_result.box, "map50", 0.0)),
        "mAP50-95": float(getattr(metrics_result.box, "map", 0.0)),
    }

    per_class_metrics: dict[str, dict[str, float]] = {}
    class_names = list(getattr(model, "names", {}).values())
    for index, class_name in enumerate(class_names):
        precision, recall, ap50, ap = metrics_result.box.class_result(index)
        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(ap50),
            "mAP50-95": float(ap),
        }

    summary_lines = [
        f"Checkpoint: {weights_path}",
        f"Dataset: {data_path}",
        f"Split: {args.split}",
        f"Output directory: {output_dir}",
    ]

    report_path = output_dir / "evaluation_report.md"
    write_report(report_path, summary_lines, class_counts, issues, metric_map, per_class_metrics)

    print("Evaluation complete.")
    for line in summary_lines:
        print(line)
    print(f"Report saved to: {report_path}")
    print("Metrics:")
    for key, value in metric_map.items():
        print(f"{key}: {value:.4f}")
    print("Per-class metrics:")
    for class_name, values in per_class_metrics.items():
        print(
            f"{class_name}: precision={values['precision']:.4f}, recall={values['recall']:.4f}, "
            f"mAP50={values['mAP50']:.4f}, mAP50-95={values['mAP50-95']:.4f}"
        )


if __name__ == "__main__":
    main()