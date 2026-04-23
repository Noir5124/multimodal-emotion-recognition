from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]

OVERVIEW_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]

RUNS = [
    ("FER Validation", "fer_validation"),
    ("FER Test", "fer_test"),
    ("Text Validation", "text_validation"),
    ("Text Test", "text_test"),
]


def load_summary_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["metric"]: float(row["value"]) for row in reader}


def load_per_class_metrics(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "label": row["label"],
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "support": float(row["support"]),
                }
            )
    return rows


def load_history(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        history: dict[str, list[float]] = {field: [] for field in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                history[key].append(float(value))
    return history


def load_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    labels = rows[0][1:]
    matrix = np.array([[float(value) for value in row[1:]] for row in rows[1:]], dtype=float)
    return labels, matrix


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_summary_grouped(summary_data: dict[str, dict[str, float]], output_path: Path) -> None:
    labels = [name for name, _ in RUNS]
    x = np.arange(len(SUMMARY_METRICS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(15, 7))
    for idx, (display_name, prefix) in enumerate(RUNS):
        values = [summary_data[prefix][metric] for metric in SUMMARY_METRICS]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=display_name)

    ax.set_title("Summary Metrics Across Models and Splits")
    ax.set_xticks(x)
    ax.set_xticklabels([metric.replace("_", "\n") for metric in SUMMARY_METRICS])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(summary_data: dict[str, dict[str, float]], output_path: Path) -> None:
    matrix = np.array([[summary_data[prefix][metric] for metric in SUMMARY_METRICS] for _, prefix in RUNS])
    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Summary Metrics Heatmap")
    ax.set_xticks(range(len(SUMMARY_METRICS)))
    ax.set_xticklabels([metric.replace("_", "\n") for metric in SUMMARY_METRICS])
    ax.set_yticks(range(len(RUNS)))
    ax.set_yticklabels([name for name, _ in RUNS])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(per_class_data: dict[str, list[dict[str, float | str]]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    metric_names = ["precision", "recall", "f1"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for ax, (display_name, prefix) in zip(axes, RUNS):
        rows = per_class_data[prefix]
        labels = [str(row["label"]) for row in rows]
        x = np.arange(len(labels))
        width = 0.22

        for idx, metric_name in enumerate(metric_names):
            values = [float(row[metric_name]) for row in rows]
            ax.bar(x + (idx - 1) * width, values, width=width, label=metric_name.title(), color=colors[idx])

        ax.set_title(display_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3)
    fig.suptitle("Per-Class Precision / Recall / F1", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(confusion_data: dict[str, tuple[list[str], np.ndarray]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.flatten()

    max_value = max(matrix.max() for _, matrix in confusion_data.values())

    for ax, (display_name, prefix) in zip(axes, RUNS):
        labels, matrix = confusion_data[prefix]
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max_value)
        ax.set_title(display_name)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = int(matrix[row_idx, col_idx])
                color = "white" if matrix[row_idx, col_idx] > max_value * 0.45 else "black"
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle("Confusion Matrices")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(fer_history: dict[str, list[float]], text_history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    fer_epochs = np.arange(1, len(fer_history["accuracy"]) + 1)
    text_epochs = np.arange(1, len(text_history["accuracy"]) + 1)

    axes[0, 0].plot(fer_epochs, fer_history["accuracy"], label="Train", linewidth=2)
    axes[0, 0].plot(fer_epochs, fer_history["val_accuracy"], label="Validation", linewidth=2)
    axes[0, 0].set_title("FER Accuracy")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(fer_epochs, fer_history["loss"], label="Train", linewidth=2)
    axes[0, 1].plot(fer_epochs, fer_history["val_loss"], label="Validation", linewidth=2)
    axes[0, 1].set_title("FER Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(text_epochs, text_history["accuracy"], label="Train", linewidth=2)
    axes[1, 0].plot(text_epochs, text_history["val_accuracy"], label="Validation", linewidth=2)
    axes[1, 0].set_title("Text Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(text_epochs, text_history["loss"], label="Train", linewidth=2)
    axes[1, 1].plot(text_epochs, text_history["val_loss"], label="Validation", linewidth=2)
    axes[1, 1].set_title("Text Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig.suptitle("Training Curves", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview_cards(summary_data: dict[str, dict[str, float]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.flatten()
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    for ax, (display_name, prefix) in zip(axes, RUNS):
        values = [summary_data[prefix][metric] for metric in OVERVIEW_METRICS]
        x = np.arange(len(OVERVIEW_METRICS))
        ax.bar(x, values, color=colors)
        ax.set_title(display_name)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([metric.replace("_", "\n") for metric in OVERVIEW_METRICS])
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)

    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    fig.suptitle("High-Level Evaluation Snapshot", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(output_dir: Path, generated_files: list[Path]) -> None:
    manifest = {
        "output_dir": str(output_dir),
        "files": [path.name for path in generated_files],
    }
    (output_dir / "evaluation_graphs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation graphs from exported emotion model metrics.")
    parser.add_argument("--artifacts", default=r"D:\emotion_model_artifacts", help="Path to exported artifact folder.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "evaluation_graphs"),
        help="Directory to save generated graph PNGs.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    summary_data = {prefix: load_summary_metrics(artifacts_dir / f"{prefix}_summary_metrics.csv") for _, prefix in RUNS}
    per_class_data = {prefix: load_per_class_metrics(artifacts_dir / f"{prefix}_per_class_metrics.csv") for _, prefix in RUNS}
    confusion_data = {prefix: load_confusion_matrix(artifacts_dir / f"{prefix}_confusion_matrix.csv") for _, prefix in RUNS}
    fer_history = load_history(artifacts_dir / "fer_training_history.csv")
    text_history = load_history(artifacts_dir / "text_training_history.csv")

    generated_files = [
        output_dir / "evaluation_snapshot.png",
        output_dir / "summary_metrics_grouped.png",
        output_dir / "summary_metrics_heatmap.png",
        output_dir / "per_class_metrics.png",
        output_dir / "confusion_matrices.png",
        output_dir / "training_curves.png",
    ]

    plot_overview_cards(summary_data, generated_files[0])
    plot_summary_grouped(summary_data, generated_files[1])
    plot_summary_heatmap(summary_data, generated_files[2])
    plot_per_class_metrics(per_class_data, generated_files[3])
    plot_confusion_matrices(confusion_data, generated_files[4])
    plot_training_curves(fer_history, text_history, generated_files[5])
    write_manifest(output_dir, generated_files)

    print(f"Saved graphs to: {output_dir}")
    for path in generated_files:
        print(path.name)


if __name__ == "__main__":
    main()
