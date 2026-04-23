from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = [
    "best_fer_model.keras",
    "fer_model.keras",
    "best_text_model.keras",
    "text_model.keras",
    "text_tokenizer.pkl",
    "label_maps.json",
    "inference_config.json",
    "fer_validation_summary_metrics.csv",
    "fer_test_summary_metrics.csv",
    "text_validation_summary_metrics.csv",
    "text_test_summary_metrics.csv",
    "fer_validation_confusion_matrix.png",
    "fer_test_confusion_matrix.png",
    "text_validation_confusion_matrix.png",
    "text_test_confusion_matrix.png",
    "README.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exported model artifacts.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts.resolve()
    print(f"Checking artifacts in: {artifacts_dir}")

    missing = []
    for name in REQUIRED_FILES:
        path = artifacts_dir / name
        if path.exists():
            print(f"OK      {name}")
        else:
            print(f"MISSING {name}")
            missing.append(name)

    if missing:
        raise SystemExit(
            "\nMissing artifacts. Run the notebook export cell, download "
            "emotion_model_artifacts.zip, and extract it into this folder."
        )

    print("\nAll required artifacts found.")


if __name__ == "__main__":
    main()
