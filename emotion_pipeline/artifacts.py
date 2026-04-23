from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_DIR_NAMES = (
    "artifacts",
    "emotion_model_artifacts",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_artifacts_dir(path: str | Path | None = None) -> Path:
    if path is not None:
        artifacts_dir = Path(path).expanduser().resolve()
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        return artifacts_dir

    root = project_root()
    for name in DEFAULT_ARTIFACT_DIR_NAMES:
        candidate = root / name
        if candidate.exists():
            return candidate

    return root / "artifacts"


def artifact_path(
    artifacts_dir: str | Path | None,
    explicit_path: str | Path | None,
    default_name: str,
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
    else:
        path = find_artifacts_dir(artifacts_dir) / default_name

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact missing: {path}\n"
            "Export emotion_model_artifacts.zip from the notebook, then extract it "
            "into the artifacts folder."
        )
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
