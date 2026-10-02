"""Portable YAML paths for offline inference and workspace-backed model storage."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def runtime_dir():
    return (
        Path(os.environ.get("LIVEMOMENTS_RUNTIME_DIR", "/workspace/livemoments-runtime"))
        .expanduser()
        .resolve()
    )


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Inference config must be a YAML mapping")
    for key, value in cfg.items():
        if key.endswith("_path"):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty path")
            value = value.replace("${LIVEMOMENTS_RUNTIME_DIR}", str(runtime_dir()))
            value = value.replace("${LIVEMOMENTS_REPO_DIR}", str(ROOT))
            value = os.path.expandvars(value)
            if "${" in value or "$" in value:
                raise ValueError(f"Unresolved environment variable in {key}")
            item = Path(value).expanduser()
            cfg[key] = str((item if item.is_absolute() else path.parent / item).resolve())
    return cfg


def load_inference_config():
    return load_config(os.environ.get("LIVEMOMENTS_CONFIG", ROOT / "config/inference_example.yml"))
