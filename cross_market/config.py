"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config["_config_path"] = str(config_path)
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the reproducible portion of config, excluding machine-local paths."""
    result = {key: value for key, value in config.items() if not key.startswith("_")}
    result = json.loads(json.dumps(result))
    if "data" in result:
        result["data"]["btc_path"] = Path(result["data"]["btc_path"]).name
        result["data"]["nq_path"] = Path(result["data"]["nq_path"]).name
    if "output" in result:
        result["output"]["directory"] = "outputs/portfolio"
    return result


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def source_hash() -> str:
    """Hash the maintained Python implementation included in a research lock."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
