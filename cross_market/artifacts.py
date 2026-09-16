"""Artifact serialization and diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def plot_reliability(model_results: dict[str, dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
    for name, result in model_results.items():
        points = result["probability_metrics"]["reliability"]
        if points:
            ax.plot([p["predicted"] for p in points], [p["observed"] for p in points], marker="o", label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title="OOF reliability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_final_snapshot(results: dict[str, Any], path: Path) -> None:
    rows = [(name, details.get("mean_net_usd", np.nan)) for name, details in results["strategies"].items()]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels, values = zip(*rows, strict=True)
    colors = ["#2a9d8f" if value > 0 else "#d1495b" for value in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Mean net P&L per trade (USD, 1 MNQ)", title="Locked final test: 2026-01-01 to 2026-05-12")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
