"""Generate publication figures from the frozen research artifacts.

This module is deliberately separate from ``cross_market`` so documentation
changes cannot alter the source hash protecting the locked final evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "outputs" / "portfolio" / "final_results.json"
MANIFEST_PATH = ROOT / "outputs" / "portfolio" / "validation_manifest.json"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

NAVY = "#16324F"
TEAL = "#2A7F9E"
SLATE = "#5B6770"
LIGHT = "#E8EEF2"
RED = "#A23B3B"

LABELS = {
    "nq_momentum_baseline": "NQ momentum baseline",
    "btc_nq_rule": "BTC-NQ agreement rule",
    "logreg_nq": "Logistic Regression, NQ only",
    "logreg_cross": "Logistic Regression, cross-market",
    "random_forest_cross": "Random Forest, cross-market",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _journal_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": SLATE,
            "axes.linewidth": 0.7,
            "xtick.color": SLATE,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_final_test_forest(results: dict[str, Any]) -> None:
    order = [
        "nq_momentum_baseline",
        "btc_nq_rule",
        "logreg_nq",
        "logreg_cross",
        "random_forest_cross",
    ]
    strategies = results["strategies"]
    means = np.array([strategies[name]["mean_net_usd"] for name in order])
    lows = np.array([strategies[name]["mean_net_usd_ci_95"][0] for name in order])
    highs = np.array([strategies[name]["mean_net_usd_ci_95"][1] for name in order])
    trades = [strategies[name]["trades"] for name in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.axvspan(-1.5, 1.5, color=LIGHT, alpha=0.45, zorder=0)
    ax.axvline(0, color=SLATE, linewidth=1.0, linestyle="--", zorder=1)
    colors = [TEAL if value >= 0 else RED for value in means]
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="none",
        ecolor=SLATE,
        elinewidth=1.4,
        capsize=3.5,
        zorder=2,
    )
    ax.scatter(means, y, s=48, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_yticks(y, [f"{LABELS[name]}  (n = {count})" for name, count in zip(order, trades, strict=True)])
    ax.invert_yaxis()
    ax.set_xlabel("Mean net P&L per accepted trade (USD, one MNQ)")
    ax.set_title(
        "Locked final test: point estimates and 95% day-block bootstrap intervals", loc="left", color=NAVY
    )
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(lows) - 6, max(highs) + 7)
    fig.text(
        0.01,
        0.01,
        "Intervals crossing zero indicate that the sign of the population mean is unresolved.",
        fontsize=8.2,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(FIGURE_DIR / "final_test_forest.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_oof_reliability(manifest: dict[str, Any]) -> None:
    models = manifest["validation"]["models"]
    order = ["logreg_nq", "logreg_cross", "random_forest_cross"]
    colors = [NAVY, TEAL, SLATE]
    markers = ["o", "s", "D"]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(
        [0.35, 0.75],
        [0.35, 0.75],
        linestyle="--",
        color="#8A949C",
        linewidth=1.0,
        label="Perfect calibration",
    )
    for name, color, marker in zip(order, colors, markers, strict=True):
        metrics = models[name]["probability_metrics"]
        points = [point for point in metrics["reliability"] if point["count"] >= 10]
        ax.plot(
            [point["predicted"] for point in points],
            [point["observed"] for point in points],
            marker=marker,
            markersize=5,
            linewidth=1.35,
            color=color,
            label=f"{LABELS[name]} (Brier {metrics['brier_score']:.4f})",
        )
    ax.set(
        xlim=(0.35, 0.75),
        ylim=(0.35, 0.75),
        xlabel="Mean predicted probability",
        ylabel="Observed positive rate",
    )
    ax.set_title("Out-of-fold probability reliability", loc="left", color=NAVY)
    ax.grid(color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "reliability_oof_journal.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _journal_style()
    plot_final_test_forest(_load_json(RESULTS_PATH))
    plot_oof_reliability(_load_json(MANIFEST_PATH))


if __name__ == "__main__":
    main()
