"""Walk-forward model estimation with fold-local preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CROSS_FEATURES, NQ_FEATURES
from .validation import expanding_folds


@dataclass
class ModelSpec:
    name: str
    features: list[str]
    estimator: Any


def model_specs(config: dict[str, Any]) -> list[ModelSpec]:
    settings = config["models"]
    return [
        ModelSpec(
            "logreg_nq",
            NQ_FEATURES,
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=settings["logistic_c"],
                            class_weight="balanced",
                            max_iter=2_000,
                            random_state=settings["random_state"],
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "logreg_cross",
            CROSS_FEATURES,
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=settings["logistic_c"],
                            class_weight="balanced",
                            max_iter=2_000,
                            random_state=settings["random_state"],
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            "random_forest_cross",
            CROSS_FEATURES,
            RandomForestClassifier(
                n_estimators=settings["rf_estimators"],
                max_depth=settings["rf_max_depth"],
                min_samples_leaf=settings["rf_min_samples_leaf"],
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=settings["random_state"],
            ),
        ),
    ]


def walk_forward_predictions(
    events: pd.DataFrame, spec: ModelSpec, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float]]:
    outputs: list[pd.DataFrame] = []
    importances: list[np.ndarray] = []
    for fold in expanding_folds(events, config["research"]["purge_minutes"]):
        train = events.iloc[list(fold.train_indices)]
        valid = events.iloc[list(fold.validation_indices)]
        if train["target"].nunique() < 2:
            continue
        estimator = clone(spec.estimator)
        estimator.fit(train[spec.features], train["target"])
        output = valid[["entry_time", "exit_time", "gross_usd", "net_usd", "net_bps", "target"]].copy()
        output["probability"] = estimator.predict_proba(valid[spec.features])[:, 1]
        output["fold"] = fold.name
        outputs.append(output)
        if spec.name == "random_forest_cross" and valid["target"].nunique() > 1:
            importance = permutation_importance(
                estimator,
                valid[spec.features],
                valid["target"],
                scoring="neg_brier_score",
                n_repeats=5,
                random_state=config["models"]["random_state"],
                n_jobs=-1,
            ).importances_mean
            importances.append(importance)
        elif spec.name.startswith("logreg"):
            importances.append(estimator.named_steps["model"].coef_[0])
    predictions = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    averaged = np.mean(importances, axis=0) if importances else np.full(len(spec.features), np.nan)
    return predictions, dict(zip(spec.features, averaged, strict=True))


def choose_threshold(predictions: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    minimum = int(config["research"]["min_validation_trades"])
    candidates = []
    for threshold in config["models"]["threshold_grid"]:
        selected = predictions[predictions["probability"] >= threshold]
        if len(selected) < minimum:
            continue
        candidates.append(
            {
                "threshold": float(threshold),
                "trades": int(len(selected)),
                "net_usd": float(selected["net_usd"].sum()),
                "mean_net_usd": float(selected["net_usd"].mean()),
            }
        )
    if not candidates:
        return {"threshold": None, "status": "rejected_minimum_trades", "candidates": []}
    best = max(candidates, key=lambda row: (row["mean_net_usd"], row["net_usd"], row["trades"]))
    return {**best, "status": "selected", "candidates": candidates}


def probability_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {"brier_score": float("nan"), "reliability": []}
    reliability = []
    bins = pd.cut(predictions["probability"], np.linspace(0, 1, 11), include_lowest=True)
    for interval, group in predictions.groupby(bins, observed=True):
        reliability.append(
            {
                "bin": str(interval),
                "predicted": float(group["probability"].mean()),
                "observed": float(group["target"].mean()),
                "count": int(len(group)),
            }
        )
    return {
        "brier_score": float(brier_score_loss(predictions["target"], predictions["probability"])),
        "reliability": reliability,
    }


def fit_final(spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    estimator = clone(spec.estimator)
    estimator.fit(train[spec.features], train["target"])
    output = test[["entry_time", "exit_time", "gross_usd", "net_usd", "net_bps", "target"]].copy()
    output["probability"] = estimator.predict_proba(test[spec.features])[:, 1]
    return output
