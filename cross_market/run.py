"""Command-line entry point for validation and locked final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import plot_final_snapshot, plot_reliability, write_json
from .config import config_hash, load_config, public_config, source_hash
from .data import load_market_data
from .events import assert_no_overlap, build_events, fit_impulse_threshold, round_trip_cost_usd
from .features import CROSS_FEATURES, NQ_FEATURES, build_features
from .metrics import edge_verdict, trading_metrics
from .models import choose_threshold, fit_final, model_specs, probability_metrics, walk_forward_predictions


def _timestamp(value: str, timezone: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=timezone)


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _prepare(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    market = load_market_data(config)
    features = build_features(market, config)
    impulse_threshold = fit_impulse_threshold(features, config)
    events = build_events(features, config, impulse_threshold)
    if events.empty:
        raise RuntimeError("No eligible events were generated")
    assert_no_overlap(events)
    return market, events, impulse_threshold


def _validation(config: dict[str, Any], output: Path) -> dict[str, Any]:
    market, events, impulse_threshold = _prepare(config)
    timezone = config["data"]["analysis_timezone"]
    test_start = _timestamp(config["research"]["test_start"], timezone)
    oof_start = pd.Timestamp("2024-10-01", tz=timezone)
    validation_events = events[
        (events["entry_time"] >= oof_start) & (events["entry_time"] < test_start)
    ].copy()
    trial_count = 2
    strategies: dict[str, Any] = {
        "nq_momentum_baseline": trading_metrics(validation_events, trial_count),
        "btc_nq_rule": trading_metrics(validation_events[validation_events["cross_rule"]], trial_count),
    }
    models: dict[str, Any] = {}
    for spec in model_specs(config):
        predictions, importance = walk_forward_predictions(
            events[events["entry_time"] < test_start].reset_index(drop=True), spec, config
        )
        selection = choose_threshold(predictions, config)
        trial_count += len(config["models"]["threshold_grid"])
        selected = (
            predictions[predictions["probability"] >= selection["threshold"]]
            if selection["threshold"] is not None
            else predictions.iloc[0:0]
        )
        models[spec.name] = {
            "features": spec.features,
            "selection": selection,
            "probability_metrics": probability_metrics(predictions),
            "selected_metrics": trading_metrics(selected, trial_count),
            "importance": importance,
        }
    plot_reliability(models, output / "reliability_oof.png")
    locked = {
        "schema_version": 1,
        "config_hash": config_hash(config),
        "source_hash": source_hash(),
        "configuration": public_config(config),
        "data_range": {
            "start": str(market["timestamp"].min()),
            "end": str(market["timestamp"].max()),
        },
        "event_definition": {
            "impulse_threshold_bps": impulse_threshold,
            "horizon_minutes": config["research"]["horizon_minutes"],
            "execution": "signal after t close; enter t+1 open; exit future open",
            "one_position_at_a_time": True,
            "round_trip_cost_usd": round_trip_cost_usd(config),
        },
        "feature_sets": {"nq_only": NQ_FEATURES, "cross_market": CROSS_FEATURES},
        "validation": {"strategies": strategies, "models": models},
        "trial_count": trial_count,
        "final_test": {
            "start": config["research"]["test_start"],
            "end": config["research"]["test_end"],
            "evaluated": False,
        },
    }
    locked["locked_setting_hash"] = _hash_payload(
        {
            "config_hash": locked["config_hash"],
            "source_hash": locked["source_hash"],
            "event_definition": locked["event_definition"],
            "feature_sets": locked["feature_sets"],
            "thresholds": {name: details["selection"]["threshold"] for name, details in models.items()},
        }
    )
    write_json(output / "validation_manifest.json", locked)
    ledger_rows = []
    for name, details in models.items():
        for row in details["selection"]["candidates"]:
            ledger_rows.append({"model": name, **row})
    pd.DataFrame(ledger_rows).to_csv(output / "trial_ledger.csv", index=False)
    return locked


def _psi(reference: pd.Series, observed: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna().to_numpy()
    observed = observed.dropna().to_numpy()
    if len(reference) == 0 or len(observed) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_hist = np.histogram(reference, bins=edges)[0] / len(reference)
    obs_hist = np.histogram(observed, bins=edges)[0] / len(observed)
    ref_hist = np.clip(ref_hist, 1e-6, None)
    obs_hist = np.clip(obs_hist, 1e-6, None)
    return float(np.sum((obs_hist - ref_hist) * np.log(obs_hist / ref_hist)))


def _final(config: dict[str, Any], output: Path, force: bool = False) -> dict[str, Any]:
    manifest_path = output / "validation_manifest.json"
    result_path = output / "final_results.json"
    if not manifest_path.exists():
        raise RuntimeError("Run --stage validation before opening the final test")
    if result_path.exists() and not force:
        raise RuntimeError("Final test is already evaluated; refusing to rerun without --force")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_hash"] != config_hash(config):
        raise RuntimeError("Configuration changed after validation lock")
    if manifest["source_hash"] != source_hash():
        raise RuntimeError("Research code changed after validation lock")
    _, events, impulse_threshold = _prepare(config)
    if not np.isclose(impulse_threshold, manifest["event_definition"]["impulse_threshold_bps"]):
        raise RuntimeError("Event threshold differs from the locked manifest")
    timezone = config["data"]["analysis_timezone"]
    test_start = _timestamp(config["research"]["test_start"], timezone)
    test_end = _timestamp(config["research"]["test_end"], timezone)
    pretest = events[events["entry_time"] < test_start].copy()
    test = events[(events["entry_time"] >= test_start) & (events["entry_time"] <= test_end)].copy()
    trial_count = int(manifest["trial_count"])
    minimum = int(config["research"]["min_test_trades"])
    strategies: dict[str, Any] = {}
    baseline_sets = {
        "nq_momentum_baseline": test,
        "btc_nq_rule": test[test["cross_rule"]],
    }
    for name, selected in baseline_sets.items():
        metrics = trading_metrics(selected, trial_count)
        metrics["verdict"] = edge_verdict(metrics, minimum)
        strategies[name] = metrics
    probabilities: dict[str, Any] = {}
    specs = {spec.name: spec for spec in model_specs(config)}
    for name, details in manifest["validation"]["models"].items():
        threshold = details["selection"]["threshold"]
        if threshold is None:
            metrics = {"trades": 0, "status": "rejected_during_validation"}
            probabilities[name] = {"brier_score": None}
        else:
            predicted = fit_final(specs[name], pretest, test)
            probabilities[name] = probability_metrics(predicted)
            metrics = trading_metrics(predicted[predicted["probability"] >= threshold], trial_count)
        metrics["verdict"] = edge_verdict(metrics, minimum)
        strategies[name] = metrics
    drift = {feature: _psi(pretest[feature], test[feature]) for feature in CROSS_FEATURES}
    results = {
        "schema_version": 1,
        "locked_setting_hash": manifest["locked_setting_hash"],
        "test_period": {"start": str(test_start), "end": str(test_end)},
        "candidate_events": int(len(test)),
        "strategies": strategies,
        "probability_metrics": probabilities,
        "feature_drift_psi": drift,
        "conclusion": "No strategy may be described as a positive edge unless its verdict is positive_edge_supported.",
    }
    write_json(result_path, results)
    plot_final_snapshot(results, output / "final_test_snapshot.png")
    manifest["final_test"]["evaluated"] = True
    manifest["final_test"]["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    write_json(manifest_path, manifest)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("validation", "final"), required=True)
    parser.add_argument(
        "--force", action="store_true", help="Explicitly rerun an already evaluated final test"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    result = _validation(config, output) if args.stage == "validation" else _final(config, output, args.force)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
