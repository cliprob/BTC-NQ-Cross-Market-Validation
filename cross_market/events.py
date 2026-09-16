"""Event construction and next-open execution simulation."""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from .data import cme_trading_dates, continuous_window
from .features import CROSS_FEATURES


def round_trip_cost_usd(config: dict[str, Any]) -> float:
    costs = config["costs"]
    slippage = 2 * costs["slippage_ticks_per_side"] * costs["tick_size_points"] * costs["point_value_usd"]
    commission = 2 * costs["commission_usd_per_side"]
    return float(slippage + commission)


def fit_impulse_threshold(features: pd.DataFrame, config: dict[str, Any]) -> float:
    train_end = pd.Timestamp(config["research"]["train_end"], tz=config["data"]["analysis_timezone"])
    training = features.loc[features["timestamp"] <= train_end, "nq_mom_15_bps"].abs().dropna()
    if training.empty:
        raise ValueError("No pre-validation data available to fit the event threshold")
    return float(training.quantile(config["research"]["impulse_quantile"]))


def build_events(features: pd.DataFrame, config: dict[str, Any], impulse_threshold: float) -> pd.DataFrame:
    research = config["research"]
    horizon = int(research["horizon_minutes"])
    start = time.fromisoformat(research["session_start"])
    entry_end = time.fromisoformat(research["entry_end"])
    session_end = time.fromisoformat(research["session_end"])
    local_time = features["timestamp"].dt.time
    trading_dates = cme_trading_dates(features["timestamp"].min(), features["timestamp"].max())
    is_trading_day = features["timestamp"].dt.date.isin(trading_dates)
    in_session = (local_time >= start) & (local_time <= entry_end) & is_trading_day
    impulse = features["nq_mom_15_bps"].abs().ge(impulse_threshold) & in_session
    starts = impulse & ~impulse.shift(fill_value=False)
    cost = round_trip_cost_usd(config)
    records: list[dict[str, Any]] = []
    last_exit = -1
    for signal_idx in np.flatnonzero(starts.to_numpy()):
        entry_idx = signal_idx + 1
        exit_idx = entry_idx + horizon
        if signal_idx <= last_exit or exit_idx >= len(features):
            continue
        if not continuous_window(features, signal_idx, exit_idx):
            continue
        signal_row = features.iloc[signal_idx]
        entry_row = features.iloc[entry_idx]
        exit_row = features.iloc[exit_idx]
        if entry_row["timestamp"].date() != exit_row["timestamp"].date():
            continue
        if exit_row["timestamp"].time() > session_end:
            continue
        if signal_row[CROSS_FEATURES].isna().any():
            continue
        direction = int(np.sign(signal_row["nq_mom_15_bps"]))
        if direction == 0:
            continue
        entry_price = float(entry_row["nq_open"])
        exit_price = float(exit_row["nq_open"])
        points = direction * (exit_price - entry_price)
        gross_usd = points * float(config["costs"]["point_value_usd"])
        net_usd = gross_usd - cost
        row = signal_row.to_dict()
        row.update(
            {
                "signal_time": signal_row["timestamp"],
                "entry_time": entry_row["timestamp"],
                "exit_time": exit_row["timestamp"],
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_points": points,
                "gross_usd": gross_usd,
                "net_usd": net_usd,
                "net_bps": points / entry_price * 10_000
                - cost / (entry_price * config["costs"]["point_value_usd"]) * 10_000,
                "target": int(net_usd > 0),
                "cross_rule": bool(
                    np.sign(signal_row["btc_mom_15_bps"]) == direction
                    and signal_row["btc_nq_spearman_55"] >= research["spearman_threshold"]
                    and signal_row["btc_nq_hit_55"] >= research["hit_ratio_threshold"]
                ),
            }
        )
        records.append(row)
        last_exit = exit_idx
    return pd.DataFrame.from_records(records)


def assert_no_overlap(events: pd.DataFrame) -> None:
    if events.empty:
        return
    ordered = events.sort_values("entry_time")
    overlap = ordered["entry_time"].iloc[1:].reset_index(drop=True) < ordered["exit_time"].iloc[
        :-1
    ].reset_index(drop=True)
    if overlap.any():
        raise AssertionError("Overlapping events detected")
