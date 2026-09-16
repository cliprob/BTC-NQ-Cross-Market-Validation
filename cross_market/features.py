"""Causal features available at the signal-bar close."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata

NQ_FEATURES = [
    "nq_mom_5_bps",
    "nq_mom_15_bps",
    "nq_mom_30_bps",
    "nq_vol_15_bps",
    "nq_vol_60_bps",
    "nq_atr_14_bps",
    "nq_rsi_14",
    "hour_sin",
    "hour_cos",
]

CROSS_FEATURES = NQ_FEATURES + [
    "btc_mom_5_bps",
    "btc_mom_15_bps",
    "btc_mom_30_bps",
    "btc_vol_15_bps",
    "btc_vol_60_bps",
    "btc_nq_spearman_55",
    "btc_nq_hit_55",
    "momentum_spread_bps",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative)


def rolling_spearman(left: pd.Series, right: pd.Series, window: int, chunk_size: int = 75_000) -> pd.Series:
    """Exact rolling Spearman correlation, chunked to bound peak memory."""
    x = left.to_numpy(dtype=float)
    y = right.to_numpy(dtype=float)
    result = np.full(len(x), np.nan)
    first_endpoint = window - 1
    for endpoint_start in range(first_endpoint, len(x), chunk_size):
        endpoint_stop = min(len(x), endpoint_start + chunk_size)
        source_start = endpoint_start - window + 1
        x_windows = sliding_window_view(x[source_start:endpoint_stop], window)
        y_windows = sliding_window_view(y[source_start:endpoint_stop], window)
        valid = np.isfinite(x_windows).all(axis=1) & np.isfinite(y_windows).all(axis=1)
        if not valid.any():
            continue
        x_rank = rankdata(x_windows[valid], axis=1)
        y_rank = rankdata(y_windows[valid], axis=1)
        x_rank -= x_rank.mean(axis=1, keepdims=True)
        y_rank -= y_rank.mean(axis=1, keepdims=True)
        denominator = np.sqrt((x_rank**2).sum(axis=1) * (y_rank**2).sum(axis=1))
        correlation = (x_rank * y_rank).sum(axis=1) / denominator
        destination = np.arange(endpoint_start, endpoint_stop)[valid]
        result[destination] = correlation
    return pd.Series(result, index=left.index)


def build_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    segment = result["continuity_break"].cumsum()
    for asset in ("nq", "btc"):
        close = result[f"{asset}_close"]
        grouped_close = close.groupby(segment)
        returns = grouped_close.pct_change(fill_method=None)
        result[f"{asset}_ret_1"] = returns
        for window in (5, 15, 30):
            result[f"{asset}_mom_{window}_bps"] = grouped_close.pct_change(window, fill_method=None) * 10_000
        for window in (15, 60):
            result[f"{asset}_vol_{window}_bps"] = (
                returns.groupby(segment).rolling(window).std().reset_index(level=0, drop=True) * 10_000
            )
    prior_close = result["nq_close"].groupby(segment).shift()
    true_range = pd.concat(
        [
            result["nq_high"] - result["nq_low"],
            (result["nq_high"] - prior_close).abs(),
            (result["nq_low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["nq_atr_14_bps"] = (
        true_range.groupby(segment).rolling(14).mean().reset_index(level=0, drop=True)
        / result["nq_close"]
        * 10_000
    )
    result["nq_rsi_14"] = result["nq_close"].groupby(segment).transform(_rsi)
    window = int(config["research"]["correlation_window"])
    result["btc_nq_spearman_55"] = rolling_spearman(result["btc_ret_1"], result["nq_ret_1"], window)
    sign_match = np.sign(result["btc_ret_1"]) == np.sign(result["nq_ret_1"])
    result["btc_nq_hit_55"] = (
        sign_match.astype(float)
        .where(result[["btc_ret_1", "nq_ret_1"]].notna().all(axis=1))
        .rolling(window)
        .mean()
    )
    result["momentum_spread_bps"] = result["btc_mom_15_bps"] - result["nq_mom_15_bps"]
    minute = result["timestamp"].dt.hour * 60 + result["timestamp"].dt.minute
    result["hour_sin"] = np.sin(2 * np.pi * minute / 1440)
    result["hour_cos"] = np.cos(2 * np.pi * minute / 1440)
    return result
