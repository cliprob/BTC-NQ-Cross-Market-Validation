from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "data": {"analysis_timezone": "America/New_York", "nq_source_timezone": "America/Chicago"},
        "research": {
            "session_start": "09:30",
            "entry_end": "15:30",
            "session_end": "16:00",
            "momentum_window": 15,
            "impulse_quantile": 0.70,
            "correlation_window": 20,
            "hit_ratio_threshold": 0.55,
            "spearman_threshold": 0.10,
            "horizon_minutes": 5,
            "purge_minutes": 10,
            "min_validation_trades": 5,
            "min_test_trades": 5,
            "train_end": "2024-09-30 23:59:59",
            "validation_end": "2025-12-31 23:59:59",
            "test_start": "2026-01-01 00:00:00",
            "test_end": "2026-05-12 23:59:59",
        },
        "costs": {
            "tick_size_points": 0.25,
            "point_value_usd": 2.0,
            "slippage_ticks_per_side": 1.0,
            "commission_usd_per_side": 1.0,
        },
        "models": {
            "logistic_c": 0.1,
            "rf_estimators": 20,
            "rf_max_depth": 3,
            "rf_min_samples_leaf": 5,
            "random_state": 42,
            "threshold_grid": [0.45, 0.5, 0.55],
        },
        "output": {"directory": "unused"},
    }


def synthetic_market(start: str = "2024-06-03", days: int = 8, minutes: int = 420) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=days)
    timestamps = pd.DatetimeIndex(
        np.concatenate(
            [pd.date_range(f"{date.date()} 09:00", periods=minutes + 1, freq="min").values for date in dates]
        )
    ).tz_localize("America/New_York")
    index = np.arange(len(timestamps))
    rng = np.random.default_rng(17)
    nq_close = 18_000 + np.cumsum(rng.normal(0, 1.5, len(index))) + 8 * np.sin(index / 11)
    btc_close = 60_000 + 3 * (nq_close - 18_000) + np.cumsum(rng.normal(0, 2.0, len(index)))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "nq_open": nq_close - 0.1,
            "nq_high": nq_close + 0.5,
            "nq_low": nq_close - 0.5,
            "nq_close": nq_close,
            "nq_volume": 100,
            "nq_contract": "NQM4",
            "btc_open": btc_close - 0.2,
            "btc_high": btc_close + 1,
            "btc_low": btc_close - 1,
            "btc_close": btc_close,
            "btc_volume": 10,
            "continuity_break": timestamps.to_series().diff().ne(pd.Timedelta(minutes=1)).to_numpy(),
        }
    )
