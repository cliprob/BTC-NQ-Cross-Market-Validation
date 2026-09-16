from __future__ import annotations

import pandas as pd

from cross_market.metrics import edge_verdict, trading_metrics
from cross_market.models import choose_threshold


def test_infinite_profit_factor_cannot_bypass_minimum(config) -> None:
    predictions = pd.DataFrame(
        {
            "probability": [0.99, 0.98],
            "net_usd": [10.0, 10.0],
        }
    )
    selected = choose_threshold(predictions, config)
    assert selected["status"] == "rejected_minimum_trades"
    assert selected["threshold"] is None


def test_edge_requires_sample_and_positive_ci() -> None:
    times = pd.date_range("2025-01-02 10:00", periods=20, freq="D", tz="America/New_York")
    trades = pd.DataFrame(
        {"entry_time": times, "net_usd": [2.0] * 20, "gross_usd": [5.0] * 20, "net_bps": [1.0] * 20}
    )
    metrics = trading_metrics(trades)
    assert edge_verdict(metrics, 50) == "inconclusive_insufficient_trades"
