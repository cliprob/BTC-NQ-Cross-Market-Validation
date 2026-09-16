"""Trading and probabilistic model metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, skew


def block_bootstrap_ci(
    trades: pd.DataFrame, value: str = "net_usd", samples: int = 2_000, seed: int = 42
) -> tuple[float, float]:
    if trades.empty:
        return (float("nan"), float("nan"))
    daily = trades.assign(day=trades["entry_time"].dt.date).groupby("day")[value].apply(np.asarray)
    days = list(daily)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for sample in range(samples):
        selected = rng.integers(0, len(days), len(days))
        values = np.concatenate([daily.iloc[index] for index in selected])
        means[sample] = values.mean()
    quantiles = np.quantile(means, [0.025, 0.975])
    return float(quantiles[0]), float(quantiles[1])


def deflated_sharpe_probability(daily_returns: pd.Series, trial_count: int) -> float:
    """Approximate DSR probability using the expected maximum Sharpe under selection."""
    values = daily_returns.dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) == 0:
        return float("nan")
    observed = values.mean() / values.std(ddof=1)
    trials = max(1, trial_count)
    expected_max = norm.ppf(1 - 1 / max(2, trials)) / np.sqrt(len(values))
    standard_error = np.sqrt((1 + 0.5 * observed**2 - skew(values) * observed) / max(1, len(values) - 1))
    return float(norm.cdf((observed - expected_max) / standard_error))


def trading_metrics(trades: pd.DataFrame, trial_count: int = 1) -> dict[str, Any]:
    if trades.empty:
        return {"trades": 0, "status": "no_trades"}
    net = trades["net_usd"].astype(float)
    gross_profit = float(net[net > 0].sum())
    gross_loss = float(-net[net < 0].sum())
    raw_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    daily = trades.assign(day=trades["entry_time"].dt.date).groupby("day")["net_usd"].sum()
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std(ddof=1)) if daily.std(ddof=1) else float("nan")
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    ci_low, ci_high = block_bootstrap_ci(trades)
    return {
        "trades": int(len(trades)),
        "gross_usd": float(trades["gross_usd"].sum()),
        "net_usd": float(net.sum()),
        "mean_net_usd": float(net.mean()),
        "mean_net_bps": float(trades["net_bps"].mean()),
        "win_rate": float((net > 0).mean()),
        "profit_factor_raw": "inf" if np.isinf(raw_pf) else float(raw_pf),
        "profit_factor_capped": float(min(raw_pf, 10.0)),
        "daily_sharpe": sharpe,
        "max_drawdown_usd": float(drawdown.min()),
        "mean_net_usd_ci_95": [ci_low, ci_high],
        "deflated_sharpe_probability": deflated_sharpe_probability(daily, trial_count),
    }


def edge_verdict(metrics: dict[str, Any], minimum_trades: int) -> str:
    if metrics.get("trades", 0) < minimum_trades:
        return "inconclusive_insufficient_trades"
    interval = metrics.get("mean_net_usd_ci_95", [float("nan"), float("nan")])
    if metrics.get("net_usd", 0) > 0 and interval[0] > 0:
        return "positive_edge_supported"
    return "hypothesis_rejected"
