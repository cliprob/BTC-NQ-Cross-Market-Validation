import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.stattools import adfuller


def fast_rolling_spearman(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling Spearman correlation using rank-transformed windows."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < window:
        return np.full(len(x), np.nan)

    x_windows = sliding_window_view(np.asarray(x), window_shape=window)
    y_windows = sliding_window_view(np.asarray(y), window_shape=window)

    x_rank = np.argsort(np.argsort(x_windows, axis=1), axis=1)
    y_rank = np.argsort(np.argsort(y_windows, axis=1), axis=1)

    rank_mean = (window - 1) / 2.0
    rank_var = (window**2 - 1) / 12.0
    covariance = np.mean((x_rank - rank_mean) * (y_rank - rank_mean), axis=1)
    correlation = covariance / rank_var

    return np.concatenate((np.full(window - 1, np.nan), correlation))


def add_correlation_metrics(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add rolling Pearson, rolling Spearman, and directional hit ratio columns."""
    metrics = df.copy()
    metrics["corr_pearson"] = metrics["btc_ret"].rolling(window=window).corr(metrics["nq_ret"])
    metrics["corr_spearman"] = fast_rolling_spearman(
        metrics["btc_ret"].to_numpy(),
        metrics["nq_ret"].to_numpy(),
        window,
    )

    btc_direction = np.sign(metrics["btc_ret"])
    nq_direction = np.sign(metrics["nq_ret"])
    metrics["hit_ratio"] = (btc_direction == nq_direction).astype(float).rolling(window=window).mean()

    return metrics.dropna(subset=["corr_pearson", "corr_spearman", "hit_ratio"]).copy()


def run_adf(series: pd.Series, name: str, sample_size: int = 5000) -> dict[str, object]:
    """Run an ADF test on a subsample while preserving time order."""
    clean = series.dropna()
    if len(clean) > sample_size:
        clean = clean.sample(n=sample_size, random_state=42).sort_index()

    stat, p_value, lags, _, critical_values, _ = adfuller(clean, maxlag=15, autolag="AIC")
    stationary = p_value < 0.05
    return {
        "Series": name,
        "N (sampled)": len(clean),
        "ADF Statistic": round(stat, 4),
        "p-value": round(p_value, 4),
        "Lags Used": lags,
        "Crit 5%": round(critical_values["5%"], 4),
        "Stationary?": "YES" if stationary else "NO",
        "Integration Order": "I(0)" if stationary else "I(1)+",
    }


def rolling_volume_weighted_corr(
    ret_x: pd.Series,
    ret_y: pd.Series,
    volume: pd.Series,
    window: int,
) -> np.ndarray:
    """Compute a rolling correlation weighted by normalized window volume."""
    result = np.full(len(ret_x), np.nan)
    x_values = ret_x.to_numpy()
    y_values = ret_y.to_numpy()
    volume_values = volume.to_numpy()

    for index in range(window - 1, len(x_values)):
        weight_slice = volume_values[index - window + 1 : index + 1]
        x_slice = x_values[index - window + 1 : index + 1]
        y_slice = y_values[index - window + 1 : index + 1]

        if np.isnan(weight_slice).any() or np.isnan(x_slice).any() or np.isnan(y_slice).any():
            continue

        total_weight = weight_slice.sum()
        weights = (
            weight_slice / total_weight
            if total_weight > 0
            else np.full(window, 1.0 / window)
        )
        x_mean = np.average(x_slice, weights=weights)
        y_mean = np.average(y_slice, weights=weights)
        covariance = np.sum(weights * (x_slice - x_mean) * (y_slice - y_mean))
        variance_x = np.sum(weights * (x_slice - x_mean) ** 2)
        variance_y = np.sum(weights * (y_slice - y_mean) ** 2)
        denominator = np.sqrt(variance_x * variance_y)
        result[index] = covariance / denominator if denominator > 1e-12 else np.nan

    return result


def build_volume_analysis_frame(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Create the reusable volume-weighted feature set used in the notebook."""
    volume_df = df[
        [
            "btc_close",
            "nq_close",
            "btc_volume",
            "nq_volume",
            "btc_ret",
            "nq_ret",
            "corr_spearman",
            "hit_ratio",
        ]
    ].copy()

    volume_df["btc_vwap"] = (
        (volume_df["btc_close"] * volume_df["btc_volume"]).rolling(window).sum()
        / volume_df["btc_volume"].rolling(window).sum()
    )
    volume_df["nq_vwap"] = (
        (volume_df["nq_close"] * volume_df["nq_volume"]).rolling(window).sum()
        / volume_df["nq_volume"].rolling(window).sum()
    )
    volume_df["btc_vwap_dev"] = (
        (volume_df["btc_close"] - volume_df["btc_vwap"])
        / volume_df["btc_close"].rolling(window).std()
    )
    volume_df["nq_vwap_dev"] = (
        (volume_df["nq_close"] - volume_df["nq_vwap"])
        / volume_df["nq_close"].rolling(window).std()
    )
    volume_df["combined_vol"] = np.sqrt(volume_df["btc_volume"] * volume_df["nq_volume"])
    volume_df["vw_corr"] = rolling_volume_weighted_corr(
        volume_df["btc_ret"],
        volume_df["nq_ret"],
        volume_df["combined_vol"],
        window,
    )
    volume_df["vol_quintile"] = pd.qcut(
        volume_df["combined_vol"],
        q=5,
        labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"],
        duplicates="drop",
    )
    volume_df["vol_rank"] = volume_df["combined_vol"].rank(pct=True)
    volume_df["conviction_score"] = volume_df["vol_rank"] * volume_df["vw_corr"].abs()

    return volume_df
