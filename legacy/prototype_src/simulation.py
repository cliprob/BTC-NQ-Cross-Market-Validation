import numpy as np
import pandas as pd


def build_signal_frame(
    df: pd.DataFrame,
    session_start: str = "10:30",
    session_end: str = "16:00",
    hit_ratio_threshold: float = 0.70,
    spearman_threshold: float = 0.60,
    std_window: int = 55,
    std_low_pct: float = 0.20,
    std_high_pct: float = 0.80,
) -> pd.DataFrame:
    """Flag valid regime bars and cluster consecutive minutes into events."""
    signals = df.copy()
    time_filter = (
        (signals.index.time >= pd.to_datetime(session_start).time())
        & (signals.index.time <= pd.to_datetime(session_end).time())
    )
    rolling_std = signals["nq_ret"].rolling(window=std_window).std()
    std_filter = (rolling_std >= rolling_std.quantile(std_low_pct)) & (
        rolling_std <= rolling_std.quantile(std_high_pct)
    )
    signals["valid_signal"] = (
        time_filter
        & (signals["hit_ratio"] >= hit_ratio_threshold)
        & (signals["corr_spearman"] >= spearman_threshold)
        & std_filter
    )
    previous_signal = signals["valid_signal"].shift(1, fill_value=False)
    signals["event_start"] = signals["valid_signal"] & (~previous_signal)
    return signals


def summarize_signal_events(signal_df: pd.DataFrame) -> dict[str, float]:
    """Return high-level counts for signal clustering diagnostics."""
    total_events = int(signal_df["event_start"].sum())
    trading_days = int(signal_df.index.normalize().nunique())
    events_per_day = total_events / trading_days if trading_days else 0.0
    return {
        "bars": int(len(signal_df)),
        "signal_minutes": int(signal_df["valid_signal"].sum()),
        "total_events": total_events,
        "trading_days": trading_days,
        "events_per_day": events_per_day,
    }


def evaluate_event_study(
    signal_df: pd.DataFrame,
    horizons: tuple[int, ...] | list[int],
    momentum_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute directional forward returns at signal start events."""
    study_df = signal_df.copy()
    for horizon in horizons:
        study_df[f"nq_fwd_{horizon}m"] = study_df["nq_close"].shift(-horizon) / study_df["nq_close"] - 1
        study_df[f"btc_fwd_{horizon}m"] = study_df["btc_close"].shift(-horizon) / study_df["btc_close"] - 1

    study_df["mom_window"] = study_df["nq_close"] / study_df["nq_close"].shift(momentum_window) - 1
    study_df["regime_dir"] = np.sign(study_df["mom_window"]).replace(0, 1)

    events_df = study_df[study_df["event_start"]].copy()
    for horizon in horizons:
        events_df[f"nq_dir_fwd_{horizon}m_bps"] = (
            events_df[f"nq_fwd_{horizon}m"] * events_df["regime_dir"] * 10000
        )
        events_df[f"btc_dir_fwd_{horizon}m_bps"] = (
            events_df[f"btc_fwd_{horizon}m"] * events_df["regime_dir"] * 10000
        )

    forward_columns = [column for column in events_df.columns if "dir_fwd" in column]
    forward_summary = (
        events_df[forward_columns]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T[["count", "mean", "std", "50%"]]
        .rename(columns={"50%": "median"})
    )

    win_rate_rows = []
    for horizon in horizons:
        win_rate_rows.append(
            {
                "horizon_minutes": horizon,
                "nq_win_rate_pct": (events_df[f"nq_dir_fwd_{horizon}m_bps"] > 0).mean() * 100,
                "btc_win_rate_pct": (events_df[f"btc_dir_fwd_{horizon}m_bps"] > 0).mean() * 100,
            }
        )
    win_rates = pd.DataFrame(win_rate_rows)

    return events_df, forward_summary, win_rates
