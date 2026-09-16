import numpy as np
import pandas as pd

try:
    from .indicators import fast_rolling_spearman
except ImportError:
    from indicators import fast_rolling_spearman


def compute_unified_robustness_score(
    oos_win_rate: float,
    oos_avg_bps: float,
    is_win_rate: float,
    is_avg_bps: float,
    total_events: int,
    weight_win_rate: float = 1.0,
    weight_bps: float = 0.35,
) -> float:
    """
    Unified robustness score that rewards OOS outperformance while penalizing overfitting.
    
    Uses asymmetric gap penalties:
    - Only penalizes when IS > OOS (overfitting)
    - Lightly rewards when OOS > IS (positive surprise)
    - Maintains preference for high-quality signals
    
    Args:
        oos_win_rate: Out-of-sample win rate (%)
        oos_avg_bps: Out-of-sample avg returns (basis points)
        is_win_rate: In-sample win rate (%)
        is_avg_bps: In-sample avg returns (basis points)
        total_events: Total number of events/signals
        weight_win_rate: Primary weight on OOS win rate (default 1.0)
        weight_bps: Secondary weight on OOS returns (default 0.35)
    
    Returns:
        Unified robustness score (higher is better)
    """
    if np.isnan([oos_win_rate, is_win_rate, oos_avg_bps, is_avg_bps]).any() or total_events < 1:
        return np.nan
    
    # Asymmetric gap penalties: only penalize overfitting, reward surprises
    win_rate_degradation = max(0, is_win_rate - oos_win_rate)  # Penalize if IS > OOS
    win_rate_surprise = max(0, oos_win_rate - is_win_rate) * 0.15  # Light reward if OOS > IS
    
    bps_degradation = max(0, is_avg_bps - oos_avg_bps)  # Penalize if IS > OOS
    bps_surprise = max(0, oos_avg_bps - is_avg_bps) * 0.08  # Light reward if OOS > IS
    
    score = (
        weight_win_rate * oos_win_rate           # Primary: OOS win rate quality
        + weight_bps * oos_avg_bps               # Secondary: OOS returns
        - 0.60 * win_rate_degradation           # Heavy penalty for win rate overfitting
        - 0.08 * bps_degradation                # Light penalty for return degradation
        + win_rate_surprise                      # Light reward for OOS surprise
        + bps_surprise                           # Light reward for OOS return surprise
        + 2.0 * np.log1p(total_events)          # Bonus for larger event sample
    )
    
    return score


def _prepare_window_frame(
    df: pd.DataFrame,
    window: int,
    hold_horizons: tuple[int, ...] | list[int],
    session_start: str,
    session_end: str,
) -> pd.DataFrame:
    """Pre-compute reusable metrics for a single window size."""
    temp_df = df[["btc_ret", "nq_ret", "nq_close"]].copy()
    temp_df["corr_spearman"] = fast_rolling_spearman(
        temp_df["btc_ret"].to_numpy(),
        temp_df["nq_ret"].to_numpy(),
        window,
    )
    direction_match = (np.sign(temp_df["btc_ret"]) == np.sign(temp_df["nq_ret"])).astype(float)
    temp_df["hit_ratio"] = direction_match.rolling(window=window).mean()
    temp_df["time_filter"] = (
        (temp_df.index.time >= pd.to_datetime(session_start).time())
        & (temp_df.index.time <= pd.to_datetime(session_end).time())
    )
    temp_df["mom_window"] = temp_df["nq_close"] / temp_df["nq_close"].shift(window) - 1
    temp_df["regime_dir"] = np.sign(temp_df["mom_window"]).replace(0, 1)
    rolling_std = temp_df["nq_ret"].rolling(window=window).std()
    temp_df["std_filter"] = (rolling_std >= rolling_std.quantile(0.20)) & (
        rolling_std <= rolling_std.quantile(0.80)
    )

    for horizon in hold_horizons:
        temp_df[f"nq_fwd_{horizon}"] = temp_df["nq_close"].shift(-horizon) / temp_df["nq_close"] - 1

    return temp_df.dropna(subset=["corr_spearman", "hit_ratio", "mom_window"]).copy()


def run_sensitivity_grid_search(
    df: pd.DataFrame,
    test_windows: tuple[int, ...] | list[int],
    hold_horizons: tuple[int, ...] | list[int],
    hit_ratio_threshold: float = 0.70,
    spearman_threshold: float = 0.60,
    session_start: str = "10:30",
    session_end: str = "16:00",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate signal robustness across metric windows and holding periods."""
    results_win_rate: dict[int, dict[str, float]] = {}
    results_bps: dict[int, dict[str, float]] = {}
    event_counts: dict[int, int] = {}
    median_bps_rows: dict[int, dict[str, float]] = {}

    for window in test_windows:
        temp_df = _prepare_window_frame(
            df=df,
            window=window,
            hold_horizons=hold_horizons,
            session_start=session_start,
            session_end=session_end,
        )
        temp_df["valid_signal"] = (
            temp_df["time_filter"]
            & (temp_df["hit_ratio"] >= hit_ratio_threshold)
            & (temp_df["corr_spearman"] >= spearman_threshold)
            & temp_df["std_filter"]
        )
        previous_signal = temp_df["valid_signal"].shift(1, fill_value=False)
        temp_df["event_start"] = temp_df["valid_signal"] & (~previous_signal)

        events = temp_df[temp_df["event_start"]].copy()
        event_counts[window] = int(len(events))

        win_rate_row: dict[str, float] = {}
        bps_row: dict[str, float] = {}
        median_bps_row: dict[str, float] = {}
        for horizon in hold_horizons:
            directional_return = events[f"nq_fwd_{horizon}"] * events["regime_dir"]
            win_rate_row[f"Hold_{horizon}m"] = (directional_return > 0).mean() * 100
            bps_row[f"Hold_{horizon}m"] = directional_return.mean() * 10000
            median_bps_row[f"Hold_{horizon}m"] = directional_return.median() * 10000

        results_win_rate[window] = win_rate_row
        results_bps[window] = bps_row
        median_bps_rows[window] = median_bps_row

    win_rate_matrix = pd.DataFrame.from_dict(results_win_rate, orient="index")
    win_rate_matrix.index.name = "Window_Bars"
    win_rate_matrix.insert(0, "Total_Events", pd.Series(event_counts))

    bps_matrix = pd.DataFrame.from_dict(results_bps, orient="index")
    bps_matrix.index.name = "Window_Bars"
    bps_matrix.insert(0, "Total_Events", pd.Series(event_counts))

    median_bps_matrix = pd.DataFrame.from_dict(median_bps_rows, orient="index")
    median_bps_matrix.index.name = "Window_Bars"
    median_bps_matrix.insert(0, "Total_Events", pd.Series(event_counts))

    return win_rate_matrix, bps_matrix, median_bps_matrix


def _directional_summary(directional_return: pd.Series) -> dict[str, float]:
    """Summarize directional forward returns in trading-friendly units."""
    clean = directional_return.dropna()
    count = int(clean.shape[0])
    if count == 0:
        return {
            "count": 0,
            "win_rate_pct": np.nan,
            "avg_bps": np.nan,
            "median_bps": np.nan,
        }
    return {
        "count": count,
        "win_rate_pct": clean.gt(0).mean() * 100,
        "avg_bps": clean.mean() * 10000,
        "median_bps": clean.median() * 10000,
    }


def run_robustness_grid_search(
    df: pd.DataFrame,
    test_windows: tuple[int, ...] | list[int],
    hold_horizons: tuple[int, ...] | list[int],
    hit_ratio_thresholds: tuple[float, ...] | list[float],
    spearman_thresholds: tuple[float, ...] | list[float],
    session_start: str = "10:30",
    session_end: str = "16:00",
    split_ratio: float = 0.70,
    min_events_per_side: int = 25,
) -> pd.DataFrame:
    """Run a broader parameter sweep and score configurations on OOS stability."""
    rows: list[dict[str, float | int | bool]] = []

    for window in test_windows:
        temp_df = _prepare_window_frame(
            df=df,
            window=window,
            hold_horizons=hold_horizons,
            session_start=session_start,
            session_end=session_end,
        )
        split_index = int(len(temp_df) * split_ratio)
        split_timestamp = temp_df.index[split_index]

        for hit_ratio_threshold in hit_ratio_thresholds:
            for spearman_threshold in spearman_thresholds:
                valid_signal = (
                    temp_df["time_filter"]
                    & (temp_df["hit_ratio"] >= hit_ratio_threshold)
                    & (temp_df["corr_spearman"] >= spearman_threshold)
                    & temp_df["std_filter"]
                )
                event_start = valid_signal & (~valid_signal.shift(1, fill_value=False))
                events = temp_df.loc[event_start].copy()

                if events.empty:
                    continue

                train_events = events[events.index < split_timestamp].copy()
                test_events = events[events.index >= split_timestamp].copy()

                for horizon in hold_horizons:
                    train_directional = train_events[f"nq_fwd_{horizon}"] * train_events["regime_dir"]
                    test_directional = test_events[f"nq_fwd_{horizon}"] * test_events["regime_dir"]
                    all_directional = events[f"nq_fwd_{horizon}"] * events["regime_dir"]

                    train_count = int(train_directional.notna().sum())
                    test_count = int(test_directional.notna().sum())
                    total_count = int(all_directional.notna().sum())
                    enough_events = (
                        train_count >= min_events_per_side and test_count >= min_events_per_side
                    )

                    is_win_rate = train_directional.gt(0).mean() * 100 if train_count else np.nan
                    oos_win_rate = test_directional.gt(0).mean() * 100 if test_count else np.nan
                    is_avg_bps = train_directional.mean() * 10000 if train_count else np.nan
                    oos_avg_bps = test_directional.mean() * 10000 if test_count else np.nan
                    is_median_bps = train_directional.median() * 10000 if train_count else np.nan
                    oos_median_bps = test_directional.median() * 10000 if test_count else np.nan

                    # Calculate directional gaps (asymmetric: reward OOS surprises)
                    win_rate_gap = is_win_rate - oos_win_rate if enough_events else np.nan
                    bps_gap = is_avg_bps - oos_avg_bps if enough_events else np.nan
                    
                    # Use unified robustness score
                    robust_score = compute_unified_robustness_score(
                        oos_win_rate=oos_win_rate,
                        oos_avg_bps=oos_avg_bps,
                        is_win_rate=is_win_rate,
                        is_avg_bps=is_avg_bps,
                        total_events=total_count,
                        weight_win_rate=1.0,
                        weight_bps=0.35,
                    ) if enough_events else np.nan

                    rows.append(
                        {
                            "window_bars": window,
                            "hold_minutes": horizon,
                            "hit_ratio_threshold": hit_ratio_threshold,
                            "spearman_threshold": spearman_threshold,
                            "train_events": train_count,
                            "test_events": test_count,
                            "total_events": total_count,
                            "passes_min_events": enough_events,
                            "is_win_rate_pct": is_win_rate,
                            "oos_win_rate_pct": oos_win_rate,
                            "is_avg_bps": is_avg_bps,
                            "oos_avg_bps": oos_avg_bps,
                            "is_median_bps": is_median_bps,
                            "oos_median_bps": oos_median_bps,
                            "win_rate_gap": win_rate_gap,
                            "bps_gap": bps_gap,
                            "robust_score": robust_score,
                        }
                    )

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results["rank_oos_bps"] = results["oos_avg_bps"].rank(ascending=False, method="dense")
    results["rank_robust"] = results["robust_score"].rank(ascending=False, method="dense")
    return results.sort_values(
        by=["passes_min_events", "robust_score", "oos_avg_bps", "oos_win_rate_pct", "total_events"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def run_walk_forward_grid_search(
    df: pd.DataFrame,
    test_windows: tuple[int, ...] | list[int],
    hold_horizons: tuple[int, ...] | list[int],
    hit_ratio_thresholds: tuple[float, ...] | list[float],
    spearman_thresholds: tuple[float, ...] | list[float],
    session_start: str = "10:30",
    session_end: str = "16:00",
    min_train_months: int = 4,
    min_train_events: int = 40,
    min_test_events: int = 10,
    min_valid_folds: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate configurations with expanding-train, monthly walk-forward OOS folds."""
    summary_rows: list[dict[str, float | int | bool]] = []
    fold_rows: list[dict[str, float | int | bool | str]] = []

    for window in test_windows:
        temp_df = _prepare_window_frame(
            df=df,
            window=window,
            hold_horizons=hold_horizons,
            session_start=session_start,
            session_end=session_end,
        )
        month_periods = temp_df.index.tz_localize(None).to_period("M")
        unique_months = month_periods.unique().sort_values()

        if len(unique_months) <= min_train_months:
            continue

        for hit_ratio_threshold in hit_ratio_thresholds:
            for spearman_threshold in spearman_thresholds:
                valid_signal = (
                    temp_df["time_filter"]
                    & (temp_df["hit_ratio"] >= hit_ratio_threshold)
                    & (temp_df["corr_spearman"] >= spearman_threshold)
                    & temp_df["std_filter"]
                )
                event_start = valid_signal & (~valid_signal.shift(1, fill_value=False))
                events = temp_df.loc[event_start].copy()
                if events.empty:
                    continue

                event_months = events.index.tz_localize(None).to_period("M")

                for horizon in hold_horizons:
                    config_fold_rows: list[dict[str, float | int | bool | str]] = []

                    for month_idx in range(min_train_months, len(unique_months)):
                        test_month = unique_months[month_idx]
                        train_mask = event_months < test_month
                        test_mask = event_months == test_month

                        train_directional = (
                            events.loc[train_mask, f"nq_fwd_{horizon}"] * events.loc[train_mask, "regime_dir"]
                        )
                        test_directional = (
                            events.loc[test_mask, f"nq_fwd_{horizon}"] * events.loc[test_mask, "regime_dir"]
                        )

                        train_stats = _directional_summary(train_directional)
                        test_stats = _directional_summary(test_directional)
                        fold_is_valid = (
                            train_stats["count"] >= min_train_events
                            and test_stats["count"] >= min_test_events
                        )

                        fold_row = {
                            "window_bars": window,
                            "hold_minutes": horizon,
                            "hit_ratio_threshold": hit_ratio_threshold,
                            "spearman_threshold": spearman_threshold,
                            "test_month": str(test_month),
                            "train_events": train_stats["count"],
                            "test_events": test_stats["count"],
                            "train_win_rate_pct": train_stats["win_rate_pct"],
                            "oos_win_rate_pct": test_stats["win_rate_pct"],
                            "train_avg_bps": train_stats["avg_bps"],
                            "oos_avg_bps": test_stats["avg_bps"],
                            "train_median_bps": train_stats["median_bps"],
                            "oos_median_bps": test_stats["median_bps"],
                            "fold_is_valid": fold_is_valid,
                        }
                        fold_rows.append(fold_row)

                        if fold_is_valid:
                            config_fold_rows.append(fold_row)

                    total_possible_folds = len(unique_months) - min_train_months
                    if not config_fold_rows:
                        summary_rows.append(
                            {
                                "window_bars": window,
                                "hold_minutes": horizon,
                                "hit_ratio_threshold": hit_ratio_threshold,
                                "spearman_threshold": spearman_threshold,
                                "valid_folds": 0,
                                "total_possible_folds": total_possible_folds,
                                "fold_coverage_pct": 0.0,
                                "avg_train_events": np.nan,
                                "avg_test_events": np.nan,
                                "total_test_events": 0,
                                "weighted_oos_win_rate_pct": np.nan,
                                "weighted_oos_avg_bps": np.nan,
                                "weighted_oos_median_bps": np.nan,
                                "mean_monthly_oos_bps": np.nan,
                                "median_monthly_oos_bps": np.nan,
                                "oos_bps_std": np.nan,
                                "positive_month_pct": np.nan,
                                "best_month_bps": np.nan,
                                "worst_month_bps": np.nan,
                                "mean_win_rate_gap": np.nan,
                                "mean_bps_gap": np.nan,
                                "passes_walk_forward": False,
                                "walk_forward_score": np.nan,
                            }
                        )
                        continue

                    config_folds = pd.DataFrame(config_fold_rows)
                    weights = config_folds["test_events"].to_numpy()
                    avg_train_events = config_folds["train_events"].mean()
                    avg_test_events = config_folds["test_events"].mean()
                    total_test_events = int(config_folds["test_events"].sum())
                    valid_folds = int(len(config_folds))
                    fold_coverage_pct = (valid_folds / total_possible_folds) * 100 if total_possible_folds else 0.0

                    weighted_oos_win_rate = np.average(config_folds["oos_win_rate_pct"], weights=weights)
                    weighted_oos_avg_bps = np.average(config_folds["oos_avg_bps"], weights=weights)
                    weighted_oos_median_bps = np.average(config_folds["oos_median_bps"], weights=weights)
                    mean_monthly_oos_bps = config_folds["oos_avg_bps"].mean()
                    median_monthly_oos_bps = config_folds["oos_avg_bps"].median()
                    oos_bps_std = config_folds["oos_avg_bps"].std(ddof=0)
                    positive_month_pct = config_folds["oos_avg_bps"].gt(0).mean() * 100
                    best_month_bps = config_folds["oos_avg_bps"].max()
                    worst_month_bps = config_folds["oos_avg_bps"].min()
                    mean_win_rate_gap = (
                        (config_folds["train_win_rate_pct"] - config_folds["oos_win_rate_pct"]).abs().mean()
                    )
                    mean_bps_gap = (
                        (config_folds["train_avg_bps"] - config_folds["oos_avg_bps"]).abs().mean()
                    )
                    passes_walk_forward = valid_folds >= min_valid_folds
                    
                    # Calculate mean training metrics for walk-forward unified score
                    mean_train_win_rate = config_folds["train_win_rate_pct"].mean()
                    mean_train_bps = config_folds["train_avg_bps"].mean()
                    
                    # Use unified robustness score adapted for walk-forward (averaged across folds)
                    walk_forward_score = compute_unified_robustness_score(
                        oos_win_rate=weighted_oos_win_rate,
                        oos_avg_bps=weighted_oos_avg_bps,
                        is_win_rate=mean_train_win_rate,
                        is_avg_bps=mean_train_bps,
                        total_events=total_test_events,
                        weight_win_rate=1.0,
                        weight_bps=0.45,  # Slightly higher emphasis on returns for walk-forward
                    ) if passes_walk_forward else np.nan

                    summary_rows.append(
                        {
                            "window_bars": window,
                            "hold_minutes": horizon,
                            "hit_ratio_threshold": hit_ratio_threshold,
                            "spearman_threshold": spearman_threshold,
                            "valid_folds": valid_folds,
                            "total_possible_folds": total_possible_folds,
                            "fold_coverage_pct": fold_coverage_pct,
                            "avg_train_events": avg_train_events,
                            "avg_test_events": avg_test_events,
                            "total_test_events": total_test_events,
                            "weighted_oos_win_rate_pct": weighted_oos_win_rate,
                            "weighted_oos_avg_bps": weighted_oos_avg_bps,
                            "weighted_oos_median_bps": weighted_oos_median_bps,
                            "mean_monthly_oos_bps": mean_monthly_oos_bps,
                            "median_monthly_oos_bps": median_monthly_oos_bps,
                            "oos_bps_std": oos_bps_std,
                            "positive_month_pct": positive_month_pct,
                            "best_month_bps": best_month_bps,
                            "worst_month_bps": worst_month_bps,
                            "mean_win_rate_gap": mean_win_rate_gap,
                            "mean_bps_gap": mean_bps_gap,
                            "passes_walk_forward": passes_walk_forward,
                            "walk_forward_score": walk_forward_score,
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    if summary_df.empty:
        return summary_df, fold_df

    summary_df["rank_walk_forward"] = summary_df["walk_forward_score"].rank(
        ascending=False,
        method="dense",
    )
    summary_df["rank_weighted_oos_bps"] = summary_df["weighted_oos_avg_bps"].rank(
        ascending=False,
        method="dense",
    )
    summary_df = summary_df.sort_values(
        by=[
            "passes_walk_forward",
            "walk_forward_score",
            "weighted_oos_avg_bps",
            "positive_month_pct",
            "total_test_events",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return summary_df, fold_df
