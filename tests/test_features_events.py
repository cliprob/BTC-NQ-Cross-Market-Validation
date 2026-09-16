from __future__ import annotations

import numpy as np
import pandas as pd

from cross_market.events import assert_no_overlap, build_events, round_trip_cost_usd
from cross_market.features import CROSS_FEATURES, build_features, rolling_spearman
from tests.conftest import synthetic_market


def test_features_do_not_use_future(config) -> None:
    market = synthetic_market()
    original = build_features(market, config)
    changed = market.copy()
    cutoff = 1_000
    changed.loc[cutoff + 1 :, "btc_close"] *= 10
    replay = build_features(changed, config)
    assert np.allclose(
        original.loc[:cutoff, CROSS_FEATURES], replay.loc[:cutoff, CROSS_FEATURES], equal_nan=True
    )


def test_next_open_costs_and_no_overlap(config) -> None:
    features = build_features(synthetic_market(days=15), config)
    events = build_events(features, config, impulse_threshold=0.1)
    assert not events.empty
    assert (events["entry_time"] == events["signal_time"] + np.timedelta64(1, "m")).all()
    assert (events["exit_time"] == events["entry_time"] + np.timedelta64(5, "m")).all()
    assert np.allclose(events["net_usd"], events["gross_usd"] - round_trip_cost_usd(config))
    assert round_trip_cost_usd(config) == 3.0
    assert_no_overlap(events)


def test_rolling_spearman_is_rank_based() -> None:
    left = np.arange(20, dtype=float)
    right = left**3
    correlation = rolling_spearman(pd.Series(left), pd.Series(right), window=10)
    assert np.isclose(correlation.iloc[-1], 1.0)


def test_features_reset_after_gap(config) -> None:
    market = synthetic_market(days=2)
    features = build_features(market, config)
    second_day = features["timestamp"].dt.date.iloc[-1]
    opening = features[features["timestamp"].dt.date == second_day].iloc[:30]
    assert opening["nq_mom_30_bps"].isna().all()
