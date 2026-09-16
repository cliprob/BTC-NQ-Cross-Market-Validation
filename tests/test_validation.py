from __future__ import annotations

import numpy as np
import pandas as pd

from cross_market.features import CROSS_FEATURES
from cross_market.models import model_specs, walk_forward_predictions
from cross_market.validation import assert_purged, expanding_folds


def event_frame() -> pd.DataFrame:
    times = pd.date_range("2024-01-02 10:00", "2025-12-30 10:00", freq="2D", tz="America/New_York")
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"entry_time": times, "exit_time": times + pd.Timedelta(minutes=5)})
    for feature in CROSS_FEATURES:
        frame[feature] = rng.normal(size=len(frame))
    score = frame["nq_mom_15_bps"] + 0.3 * frame["btc_mom_15_bps"]
    frame["target"] = (score > 0).astype(int)
    frame["gross_usd"] = np.where(frame["target"], 5.0, -5.0)
    frame["net_usd"] = frame["gross_usd"] - 3.0
    frame["net_bps"] = frame["net_usd"] / 4
    return frame


def test_splits_are_deterministic_and_purged(config) -> None:
    events = event_frame()
    first = list(expanding_folds(events, config["research"]["purge_minutes"]))
    second = list(expanding_folds(events, config["research"]["purge_minutes"]))
    assert first == second
    assert first
    for fold in first:
        assert_purged(events, fold, config["research"]["purge_minutes"])


def test_future_rows_do_not_change_prior_oof_predictions(config) -> None:
    events = event_frame()
    spec = model_specs(config)[0]
    base, _ = walk_forward_predictions(events, spec, config)
    modified = events.copy()
    mask = modified["entry_time"] >= pd.Timestamp("2025-10-01", tz="America/New_York")
    modified.loc[mask, spec.features] = 1_000_000
    replay, _ = walk_forward_predictions(modified, spec, config)
    before = base["entry_time"] < pd.Timestamp("2025-10-01", tz="America/New_York")
    matching = replay["entry_time"] < pd.Timestamp("2025-10-01", tz="America/New_York")
    assert np.allclose(base.loc[before, "probability"], replay.loc[matching, "probability"])
