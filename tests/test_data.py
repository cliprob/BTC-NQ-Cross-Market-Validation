from __future__ import annotations

import pandas as pd

from cross_market.data import cme_trading_dates, continuous_window, load_nq


def test_dst_aware_source_conversion(tmp_path) -> None:
    path = tmp_path / "nq.csv"
    pd.DataFrame(
        {
            "date": ["2025-03-07 08:30:00", "2025-03-10 08:30:00"],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [1, 1],
            "contract": ["NQH5", "NQH5"],
        }
    ).to_csv(path, index=False)
    loaded = load_nq(path, "America/Chicago")
    utc_hours = loaded["timestamp"].dt.hour.tolist()
    assert utc_hours == [14, 13]


def test_cme_calendar_excludes_weekend() -> None:
    dates = cme_trading_dates(pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-06"))
    assert pd.Timestamp("2025-01-04").date() not in dates
    assert pd.Timestamp("2025-01-06").date() in dates


def test_roll_and_gap_invalidate_execution_window() -> None:
    frame = pd.DataFrame(
        {
            "continuity_break": [True, False, True, False],
            "nq_contract": ["NQH5", "NQH5", "NQM5", "NQM5"],
        }
    )
    assert continuous_window(frame, 0, 1)
    assert not continuous_window(frame, 0, 3)
