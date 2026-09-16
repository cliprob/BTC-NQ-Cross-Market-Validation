"""Market-data loaders with explicit timezone and continuity handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def _timestamp(series: pd.Series, source_timezone: str | None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        unit = "ms" if numeric.dropna().median() > 10**11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        if source_timezone is None:
            raise ValueError("Naive timestamps require a source timezone")
        parsed = parsed.dt.tz_localize(source_timezone, ambiguous="NaT", nonexistent="shift_forward")
    return parsed.dt.tz_convert("UTC")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    aliases = {
        "open_time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "symbol": "contract",
    }
    return frame.rename(columns={key: value for key, value in aliases.items() if key in frame})


def load_btc(path: str | Path) -> pd.DataFrame:
    frame = _normalise_columns(pd.read_csv(path))
    if "timestamp" not in frame:
        raise ValueError("BTC data requires open_time, date, datetime, or timestamp")
    frame["timestamp"] = _timestamp(frame["timestamp"], None)
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"BTC data missing columns: {sorted(missing)}")
    result = frame[["timestamp", *required]].rename(columns={name: f"btc_{name}" for name in required})
    return result.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")


def load_nq(path: str | Path, source_timezone: str) -> pd.DataFrame:
    frame = _normalise_columns(pd.read_csv(path))
    if "timestamp" not in frame:
        raise ValueError("NQ data requires date, datetime, or timestamp")
    frame["timestamp"] = _timestamp(frame["timestamp"], source_timezone)
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"NQ data missing columns: {sorted(missing)}")
    if "contract" not in frame:
        frame["contract"] = "UNKNOWN"
    result = frame[["timestamp", *required, "contract"]].rename(
        columns={name: f"nq_{name}" for name in required} | {"contract": "nq_contract"}
    )
    return result.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")


def load_market_data(config: dict[str, Any]) -> pd.DataFrame:
    btc = load_btc(config["data"]["btc_path"])
    nq = load_nq(config["data"]["nq_path"], config["data"]["nq_source_timezone"])
    merged = nq.merge(btc, on="timestamp", how="inner", validate="one_to_one").sort_values("timestamp")
    analysis_tz = config["data"]["analysis_timezone"]
    merged["timestamp"] = merged["timestamp"].dt.tz_convert(analysis_tz)
    gap = merged["timestamp"].diff().ne(pd.Timedelta(minutes=1))
    roll = merged["nq_contract"].ne(merged["nq_contract"].shift())
    merged["continuity_break"] = (gap | roll).astype(bool)
    merged.loc[merged.index[0], "continuity_break"] = True
    numeric = [column for column in merged if column.startswith(("btc_", "nq_")) and column != "nq_contract"]
    merged[numeric] = merged[numeric].replace([np.inf, -np.inf], np.nan)
    return merged.reset_index(drop=True)


def continuous_window(frame: pd.DataFrame, start: int, end: int) -> bool:
    """Whether a positional interval has one-minute bars and a single contract."""
    if start < 0 or end >= len(frame) or end <= start:
        return False
    interval = frame.iloc[start : end + 1]
    if interval["continuity_break"].iloc[1:].any():
        return False
    return bool(interval["nq_contract"].nunique() == 1)


def cme_trading_dates(start: pd.Timestamp, end: pd.Timestamp) -> set[object]:
    """CME Equity session labels, used to exclude weekends and exchange holidays."""
    calendar = mcal.get_calendar("CME_Equity")
    valid = calendar.valid_days(start_date=start.date(), end_date=end.date())
    return set(valid.date)
