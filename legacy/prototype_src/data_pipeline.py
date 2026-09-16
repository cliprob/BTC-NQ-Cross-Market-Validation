from pathlib import Path

import pandas as pd

try:
    from .config import BTC_DATA_FILE, NQ_DATA_FILE
except ImportError:
    from config import BTC_DATA_FILE, NQ_DATA_FILE


def _standardize_market_data(path: str | Path, suffix: str) -> pd.DataFrame:
    """Load a market CSV and normalize schema to a shared UTC index."""
    df = pd.read_csv(path)
    df.columns = [column.strip().lower() for column in df.columns]

    if "open time" in df.columns:
        df["datetime"] = pd.to_datetime(df["open time"], utc=True)
    elif "date" in df.columns:
        df["datetime"] = (
            pd.to_datetime(df["date"], format="%Y%m%d %H:%M:%S")
            .dt.tz_localize(
                "America/Chicago",
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
            .dt.tz_convert("UTC")
        )
    else:
        raise ValueError(f"Unsupported schema for {path}.")

    rename_map = {
        "open": f"{suffix}_open",
        "high": f"{suffix}_high",
        "low": f"{suffix}_low",
        "close": f"{suffix}_close",
        "volume": f"{suffix}_volume",
    }
    keep_columns = ["datetime", *rename_map.keys()]
    df = df[keep_columns].rename(columns=rename_map)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df.index.name = "datetime"

    numeric_columns = [column for column in df.columns if column.endswith(("open", "high", "low", "close", "volume"))]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def load_market_data(
    btc_path: str | Path = BTC_DATA_FILE,
    nq_path: str | Path = NQ_DATA_FILE,
    how: str = "inner",
) -> pd.DataFrame:
    """Load BTC and NQ minute data, align on UTC timestamps, and add returns."""
    btc = _standardize_market_data(btc_path, "btc")
    nq = _standardize_market_data(nq_path, "nq")

    df = nq.join(btc, how=how)
    df["btc_ret"] = df["btc_close"].pct_change()
    df["nq_ret"] = df["nq_close"].pct_change()
    return df.dropna(subset=["btc_ret", "nq_ret"]).copy()


def convert_index_timezone(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Return a copy with its DatetimeIndex converted to a target timezone."""
    converted = df.copy()
    converted.index = converted.index.tz_convert(timezone)
    return converted


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach commonly used time breakdown columns without changing the index."""
    enriched = df.copy()
    enriched["hour"] = enriched.index.hour
    enriched["minute"] = enriched.index.minute
    enriched["date"] = enriched.index.date
    enriched["dayofweek"] = enriched.index.dayofweek
    return enriched


def get_train_test_split(df: pd.DataFrame, split_ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe chronologically."""
    split_index = int(len(df) * split_ratio)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()
