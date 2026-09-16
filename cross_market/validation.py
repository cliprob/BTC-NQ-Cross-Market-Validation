"""Purged expanding walk-forward split definitions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def expanding_folds(events: pd.DataFrame, purge_minutes: int) -> Iterator[Fold]:
    if events.empty:
        return
    times = pd.DatetimeIndex(events["entry_time"])
    first = max(times.min() + pd.DateOffset(months=6), pd.Timestamp("2024-10-01", tz=times.tz))
    boundaries = pd.date_range(first.normalize(), pd.Timestamp("2026-01-01", tz=times.tz), freq="QS")
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        purge_cutoff = start - pd.Timedelta(minutes=purge_minutes)
        train = np_flatnonzero(times < purge_cutoff)
        valid = np_flatnonzero((times >= start) & (times < end))
        if len(train) and len(valid):
            yield Fold(
                name=f"{start.year}Q{(start.month - 1) // 3 + 1}",
                train_indices=tuple(train),
                validation_indices=tuple(valid),
                validation_start=start,
                validation_end=end,
            )


def np_flatnonzero(mask: Iterable[object]) -> list[int]:
    return [index for index, value in enumerate(mask) if bool(value)]


def assert_purged(events: pd.DataFrame, fold: Fold, purge_minutes: int) -> None:
    latest_train_exit = events.iloc[list(fold.train_indices)]["exit_time"].max()
    boundary = fold.validation_start
    if latest_train_exit >= boundary:
        raise AssertionError("Training labels enter the purge interval")
