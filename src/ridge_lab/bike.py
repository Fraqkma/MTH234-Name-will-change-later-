"""Capital Bikeshare hourly data and fixed chronological evaluation splits."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

HOUR_CSV_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
DEFAULT_HOUR_CSV = Path(__file__).resolve().parents[2] / "data" / "bike" / "hour.csv"
EXPECTED_COLUMNS = (
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
)


@dataclass(frozen=True)
class BikeData:
    """Only approved predictors, target, and split metadata; arrays are read-only."""

    dates: tuple[date, ...]
    hours: NDArray[np.int64]
    working_days: NDArray[np.int64]
    counts: NDArray[np.float64]
    row_ids: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class BikeFold:
    """One expanding training window followed by a disjoint validation window."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date


@dataclass(frozen=True)
class BikeSplit:
    """Shared development folds and sealed future holdout membership."""

    development: tuple[int, ...]
    holdout: tuple[int, ...]
    folds: tuple[BikeFold, ...]
    holdout_start: date
    holdout_end: date


def load_bike_hourly(path: Path = DEFAULT_HOUR_CSV) -> BikeData:
    """Load UCI hour.csv, verifying source bytes and the approved raw schema."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != HOUR_CSV_SHA256:
        raise ValueError(f"hour.csv SHA-256 mismatch: {digest}")
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError("hour.csv columns differ from the approved UCI schema")
        records: list[tuple[date, int, int, float]] = []
        for line, row in enumerate(reader, start=2):
            try:
                day = date.fromisoformat(row["dteday"])
                hour = int(row["hr"])
                working = int(row["workingday"])
                count = int(row["cnt"]) if day < date(2012, 8, 8) else float("nan")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid bike row at CSV line {line}") from exc
            if not 0 <= hour <= 23 or working not in (0, 1) or count < 0:
                raise ValueError(f"out-of-range bike value at CSV line {line}")
            records.append((day, hour, working, float(count)))
    if not records:
        raise ValueError("hour.csv contains no observations")
    records.sort(key=lambda item: (item[0], item[1]))
    row_ids = tuple(f"{day.isoformat()}:{hour:02d}" for day, hour, _, _ in records)
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("hour.csv has duplicate date-hour observations")
    hours = np.array([row[1] for row in records], dtype=np.int64)
    working_days = np.array([row[2] for row in records], dtype=np.int64)
    counts = np.array([row[3] for row in records], dtype=np.float64)
    for values in (hours, working_days, counts):
        values.setflags(write=False)
    return BikeData(
        dates=tuple(row[0] for row in records),
        hours=hours,
        working_days=working_days,
        counts=counts,
        row_ids=row_ids,
        sha256=digest,
    )


def make_bike_split(data: BikeData) -> BikeSplit:
    """Seal the last 20% of dates and make five expanding development folds."""
    dates = sorted(set(data.dates))
    if len(dates) < 12:
        raise ValueError("at least 12 dates are required for the bike split")
    holdout_days = round(0.2 * len(dates))
    development_dates = dates[:-holdout_days]
    holdout_dates = dates[-holdout_days:]
    blocks = np.array_split(np.array(development_dates, dtype=object), 6)
    if any(len(block) == 0 for block in blocks):
        raise ValueError("all development windows must contain dates")
    indices_by_date: dict[date, list[int]] = {}
    for index, day in enumerate(data.dates):
        indices_by_date.setdefault(day, []).append(index)

    def indices(days: list[date]) -> tuple[int, ...]:
        return tuple(index for day in days for index in indices_by_date[day])

    folds = []
    for number in range(5):
        train_days = [day for block in blocks[: number + 1] for day in block]
        validation_days = list(blocks[number + 1])
        folds.append(BikeFold(
            train=indices(train_days),
            validation=indices(validation_days),
            train_start=train_days[0],
            train_end=train_days[-1],
            validation_start=validation_days[0],
            validation_end=validation_days[-1],
        ))
    return BikeSplit(
        development=indices(development_dates),
        holdout=indices(holdout_dates),
        folds=tuple(folds),
        holdout_start=holdout_dates[0],
        holdout_end=holdout_dates[-1],
    )
