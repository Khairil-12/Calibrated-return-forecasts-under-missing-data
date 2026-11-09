"""Walk-forward splits with purge gap for CalibMiss."""

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitIndices:
    """Indices for a single walk-forward split."""

    train_idx: np.ndarray[tuple[int, ...], np.dtype[np.int_]]
    calibrate_idx: np.ndarray[tuple[int, ...], np.dtype[np.int_]]
    test_idx: np.ndarray[tuple[int, ...], np.dtype[np.int_]]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibrate_start: pd.Timestamp
    calibrate_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _years_to_bdays(years: int) -> int:
    """Convert years to business days (approx 252 trading days/year)."""
    return years * 252


def _months_to_bdays(months: int) -> int:
    """Convert months to business days (approx 21 trading days/month)."""
    return months * 21


def walk_forward_splits(
    n_samples: int,
    train_years: int,
    calibrate_months: int,
    test_months: int,
    roll_forward_months: int,
    purge_days: int,
    freq: str = "B",
) -> Generator[
    tuple[
        np.ndarray[tuple[int, ...], np.dtype[np.int_]],
        np.ndarray[tuple[int, ...], np.dtype[np.int_]],
        np.ndarray[tuple[int, ...], np.dtype[np.int_]],
    ],
    None,
    None,
]:
    """Generate walk-forward splits with purge gaps.

    Args:
        n_samples: Total number of samples
        train_years: Training window in years
        calibrate_months: Calibration window in months
        test_months: Test window in months
        roll_forward_months: How much to roll forward each fold (months)
        purge_days: Number of days to purge between train/cal and cal/test
        freq: Frequency ("B" for business days)

    Yields:
        Tuples of (train_indices, calibrate_indices, test_indices)
    """
    train_size = _years_to_bdays(train_years)
    calibrate_size = _months_to_bdays(calibrate_months)
    test_size = _months_to_bdays(test_months)
    roll_forward = _months_to_bdays(roll_forward_months)

    # First fold: train starts at 0
    train_start = 0

    while True:
        train_end = train_start + train_size
        cal_start = train_end + purge_days
        cal_end = cal_start + calibrate_size
        test_start = cal_end + purge_days
        test_end = test_start + test_size

        # Check if we have enough data for all windows
        if test_end > n_samples:
            break

        train_idx = np.arange(train_start, train_end)
        cal_idx = np.arange(cal_start, cal_end)
        test_idx = np.arange(test_start, test_end)

        yield train_idx, cal_idx, test_idx

        # Roll forward
        train_start += roll_forward


def get_split_indices(dates: pd.DatetimeIndex, splits_config: dict[str, int]) -> list[SplitIndices]:
    """Generate walk-forward splits using actual dates.

    Args:
        dates: DatetimeIndex of all dates in the dataset
        splits_config: Dict with keys: train_years, calibrate_months,
            test_months, roll_forward_months, purge_days

    Returns:
        List of SplitIndices objects
    """
    train_years = splits_config.get("train_years", 3)
    calibrate_months = splits_config.get("calibrate_months", 6)
    test_months = splits_config.get("test_months", 6)
    roll_forward_months = splits_config.get("roll_forward_months", 6)
    purge_days = splits_config.get("purge_days", 5)

    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be sorted and unique")
    if min(train_years, calibrate_months, test_months, roll_forward_months) <= 0:
        raise ValueError("window lengths must be positive")
    if purge_days < 0:
        raise ValueError("purge_days must be non-negative")

    splits: list[SplitIndices] = []
    train_start = dates[0]
    last_date = dates[-1]
    train_offset = pd.DateOffset(years=train_years)
    calibrate_offset = pd.DateOffset(months=calibrate_months)
    test_offset = pd.DateOffset(months=test_months)
    roll_offset = pd.DateOffset(months=roll_forward_months)

    while train_start + train_offset <= last_date:
        train_end_position = int(dates.searchsorted(train_start + train_offset, side="left"))
        train_positions = np.arange(int(dates.searchsorted(train_start)), train_end_position)
        cal_start_position = train_end_position + purge_days
        cal_start_date = train_start + train_offset
        cal_end_date = cal_start_date + calibrate_offset
        cal_end_position = int(dates.searchsorted(cal_end_date, side="left"))
        cal_positions = np.arange(cal_start_position, cal_end_position)
        test_start_position = cal_end_position + purge_days
        test_end_date = cal_end_date + test_offset
        test_end_position = int(dates.searchsorted(test_end_date, side="left"))
        test_positions = np.arange(test_start_position, test_end_position)

        if (
            test_end_date > last_date
            or not train_positions.size
            or not cal_positions.size
            or not test_positions.size
        ):
            break

        splits.append(
            SplitIndices(
                train_idx=train_positions,
                calibrate_idx=cal_positions,
                test_idx=test_positions,
                train_start=dates[train_positions[0]],
                train_end=dates[train_positions[-1]],
                calibrate_start=dates[cal_positions[0]],
                calibrate_end=dates[cal_positions[-1]],
                test_start=dates[test_positions[0]],
                test_end=dates[test_positions[-1]],
            )
        )
        train_start += roll_offset

    return splits
