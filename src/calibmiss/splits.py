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
    freq: str = "B"
) -> Generator[
    tuple[
        np.ndarray[tuple[int, ...], np.dtype[np.int_]],
        np.ndarray[tuple[int, ...], np.dtype[np.int_]],
        np.ndarray[tuple[int, ...], np.dtype[np.int_]]
    ],
    None,
    None
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


def get_split_indices(
    dates: pd.DatetimeIndex,
    splits_config: dict[str, int]
) -> list[SplitIndices]:
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

    train_bdays = _years_to_bdays(train_years)
    calibrate_bdays = _months_to_bdays(calibrate_months)
    test_bdays = _months_to_bdays(test_months)
    roll_forward_bdays = _months_to_bdays(roll_forward_months)

    n_samples = len(dates)
    splits = []

    # First fold: train starts at 0
    train_start = 0

    while True:
        train_end = train_start + train_bdays
        cal_start = train_end + purge_days
        cal_end = cal_start + calibrate_bdays
        test_start = cal_end + purge_days
        test_end = test_start + test_bdays

        if test_end > n_samples:
            break

        train_idx = np.arange(train_start, train_end)
        cal_idx = np.arange(cal_start, cal_end)
        test_idx = np.arange(test_start, test_end)

        split = SplitIndices(
            train_idx=train_idx,
            calibrate_idx=cal_idx,
            test_idx=test_idx,
            train_start=dates[train_start],
            train_end=dates[train_end - 1],
            calibrate_start=dates[cal_start],
            calibrate_end=dates[cal_end - 1],
            test_start=dates[test_start],
            test_end=dates[test_end - 1]
        )
        splits.append(split)

        # Roll forward
        train_start += roll_forward_bdays

    return splits
