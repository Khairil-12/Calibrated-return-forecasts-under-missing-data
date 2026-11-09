"""Tests for walk-forward splits with purge gap."""

import pandas as pd  # type: ignore[import-untyped]
from calibmiss.splits import SplitIndices, get_split_indices, walk_forward_splits


def test_walk_forward_splits_basic() -> None:
    """Test basic walk-forward split generation."""
    n_samples = 2000
    splits = list(walk_forward_splits(
        n_samples=n_samples,
        train_years=2,
        calibrate_months=3,
        test_months=3,
        roll_forward_months=3,
        purge_days=5,
        freq="B"
    ))
    assert len(splits) > 0
    for train, cal, test in splits:
        assert len(train) > 0 and len(cal) > 0 and len(test) > 0
        # Train before calibrate before test
        assert train.max() < cal.min()
        assert cal.max() < test.min()
        # Purge gap
        assert cal.min() - train.max() >= 5


def test_purge_gap_enforced() -> None:
    """Test that purge gap is enforced between train/calibrate and calibrate/test."""
    n_samples = 1000
    splits = list(walk_forward_splits(
        n_samples=n_samples,
        train_years=2,
        calibrate_months=3,
        test_months=3,
        roll_forward_months=3,
        purge_days=5,
        freq="B"
    ))
    for train, cal, test in splits:
        assert cal.min() - train.max() >= 5
        assert test.min() - cal.max() >= 5


def test_no_data_leakage_between_folds() -> None:
    """Test that no future data leaks into train (no lookahead)."""
    n_samples = 2000
    splits = list(walk_forward_splits(
        n_samples=n_samples,
        train_years=2,
        calibrate_months=3,
        test_months=3,
        roll_forward_months=3,
        purge_days=5,
        freq="B"
    ))
    # Check that train never includes data from its own or future test sets
    # (this would be lookahead bias)
    for i, (train, _, test) in enumerate(splits):
        # Current and future test indices
        future_test: set[int] = set()
        for j in range(i, len(splits)):
            future_test.update(splits[j][2])
        # Train should not include current or future test data
        assert not (set(train) & future_test), f"Fold {i}: train includes future test data"


def test_get_split_indices_with_dates() -> None:
    """Test get_split_indices with actual dates."""
    # Create business day dates
    dates = pd.date_range("2020-01-01", periods=2000, freq="B")

    splits_config = {
        "train_years": 2,
        "calibrate_months": 3,
        "test_months": 3,
        "roll_forward_months": 3,
        "purge_days": 5
    }

    split_indices = get_split_indices(dates, splits_config)

    assert len(split_indices) > 0
    for split in split_indices:
        assert isinstance(split, SplitIndices)
        assert len(split.train_idx) > 0
        assert len(split.calibrate_idx) > 0
        assert len(split.test_idx) > 0
        # Verify dates are in correct order
        assert dates[split.train_idx].max() < dates[split.calibrate_idx].min()
        assert dates[split.calibrate_idx].max() < dates[split.test_idx].min()
        # Purge gap
        assert (dates[split.calibrate_idx].min() - dates[split.train_idx].max()).days >= 5
        assert (dates[split.test_idx].min() - dates[split.calibrate_idx].max()).days >= 5
