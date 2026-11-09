# Task 3: Walk-Forward Splits with Purge Gap (`splits.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement time-ordered walk-forward splits with purge gap to prevent data leakage between train/calibration/test windows.

**Architecture:** Module `src/calibmiss/splits.py` with generator function yielding train/calibrate/test indices for each fold.

**Tech Stack:** Python 3.11+, pandas, numpy.

**Spec:** PRD Sections 10.1, 20.7 (rules 3, 4)

## Global Constraints (from main plan)

- **Walk-forward only:** No random shuffling in splits.
- **Purge gap:** Minimum 5 trading days between training and test windows.
- **Code quality:** Type hints, docstrings, ruff, pytest, ≥70% coverage on splits module.

## Audit Gap: Date-Aware Window Semantics

`get_split_indices` must calculate train/calibration/test windows from actual calendar offsets and locate observations by date, not approximate them as 252/21 positional samples. Purge removes the configured number of trading observations at each boundary. Keep `walk_forward_splits` as the sample-count API. Validate sorted unique dates and test holidays, actual window boundaries, purge counts, and fold leakage. Acceptance: irregular trading calendars retain calendar-defined window lengths and configured trading-observation purges.

## Review Focus

1. **Walk-forward integrity:** Train always before calibrate, calibrate always before test.
2. **Purge gap enforcement:** ≥5 trading days between train.max() and cal.min(), and between cal.max() and test.min().
3. **No leakage between folds:** Training data in fold N doesn't overlap with test data in fold N-1.

---

### Files
- Create: `src/calibmiss/splits.py`
- Test: `tests/test_splits.py` (already exists per PRD structure)

### Interfaces
- **Consumes:** Feature matrix and target from `features.py`, `config/experiment.yaml` (splits config: train_years=3, calibrate_months=6, test_months=6, roll_forward_months=6, purge_days=5)
- **Produces:** Generator yielding `(train_idx, calibrate_idx, test_idx)` for each fold. Indices are integer positions into the feature matrix. Also `SplitIndices` named tuple with date ranges.

---

### Step 1: Write failing test for split generation

```python
# tests/test_splits.py
import pytest
import numpy as np
from calibmiss.splits import walk_forward_splits

def test_walk_forward_splits_basic():
    n_samples = 1000
    splits = list(walk_forward_splits(
        n_samples=n_samples,
        train_years=3,
        calibrate_months=6,
        test_months=6,
        roll_forward_months=6,
        purge_days=5,
        freq="B"  # business days
    ))
    assert len(splits) > 0
    for train, cal, test in splits:
        assert len(train) > 0 and len(cal) > 0 and len(test) > 0
        # Train before calibrate before test
        assert train.max() < cal.min()
        assert cal.max() < test.min()
        # Purge gap
        assert cal.min() - train.max() >= 5
```

**Run:** `pytest tests/test_splits.py::test_walk_forward_splits_basic -v`
**Expected:** FAIL

---

### Step 2: Implement `walk_forward_splits` in `src/calibmiss/splits.py`

**Signature:**
```python
def walk_forward_splits(
    n_samples: int,
    train_years: int,
    calibrate_months: int,
    test_months: int,
    roll_forward_months: int,
    purge_days: int,
    freq: str = "B"
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
```

**Implementation approach:**
- Convert years/months to business days: ~252 days/year, ~21 days/month
- `train_size = train_years * 252`
- `cal_size = calibrate_months * 21`
- `test_size = test_months * 21`
- `roll_size = roll_forward_months * 21`
- Start with `train_end = train_size`
- Loop: train = [0, train_end - purge_days), cal = [train_end, train_end + cal_size), test = [train_end + cal_size + purge_days, train_end + cal_size + purge_days + test_size)
- Increment `train_end += roll_size` each iteration
- Stop when test window exceeds n_samples
- Return indices as numpy arrays

---

### Step 2b: Implement `get_split_indices` for real dates

```python
from dataclasses import dataclass
from typing import Generator

@dataclass
class SplitIndices:
    train: np.ndarray
    calibrate: np.ndarray
    test: np.ndarray
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    cal_dates: tuple[pd.Timestamp, pd.Timestamp]
    test_dates: tuple[pd.Timestamp, pd.Timestamp]

def get_split_indices(
    dates: pd.DatetimeIndex,
    train_years: int,
    calibrate_months: int,
    test_months: int,
    roll_forward_months: int,
    purge_days: int
) -> list[SplitIndices]:
    """Work with actual trading dates from data."""
    # Similar logic but using date offsets instead of sample counts
    # Use pd.offsets.BDay for business day arithmetic
```

---

### Step 3: Write test for purge gap enforcement

```python
def test_purge_gap_enforced():
    splits = list(walk_forward_splits(..., purge_days=5))
    for train, cal, test in splits:
        assert cal.min() - train.max() >= 5
        assert test.min() - cal.max() >= 5
```

---

### Step 4: Write test for no overlap between folds

```python
def test_no_data_leakage_between_folds():
    splits = list(walk_forward_splits(...))
    all_train = set()
    all_test = set()
    for train, _, test in splits:
        assert not (set(train) & all_test)  # Train doesn't overlap with previous test
        all_train.update(train)
        all_test.update(test)
```

---

### Step 5: Write test for date-aware splits

```python
def test_date_aware_splits():
    # Create date range with known holidays/gaps
    dates = pd.date_range("2010-01-01", "2024-12-31", freq="B")
    splits = get_split_indices(dates, train_years=3, calibrate_months=6, 
                                test_months=6, roll_forward_months=6, purge_days=5)
    for s in splits:
        # Verify purge gap in actual dates
        assert (s.cal_dates[0] - s.train_dates[1]).days >= 5
        assert (s.test_dates[0] - s.cal_dates[1]).days >= 5
```

---

### Step 6: Run all split tests

**Run:** `pytest tests/test_splits.py -v`
**Expected:** PASS

---

### Step 7: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 8: Verify coverage ≥70% on splits module

**Run:** `pytest --cov=src/calibmiss.splits tests/test_splits.py`
**Expected:** Coverage ≥70%

---

### Step 9: Commit

```bash
git add src/calibmiss/splits.py tests/test_splits.py
git commit -m "feat: add walk-forward splits with purge gap"
```

---

## Dependencies

- **Requires:** Task 2 (features.py) for feature matrix structure
- **Blocks:** Task 9 (pipeline.py), Task 10 (run_experiment.py)

---

## Success Criteria

- [ ] `walk_forward_splits` generates correct number of folds
- [ ] Train < Calibrate < Test ordering enforced
- [ ] Purge gap ≥5 trading days between all windows
- [ ] No data leakage between consecutive folds
- [ ] `get_split_indices` works with actual trading dates (handles holidays)
- [ ] All tests pass
- [ ] Coverage ≥70% on splits module
- [ ] Linting and type checking pass