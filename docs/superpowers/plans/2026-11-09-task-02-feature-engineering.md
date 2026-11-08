# Task 2: Feature Engineering with Lookahead Safety (`features.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement feature engineering from raw OHLCV data with strict lookahead bias prevention and unit test enforcement.

**Architecture:** Single module `src/calibmiss/features.py` with `compute_features` function that computes all features using only data up to day t, plus a critical lookahead safety test.

**Tech Stack:** Python 3.11+, pandas, numpy.

**Spec:** PRD Sections 7.1-7.2, 20.7 (rule 1)

## Global Constraints (from main plan)

- **No lookahead bias:** Features computed using only data up to day t. Unit test enforces this.
- **Code quality:** Type hints on public functions, docstrings, ruff linting, pytest tests, ≥70% coverage on features module.

## Review Focus

1. **Lookahead bias:** ANY feature computation using future data (t+1 or later) will invalidate ALL results. The lookahead test must catch this.
2. **Target definition:** Correctly drops zero-return days, binary 1/0 for up/down.

---

### Files
- Create: `src/calibmiss/features.py`
- Test: `tests/test_features_lookahead.py` (already exists per PRD structure)

### Interfaces
- **Consumes:** Raw OHLCV DataFrame from `data.py` (MultiIndex: ticker, date; columns: open, high, low, close, adj_close, volume)
- **Produces:** 
  - Feature matrix DataFrame with MultiIndex (ticker, date) and columns:
    - `return_1d`, `return_5d`, `return_20d` (log returns)
    - `volatility_5d`, `volatility_20d` (rolling std * sqrt(252))
    - `volume_ratio` (volume / 20-day avg volume)
    - `hl_range` ((high - low) / close)
    - `price_vs_ma50` (adj_close / 50-day MA)
  - Target Series `y` with same index (binary: 1 if next-day adj_close > current adj_close, 0 otherwise; zero-return days dropped)

---

### Step 1: Write failing test for feature computation

```python
# tests/test_features_lookahead.py
import pytest
import pandas as pd
import numpy as np
from calibmiss.features import compute_features

def test_compute_features_returns_expected_columns():
    raw = load_sample_raw_data()  # helper to create sample raw OHLCV
    features, target = compute_features(raw)
    expected_cols = [
        "return_1d", "return_5d", "return_20d",
        "volatility_5d", "volatility_20d",
        "volume_ratio", "hl_range", "price_vs_ma50"
    ]
    assert list(features.columns) == expected_cols
    assert target.name == "target"
    assert set(target.unique()) == {0, 1}
```

**Run:** `pytest tests/test_features_lookahead.py::test_compute_features_returns_expected_columns -v`
**Expected:** FAIL

---

### Step 2: Implement `compute_features` in `src/calibmiss/features.py`

**Signature:**
```python
def compute_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
```

**Implementation (per ticker group, using only data up to day t):**

```python
def compute_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Work per ticker to avoid cross-ticker leakage
    def _compute_one_ticker(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()  # ensure chronological order
        
        # Log returns: ln(adj_close_t / adj_close_{t-n})
        df["return_1d"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
        df["return_5d"] = np.log(df["adj_close"] / df["adj_close"].shift(5))
        df["return_20d"] = np.log(df["adj_close"] / df["adj_close"].shift(20))
        
        # Rolling volatility (annualized): std of 1d returns * sqrt(252)
        df["volatility_5d"] = df["return_1d"].rolling(5).std() * np.sqrt(252)
        df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)
        
        # Volume ratio: today's volume / 20-day average volume
        df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        
        # High-low range / close
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        
        # Price vs 50-day MA
        df["price_vs_ma50"] = df["adj_close"] / df["adj_close"].rolling(50).mean()
        
        return df
    
    # Apply per ticker
    featured = raw.groupby(level="ticker", group_keys=False).apply(_compute_one_ticker)
    
    # Target: 1 if next-day adj_close > current adj_close, else 0
    # Drop zero-return days
    next_close = featured.groupby(level="ticker")["adj_close"].shift(-1)
    target = (next_close > featured["adj_close"]).astype(int)
    # Drop where next_close == current (zero return)
    zero_return_mask = (next_close == featured["adj_close"])
    target = target[~zero_return_mask]
    featured = featured[~zero_return_mask]
    
    # Select feature columns only
    feature_cols = [
        "return_1d", "return_5d", "return_20d",
        "volatility_5d", "volatility_20d",
        "volume_ratio", "hl_range", "price_vs_ma50"
    ]
    features = featured[feature_cols]
    
    target.name = "target"
    return features, target
```

---

### Step 3: Write lookahead safety test (CRITICAL)

```python
def test_no_lookahead_bias():
    """Shift future data and verify features don't change."""
    raw = load_sample_raw_data()
    features, _ = compute_features(raw)
    
    # Create corrupted raw data where future values are different
    raw_corrupted = raw.copy()
    # Shift adj_close forward by 1 day (simulating lookahead access)
    raw_corrupted["adj_close"] = raw_corrupted.groupby("ticker")["adj_close"].shift(-1)
    
    features_corrupted, _ = compute_features(raw_corrupted)
    
    # Features should be IDENTICAL - no lookahead
    pd.testing.assert_frame_equal(features, features_corrupted)
```

**Run:** `pytest tests/test_features_lookahead.py::test_no_lookahead_bias -v`
**Expected:** PASS (this is the critical safety test)

---

### Step 4: Write test for target definition

```python
def test_target_drops_zero_return_days():
    raw = create_raw_with_zero_returns()  # helper with known zero-return day
    _, target = compute_features(raw)
    # Zero-return days should not be in target index
    zero_return_date = pd.Timestamp("2023-01-15")  # known zero-return date in helper
    assert all(target.index.get_level_values("date") != zero_return_date)
```

---

### Step 5: Run all feature tests

**Run:** `pytest tests/test_features_lookahead.py -v`
**Expected:** PASS

---

### Step 6: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 7: Verify coverage ≥70% on features module

**Run:** `pytest --cov=src/calibmiss.features tests/test_features_lookahead.py`
**Expected:** Coverage ≥70%

---

### Step 8: Commit

```bash
git add src/calibmiss/features.py tests/test_features_lookahead.py
git commit -m "feat: add feature engineering with lookahead safety test"
```

---

## Dependencies

- **Requires:** Task 1 (data.py) for raw data structure
- **Blocks:** Task 3 (splits.py), Task 4 (missingness.py), Task 12 (compute_features.py), Task 14 (exploration.ipynb)

---

## Success Criteria

- [ ] `compute_features` returns DataFrame with exactly 8 feature columns
- [ ] Target is binary (0/1) with zero-return days dropped
- [ ] Lookahead safety test passes (corrupted future data doesn't change features)
- [ ] All feature computations use only data up to day t (verified by test)
- [ ] All tests pass
- [ ] Coverage ≥70% on features module
- [ ] Linting and type checking pass