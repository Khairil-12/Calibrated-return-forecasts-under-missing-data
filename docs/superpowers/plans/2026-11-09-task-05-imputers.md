# Task 5: Imputation Strategies (`imputers.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three imputation strategies (forward fill, median, iterative/MICE) with strict fit-on-train-only enforcement.

**Architecture:** Module `src/calibmiss/imputers.py` with base class and three concrete imputers, plus factory function.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn (IterativeImputer).

**Spec:** PRD Sections 9.1, 20.7 (rule 2)

## Global Constraints

- **Fit on train only:** Imputers fit ONLY on training split, then applied to test.
- **No data leakage:** Test imputation uses only training statistics.

## Review Focus

1. **Forward fill per ticker:** Must forward fill within each ticker group, not across tickers.
2. **Median from train only:** Test NaNs filled with training medians.
3. **Iterative imputer fit on train:** sklearn's IterativeImputer fit on training data only.
4. **Fit-on-train-only test:** Critical test verifying no test info leakage.

---

### Files
- Create: `src/calibmiss/imputers.py`
- Test: `tests/test_imputers.py` (already exists per PRD structure)

### Interfaces
- **Consumes:** Training feature matrix (with NaN), test feature matrix (with NaN), imputer config from experiment.yaml
- **Produces:** Fitted imputer object with `fit()`, `transform()` methods; imputed train/test matrices

---

### Step 1: Write failing test for forward fill imputer

```python
# tests/test_imputers.py
import pytest
import pandas as pd
import numpy as np
from calibmiss.imputers import ForwardFillImputer

def test_forward_fill_within_ticker():
    # MultiIndex (ticker, date) with NaN
    idx = pd.MultiIndex.from_tuples([
        ("AAPL", "2023-01-01"), ("AAPL", "2023-01-02"), ("AAPL", "2023-01-03"),
        ("MSFT", "2023-01-01"), ("MSFT", "2023-01-02"), ("MSFT", "2023-01-03")
    ], names=["ticker", "date"])
    X_train = pd.DataFrame({"feat": [1.0, np.nan, 3.0, 10.0, np.nan, 30.0]}, index=idx)
    X_test = pd.DataFrame({"feat": [np.nan, 5.0, np.nan, np.nan, 20.0, np.nan]}, index=idx)
    
    imputer = ForwardFillImputer()
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    assert not X_train_imp.isna().any().any()
    assert not X_test_imp.isna().any().any()
    # AAPL: [1, NaN, 3] -> [1, 1, 3] (forward fill)
    # MSFT: [10, NaN, 30] -> [10, 10, 30]
    assert X_train_imp.loc[("AAPL", "2023-01-02"), "feat"] == 1.0
    assert X_train_imp.loc[("MSFT", "2023-01-02"), "feat"] == 10.0
```

---

### Step 2: Implement `ForwardFillImputer`

```python
class ForwardFillImputer:
    def fit(self, X: pd.DataFrame) -> "ForwardFillImputer":
        # Compute column medians for leading NaNs (per ticker)
        self.medians_ = {}
        for col in X.columns:
            # Get first non-NaN per ticker group, or overall median
            first_valid = X.groupby(level="ticker")[col].apply(
                lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan
            )
            overall_median = X[col].median()
            self.medians_[col] = first_valid.fillna(overall_median).to_dict()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imp = X.copy()
        for col in X.columns:
            # Forward fill within each ticker
            X_imp[col] = X_imp.groupby(level="ticker")[col].ffill()
            # Fill remaining leading NaNs with stored medians
            for ticker, med in self.medians_[col].items():
                mask = (X_imp.index.get_level_values("ticker") == ticker) & X_imp[col].isna()
                X_imp.loc[mask, col] = med
        return X_imp
```

---

### Step 3: Write failing test for median imputer

```python
from calibmiss.imputers import MedianImputer

def test_median_imputer_fit_on_train_only():
    X_train = pd.DataFrame({"a": [1, 2, np.nan], "b": [np.nan, 5, 6]})
    X_test = pd.DataFrame({"a": [np.nan, 8], "b": [9, np.nan]})
    
    imputer = MedianImputer()
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    assert imputer.medians_["a"] == 1.5
    assert imputer.medians_["b"] == 5.5
    assert X_test_imp["a"].iloc[0] == 1.5
    assert X_test_imp["b"].iloc[1] == 5.5
```

---

### Step 4: Implement `MedianImputer`

```python
class MedianImputer:
    def fit(self, X: pd.DataFrame) -> "MedianImputer":
        self.medians_ = X.median().to_dict()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imp = X.copy()
        for col, med in self.medians_.items():
            X_imp[col] = X_imp[col].fillna(med)
        return X_imp
```

---

### Step 5: Write failing test for iterative imputer

```python
from calibmiss.imputers import IterativeImputerWrapper

def test_iterative_imputer():
    # Correlated features
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "a": np.random.randn(n),
        "b": np.random.randn(n) * 0.5 + 0.5 * np.random.randn(n),  # correlated with a
    })
    # Introduce NaN
    mask = np.random.random((n, 2)) < 0.1
    X_miss = X.copy()
    X_miss.values[mask] = np.nan
    
    X_train = X_miss.iloc[:100]
    X_test = X_miss.iloc[100:]
    
    imputer = IterativeImputerWrapper(max_iter=10, random_state=42)
    imputer.fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    assert not X_train_imp.isna().any().any()
    assert not X_test_imp.isna().any().any()
```

---

### Step 6: Implement `IterativeImputerWrapper`

```python
from sklearn.impute import IterativeImputer

class IterativeImputerWrapper:
    def __init__(self, max_iter=10, random_state=42):
        self.imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    
    def fit(self, X: pd.DataFrame) -> "IterativeImputerWrapper":
        self.columns_ = X.columns.tolist()
        self.index_ = X.index
        self.imputer.fit(X.values)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imp = pd.DataFrame(
            self.imputer.transform(X.values),
            columns=self.columns_,
            index=X.index
        )
        return X_imp
```

---

### Step 7: Write critical test for fit-on-train-only enforcement

```python
def test_imputers_never_fit_on_test():
    X_train = pd.DataFrame({"a": [1, 2, np.nan]})
    X_test = pd.DataFrame({"a": [100, np.nan]})  # Very different distribution
    
    for ImputerClass in [ForwardFillImputer, MedianImputer, IterativeImputerWrapper]:
        imputer = ImputerClass()
        imputer.fit(X_train)
        X_test_imp = imputer.transform(X_test)
        if hasattr(imputer, "medians_"):
            assert X_test_imp["a"].iloc[1] == imputer.medians_["a"]
```

---

### Step 8: Implement factory function

```python
from calibmiss.config import ImputerConfig

def get_imputer(config: ImputerConfig):
    if config.name == "forward_fill":
        return ForwardFillImputer()
    elif config.name == "median":
        return MedianImputer()
    elif config.name == "iterative":
        return IterativeImputerWrapper(**config.params)
    raise ValueError(f"Unknown imputer: {config.name}")
```

---

### Step 9: Run all imputer tests

**Run:** `pytest tests/test_imputers.py -v`
**Expected:** PASS

---

### Step 10: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 11: Verify coverage

**Run:** `pytest --cov=src/calibmiss.imputers tests/test_imputers.py`
**Expected:** Coverage ≥70%

---

### Step 12: Commit

```bash
git add src/calibmiss/imputers.py tests/test_imputers.py
git commit -m "feat: add three imputation strategies with fit-on-train-only enforcement"
```

---

## Dependencies

- **Requires:** Task 2 (features.py), Task 4 (missingness.py)
- **Blocks:** Task 7 (models.py), Task 9 (pipeline.py)

---

## Success Criteria

- [ ] Forward fill works per ticker group
- [ ] Median imputer uses training medians for test
- [ ] Iterative imputer wraps sklearn correctly
- [ ] Fit-on-train-only test passes for all three imputers
- [ ] Factory function returns correct imputer
- [ ] All tests pass
- [ ] Coverage ≥70%
- [ ] Linting and type checking pass