# Task 5: Imputation Strategies (`imputers.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three imputation strategies (forward fill, median, iterative/MICE) with strict fit-on-train-only enforcement.

**Architecture:** Module `src/calibmiss/imputers.py` with base class and three concrete imputers, plus factory function.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn (IterativeImputer).

**Spec:** PRD Sections 9.1, 20.7 (rule 2)

## Global Constraints

- **Fit on train only:** Imputers fit ONLY on the training split, then apply to train, calibration, and test data.
- **No data leakage:** Learned fallback statistics come only from training data. Forward fill may use earlier observed feature values within the transformed split, including test, because that is causal time-series history; it must never use later rows or cross ticker boundaries.
- **Chronological input:** Data passed to `ForwardFillImputer.transform()` is ordered by date within ticker.

## Review Focus

1. **Causal forward fill:** Within each ticker, a missing value can use only an earlier observed value; later rows must not affect earlier imputations.
2. **Ticker isolation:** Forward fill must never carry values across ticker groups.
3. **Training-only fallback:** Leading gaps use per-ticker training median, falling back to overall training median for unseen tickers or all-missing ticker columns.
4. **Median and iterative fit:** Median and sklearn IterativeImputer fit on training data only; test distribution must not change fitted statistics or learned parameters.
5. **Input order:** Forward-fill inputs must be sorted chronologically within ticker before transformation.

---

### Files
- Create: `src/calibmiss/imputers.py`
- Test: `tests/test_imputers.py` (already exists per PRD structure)

### Interfaces
- **Consumes:** Training feature matrix (with NaN), test feature matrix (with NaN), imputer config from experiment.yaml
- **Produces:** Fitted imputer object with `fit()`, `transform()` methods; imputed train/test matrices

---

### Step 1: Write failing tests for causal forward fill and fallback

Add tests to `tests/test_imputers.py`:

- `test_forward_fill_within_ticker` verifies interior NaNs use the previous value from the same ticker and never another ticker's value.
- `test_forward_fill_uses_training_median_for_leading_gaps` fits on a training MultiIndex with different values per ticker, then transforms a later test split whose first row per ticker is missing. Assert each leading gap uses that ticker's training median.
- `test_forward_fill_uses_global_training_median_for_unseen_ticker` transforms a ticker absent from training and verifies fallback uses the overall training median.
- `test_forward_fill_is_causal` changes a later test observation and verifies earlier imputed rows remain identical; assert a later non-missing value stays unchanged.

Use chronological date indexes in each ticker group. The test split must have dates after the training split. Assert all returned values are non-missing when training fallback values are available.

---

### Step 2: Implement `ForwardFillImputer`

`fit(X)` stores the overall training median per column and each ticker's training median per column. If a ticker has no observed training value for a column, its fallback is the overall training median. `transform(X)` forward-fills each column within ticker in input order, then fills remaining leading NaNs from the fitted per-ticker fallback or overall fallback for unseen tickers. For a simple index without a ticker level, use forward fill followed by the overall training median. Do not sort or otherwise reorder caller rows.

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

### Step 7: Write tests for train-only fitting

Add tests to `tests/test_imputers.py`:

- `test_median_imputer_fit_on_train_only` asserts the fitted medians equal training medians and a missing test value uses the training median despite extreme observed test values.
- `test_iterative_imputer_fit_on_train_only` fits two wrappers on identical training data, transforms test frames with different extreme values, and verifies each wrapper's learned `imputer.initial_imputer_.statistics_` is unchanged by `transform()`.
- `test_forward_fill_fallback_fit_on_train_only` fits two imputers on identical training data and verifies changing test values cannot change fitted per-ticker or overall fallback medians; verify a leading missing value is filled from the training fallback.

Do not assert that an interior forward-fill gap uses a training median: it must use the preceding observed value in the same chronological test series.

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

- [ ] Forward fill is causal, remains isolated per ticker, and preserves input row order
- [ ] Leading gaps use per-ticker training median or overall training median for unseen/all-missing tickers
- [ ] Median imputer uses training medians for test
- [ ] Iterative imputer wraps sklearn correctly and fits only on training data
- [ ] Tests prove transformed test history does not alter learned fallback statistics or fitted imputer parameters
- [ ] Factory function returns correct imputer
- [ ] All tests pass
- [ ] Coverage ≥70%
- [ ] Linting and type checking pass