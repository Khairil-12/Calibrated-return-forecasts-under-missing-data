# Task 9: Single Configuration Pipeline (`pipeline.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement orchestration for a single experimental configuration (model, imputer, calibrator, mechanism, rate, seed) on one fold.

**Architecture:** Module `src/calibmiss/pipeline.py` with `run_single_config` function that coordinates all components.

**Tech Stack:** Python 3.11+, pandas, numpy, joblib (for parallel later).

**Spec:** PRD Sections 10.2, 20.7 (rules 1-5)

## Global Constraints

- **No lookahead:** Features already validated in Task 2.
- **Fit on train only:** Imputers, scalers, calibrators fit ONLY on training split.
- **Walk-forward only:** Splits from Task 3.
- **Seeds fixed:** All randomness controlled via config seeds.

## Review Focus

1. **Correct missingness injection:** Train and test get missingness; calibration set handling per PRD.
2. **Imputer fit on train only:** Critical - no test data leakage.
3. **Calibrator fit on calibration split only:** Critical - no test label leakage.
4. **Baseline (0% missingness) works:** No imputation needed, matches oracle.

---

### Files
- Create: `src/calibmiss/pipeline.py`
- Test: `tests/test_pipeline.py` (new)

### Interfaces
- **Consumes:** Complete feature matrix (X), target (y), single config dict, split indices (train_idx, cal_idx, test_idx)
- **Produces:** Dictionary with all metrics for that configuration on the test set

---

### Step 1: Write failing test for single pipeline run

```python
# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from calibmiss.pipeline import run_single_config

def test_run_single_config():
    X, y = create_sample_data()  # helper
    train_idx, cal_idx, test_idx = create_sample_splits()
    
    config = {
        "model": "logistic_regression",
        "imputer": "median",
        "calibrator": "platt",
        "mechanism": "MCAR",
        "rate": 0.1,
        "seed": 42
    }
    
    result = run_single_config(X, y, train_idx, cal_idx, test_idx, config)
    
    assert "brier" in result
    assert "ece" in result
    assert "roc_auc" in result
    assert "reliability" in result
```

---

### Step 2: Implement `run_single_config`

```python
def run_single_config(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    config: dict
) -> dict:
    # 1. Split data
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_cal, y_cal = X.iloc[cal_idx], y.iloc[cal_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    # 2. Inject missingness on train and test (per PRD: "on test set and training set separately")
    # Calibration set: also inject missingness (same mechanism/rate) then impute
    from calibmiss.missingness import inject_mcar, inject_mar
    from calibmiss.config import get_config, get_mar_rules
    
    cfg = get_config()
    mar_rules = get_mar_rules()
    
    rate = config["rate"]
    seed = config["seed"]
    mechanism = config["mechanism"]
    
    if rate > 0:
        if mechanism == "MCAR":
            X_train_miss, _ = inject_mcar(X_train, rate, seed)
            X_cal_miss, _ = inject_mcar(X_cal, rate, seed + 1)  # different seed for cal
            X_test_miss, _ = inject_mcar(X_test, rate, seed + 2)
        elif mechanism == "MAR":
            rule = mar_rules.mar_rules[0]
            X_train_miss, _ = inject_mar(X_train, rule, rate, seed)
            X_cal_miss, _ = inject_mar(X_cal, rule, rate, seed + 1)
            X_test_miss, _ = inject_mar(X_test, rule, rate, seed + 2)
    else:
        X_train_miss, X_cal_miss, X_test_miss = X_train, X_cal, X_test
    
    # 3. Fit imputer on X_train_miss, transform all three
    from calibmiss.imputers import get_imputer
    from calibmiss.config import ImputerConfig
    
    imputer_cfg = ImputerConfig(name=config["imputer"], params={})
    imputer = get_imputer(imputer_cfg)
    imputer.fit(X_train_miss)
    X_train_imp = imputer.transform(X_train_miss)
    X_cal_imp = imputer.transform(X_cal_miss)
    X_test_imp = imputer.transform(X_test_miss)
    
    # 4. Train model on imputed train
    from calibmiss.models import get_model
    from calibmiss.config import ModelConfig
    
    model_cfg = ModelConfig(name=config["model"], params={})
    model = get_model(model_cfg)
    model.fit(X_train_imp, y_train)
    
    # 5. Get uncalibrated probs on cal and test
    p_cal = model.predict_proba(X_cal_imp)
    p_test = model.predict_proba(X_test_imp)
    
    # 6. Fit calibrator on calibration probs and labels
    from calibmiss.calibration import get_calibrator
    from calibmiss.config import CalibratorConfig
    
    cal_cfg = CalibratorConfig(name=config["calibrator"], params={})
    calibrator = get_calibrator(cal_cfg)
    calibrator.fit(p_cal, y_cal.values)
    
    # 7. Get calibrated probs on test
    p_test_cal = calibrator.predict_proba(p_test)
    
    # 8. Compute metrics
    from calibmiss.metrics import compute_metrics
    from calibmiss.config import get_config
    
    n_bins = get_config().experiment.evaluation.n_bins_ece
    metrics = compute_metrics(y_test.values, p_test_cal, n_bins=n_bins)
    
    # 9. Return metrics + config info
    result = {**metrics, **config, "fold": 0}  # fold added by caller
    return result
```

---

### Step 3: Write test for baseline (0% missingness)

```python
def test_baseline_zero_missingness():
    X, y = create_sample_data()
    train_idx, cal_idx, test_idx = create_sample_splits()
    
    config = {
        "model": "logistic_regression",
        "imputer": "median",
        "calibrator": "platt",
        "mechanism": "MCAR",
        "rate": 0.0,
        "seed": 42
    }
    
    result = run_single_config(X, y, train_idx, cal_idx, test_idx, config)
    assert result["brier"] >= 0
```

---

### Step 4: Run all pipeline tests

**Run:** `pytest tests/test_pipeline.py -v`
**Expected:** PASS

---

### Step 5: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 6: Commit

```bash
git add src/calibmiss/pipeline.py tests/test_pipeline.py
git commit -m "feat: add single configuration pipeline orchestration"
```

---

## Dependencies

- **Requires:** Tasks 3, 4, 5, 6, 7, 8
- **Blocks:** Task 10 (run_experiment.py)

---

## Success Criteria

- [ ] Single config runs end-to-end
- [ ] Missingness injected on train, cal, test
- [ ] Imputer fit on train only
- [ ] Model trained on imputed train
- [ ] Calibrator fit on calibration split only
- [ ] Metrics computed on test set
- [ ] Baseline (0% missingness) works
- [ ] All tests pass
- [ ] Linting and type checking pass