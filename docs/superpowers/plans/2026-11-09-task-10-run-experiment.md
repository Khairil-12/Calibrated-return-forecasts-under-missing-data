# Task 10: Experiment Grid Runner (`run_experiment.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement full experimental grid runner that loops over all configurations and saves results to CSV.

**Architecture:** Module `src/calibmiss/run_experiment.py` with `run_grid` function and CLI; entry point script `scripts/run_grid.py`.

**Tech Stack:** Python 3.11+, pandas, numpy, joblib (parallelism), argparse.

**Spec:** PRD Sections 10.2, 11 (FR-4, FR-6), 20.4, 20.6

## Global Constraints

- **Reproducibility:** Fixed seeds, library versions recorded.
- **Runtime:** Full grid < 2 hours on laptop. Use caching and parallelism.
- **Output:** `results/metrics.csv` with one row per (fold, seed, model, imputer, calibrator, mechanism, rate)

---

### Files
- Create: `src/calibmiss/run_experiment.py`
- Create: `scripts/run_grid.py`
- Test: `tests/test_run_experiment.py` (new)

### Interfaces
- **Consumes:** Full config from experiment.yaml, feature matrix, target, splits
- **Produces:** `results/metrics.csv` with all metrics per configuration

---

### Step 1: Write failing test for grid structure

```python
# tests/test_run_experiment.py
import pytest
from calibmiss.config import get_config

def test_grid_size():
    cfg = get_config()
    n_models = len(cfg.experiment.models)
    n_imputers = len(cfg.experiment.imputers)
    n_calibrators = len(cfg.experiment.calibrators)
    n_mechanisms = len(cfg.experiment.missingness.mechanisms)
    n_rates = len(cfg.experiment.missingness.rates)
    n_seeds = len(cfg.experiment.missingness.seeds)
    # n_folds depends on data length
    # Expected: 2 * 3 * 3 * 2 * 4 * 5 * n_folds = 720 * n_folds
```

---

### Step 2: Implement `run_grid`

```python
def run_grid(X: pd.DataFrame, y: pd.Series, splits: list, config) -> pd.DataFrame:
    """Run full experimental grid."""
    from joblib import Parallel, delayed
    from calibmiss.pipeline import run_single_config
    from calibmiss.config import get_config
    
    cfg = config or get_config()
    
    # Generate all combinations
    import itertools
    combinations = list(itertools.product(
        range(len(splits)),  # fold
        cfg.experiment.missingness.seeds,  # seed
        cfg.experiment.models,  # model
        cfg.experiment.imputers,  # imputer
        cfg.experiment.calibrators,  # calibrator
        cfg.experiment.missingness.mechanisms,  # mechanism
        cfg.experiment.missingness.rates,  # rate
    ))
    
    def run_one(combo):
        fold_idx, seed, model_cfg, imputer_cfg, cal_cfg, mechanism, rate = combo
        train_idx, cal_idx, test_idx = splits[fold_idx]
        
        config_dict = {
            "model": model_cfg.name,
            "imputer": imputer_cfg.name,
            "calibrator": cal_cfg.name,
            "mechanism": mechanism,
            "rate": rate,
            "seed": seed
        }
        
        result = run_single_config(X, y, train_idx, cal_idx, test_idx, config_dict)
        result["fold"] = fold_idx
        return result
    
    # Parallel execution
    n_jobs = -1  # Use all cores
    results = Parallel(n_jobs=n_jobs)(delayed(run_one)(c) for c in combinations)
    
    df = pd.DataFrame(results)
    return df
```

---

### Step 3: Implement CLI in `run_experiment.py`

```python
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--imputer", type=str)
    parser.add_argument("--calibrator", type=str)
    parser.add_argument("--mechanism", type=str)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    
    # Load data
    from calibmiss.data import download_ohlcv
    from calibmiss.features import compute_features
    from calibmiss.splits import get_split_indices
    from calibmiss.config import get_config
    
    cfg = get_config()
    raw = download_ohlcv(cfg.universe.tickers, cfg.universe.start_date, cfg.universe.end_date)
    X, y = compute_features(raw)
    
    # Get splits
    dates = X.index.get_level_values("date").unique()
    splits = get_split_indices(dates, **cfg.experiment.splits.model_dump())
    
    if args.model:  # Single config mode
        config = {
            "model": args.model,
            "imputer": args.imputer or "median",
            "calibrator": args.calibrator or "none",
            "mechanism": args.mechanism or "MCAR",
            "rate": args.rate or 0.1,
            "seed": args.seed or 42
        }
        # Use first fold for debugging
        train_idx, cal_idx, test_idx = splits[0]
        from calibmiss.pipeline import run_single_config
        result = run_single_config(X, y, train_idx, cal_idx, test_idx, config)
        print(result)
    else:  # Full grid mode
        df = run_grid(X, y, splits, cfg)
        df.to_csv("results/metrics.csv", index=False)
        print(f"Saved {len(df)} rows to results/metrics.csv")
```

---

### Step 4: Create `scripts/run_grid.py`

```python
#!/usr/bin/env python
"""Entry point for running experimental grid."""
from calibmiss.run_experiment import main

if __name__ == "__main__":
    main()
```

---

### Step 5: Write test for CSV output format

```python
def test_metrics_csv_columns():
    df = run_grid_on_sample()  # Helper using small sample
    required_cols = [
        "fold", "seed", "model", "imputer", "calibrator", 
        "mechanism", "rate", "brier", "ece", "roc_auc"
    ]
    for col in required_cols:
        assert col in df.columns
```

---

### Step 6: Run integration test (small sample)

**Run:** `pytest tests/test_run_experiment.py -v -k "not slow"`
**Expected:** PASS

---

### Step 7: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 8: Commit

```bash
git add src/calibmiss/run_experiment.py scripts/run_grid.py tests/test_run_experiment.py
git commit -m "feat: add experiment grid runner with parallel execution"
```

---

## Dependencies

- **Requires:** Task 9 (pipeline.py), Task 11 (download_data.py), Task 12 (compute_features.py)
- **Blocks:** Task 13 (generate_figures.py)

---

## Success Criteria

- [ ] Grid runs all 2×3×3×2×4×5×n_folds combinations
- [ ] Parallel execution with joblib
- [ ] Single config CLI mode works for debugging
- [ ] Output CSV has all required columns
- [ ] Results saved to `results/metrics.csv`
- [ ] All tests pass
- [ ] Linting and type checking pass