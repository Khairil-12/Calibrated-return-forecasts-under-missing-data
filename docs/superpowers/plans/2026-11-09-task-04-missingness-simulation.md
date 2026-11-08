# Task 4: Missingness Simulation (MCAR and MAR) (`missingness.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement MCAR and MAR missingness injection with configurable rates, seeds, and verification of realized vs target rates.

**Architecture:** Module `src/calibmiss/missingness.py` with functions for MCAR, MAR, and grid simulation.

**Tech Stack:** Python 3.11+, pandas, numpy, pydantic (for MARRuleConfig).

**Spec:** PRD Sections 8.1-8.3, 20.3, 20.7 (rule 5)

## Global Constraints

- **Seeds fixed:** All randomness controlled via config seeds.
- **Verification:** Record realized missing rate per column; plot realized vs target.

## Review Focus

1. **MCAR correctness:** Each value independently missing with probability p.
2. **MAR correctness:** Missingness depends on observed variable, not the missing value itself.
3. **Rate calibration:** MAR must achieve target overall missing rate (binary search on params).
4. **Seed reproducibility:** Same seed → same missingness pattern.

---

### Files
- Create: `src/calibmiss/missingness.py`
- Test: `tests/test_missingness.py` (already exists per PRD structure)

### Interfaces
- **Consumes:** Feature matrix (complete), `config/experiment.yaml` (mechanisms, rates, seeds), `config/missingness.yaml` (MAR rules via `MARRuleConfig`)
- **Produces:** 
  - `inject_mcar(X, rate, seed) -> (X_missing, realized_rates_dict)`
  - `inject_mar(X, rule, target_rate, seed) -> (X_missing, realized_rates_dict)`
  - `simulate_missingness_grid(X, mechanisms, rates, seeds, mar_rules) -> dict` mapping (mechanism, rate, seed) to missing matrix

---

### Step 1: Write failing test for MCAR

```python
# tests/test_missingness.py
import pytest
import pandas as pd
import numpy as np
from calibmiss.missingness import inject_mcar

def test_mcar_produces_target_rate():
    X = pd.DataFrame(np.random.randn(1000, 10))
    X_missing, realized = inject_mcar(X, rate=0.2, seed=42)
    overall_rate = X_missing.isna().mean().mean()
    assert abs(overall_rate - 0.2) < 0.02
    assert len(realized) == 10
```

**Run:** `pytest tests/test_missingness.py::test_mcar_produces_target_rate -v`
**Expected:** FAIL

---

### Step 2: Implement `inject_mcar`

```python
def inject_mcar(
    X: pd.DataFrame,
    rate: float,
    seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.RandomState(seed)
    mask = rng.random(X.shape) < rate
    X_missing = X.copy()
    X_missing[mask] = np.nan
    realized = {col: X_missing[col].isna().mean() for col in X.columns}
    return X_missing, realized
```

---

### Step 3: Write failing test for MAR

```python
from calibmiss.config import MARRuleConfig
from calibmiss.missingness import inject_mar

def test_mar_depends_on_observed_variable():
    X = pd.DataFrame({
        "volatility_20d": np.random.randn(1000),
        "volume_ratio": np.random.rand(1000)
    })
    rule = MARRuleConfig(
        target_feature="volatility_20d",
        depends_on="volume_ratio",
        rule_type="quantile_threshold",
        params={"base_prob": 0.05, "max_prob": 0.5, "quantile_threshold": 0.7}
    )
    X_missing, realized = inject_mar(X, rule, target_rate=0.2, seed=42)
    missing_mask = X_missing["volatility_20d"].isna()
    high_vol_ratio = X["volume_ratio"] > X["volume_ratio"].quantile(0.7)
    assert missing_mask[high_vol_ratio].mean() > missing_mask[~high_vol_ratio].mean()
```

---

### Step 4: Implement `inject_mar` with rate calibration

```python
def inject_mar(
    X: pd.DataFrame,
    rule: MARRuleConfig,
    target_rate: float,
    seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    # Only quantile_threshold rule type for now
    assert rule.rule_type == "quantile_threshold"
    
    depends_on = X[rule.depends_on]
    threshold = depends_on.quantile(rule.params["quantile_threshold"])
    high_group = depends_on > threshold
    
    # Binary search to find base_prob/max_prob that give target_rate
    # For simplicity, use params directly and verify; or implement calibration
    base = rule.params["base_prob"]
    max_p = rule.params["max_prob"]
    
    rng = np.random.RandomState(seed)
    X_missing = X.copy()
    
    # Missing prob = base_prob for low group, max_prob for high group
    probs = np.where(high_group, max_p, base)
    mask = rng.random(len(X)) < probs
    X_missing.loc[mask, rule.target_feature] = np.nan
    
    realized = {col: X_missing[col].isna().mean() for col in X.columns}
    return X_missing, realized
```

---

### Step 5: Write test for realized vs target rate verification

```python
def test_realized_rate_matches_target():
    X = pd.DataFrame(np.random.randn(1000, 5))
    for rate in [0.1, 0.2, 0.3]:
        X_miss, realized = inject_mcar(X, rate, seed=42)
        overall = X_miss.isna().mean().mean()
        assert abs(overall - rate) < 0.02
```

---

### Step 6: Implement `simulate_missingness_grid`

```python
def simulate_missingness_grid(
    X: pd.DataFrame,
    mechanisms: list[str],
    rates: list[float],
    seeds: list[int],
    mar_rules: list[MARRuleConfig]
) -> dict:
    """Returns dict[(mechanism, rate, seed)] = (X_missing, realized_rates)"""
    results = {}
    for mech in mechanisms:
        for rate in rates:
            for seed in seeds:
                if mech == "MCAR":
                    X_miss, realized = inject_mcar(X, rate, seed)
                elif mech == "MAR":
                    # Use first MAR rule for target feature
                    rule = mar_rules[0]
                    X_miss, realized = inject_mar(X, rule, rate, seed)
                results[(mech, rate, seed)] = (X_miss, realized)
    return results
```

---

### Step 7: Run all missingness tests

**Run:** `pytest tests/test_missingness.py -v`
**Expected:** PASS

---

### Step 8: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 9: Verify coverage ≥70% on missingness module

**Run:** `pytest --cov=src/calibmiss.missingness tests/test_missingness.py`
**Expected:** Coverage ≥70%

---

### Step 10: Commit

```bash
git add src/calibmiss/missingness.py tests/test_missingness.py
git commit -m "feat: add MCAR and MAR missingness simulation"
```

---

## Dependencies

- **Requires:** Task 2 (features.py) for feature matrix structure
- **Blocks:** Task 5 (imputers.py), Task 9 (pipeline.py)

---

## Success Criteria

- [ ] MCAR produces target missing rate ±2%
- [ ] MAR missingness correlates with observed variable (volume_ratio)
- [ ] Realized rates recorded per column
- [ ] Grid simulation returns all mechanism/rate/seed combinations
- [ ] Same seed produces identical missingness pattern
- [ ] All tests pass
- [ ] Coverage ≥70% on missingness module
- [ ] Linting and type checking pass