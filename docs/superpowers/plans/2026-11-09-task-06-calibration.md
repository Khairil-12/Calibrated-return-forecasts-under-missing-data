# Task 6: Calibration Methods (`calibration.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Platt scaling, isotonic regression, and no-calibration with strict fit-on-calibration-split-only enforcement.

**Architecture:** Module `src/calibmiss/calibration.py` with base class and three calibrators, plus factory function.

**Tech Stack:** Python 3.11+, numpy, scikit-learn (CalibratedClassifierCV).

**Spec:** PRD Sections 9.3, 20.7 (rule 2)

## Global Constraints

- **Fit on calibration split only:** Calibrators fit ONLY on calibration split (not training, not test).
- **No data leakage:** Test probabilities calibrated using only calibration data.

## Review Focus

1. **Platt scaling:** Logistic regression on calibration probs → test probs.
2. **Isotonic regression:** Non-parametric monotone mapping.
3. **No calibration:** Identity transform.
4. **Fit-on-cal-only test:** Critical test verifying no test/train label leakage.

---

### Files
- Create: `src/calibmiss/calibration.py`
- Test: `tests/test_calibration.py` (already exists per PRD structure)

### Interfaces
- **Consumes:** Uncalibrated probabilities on calibration set, true labels on calibration set, test probabilities
- **Produces:** Calibrated probabilities on test set

---

### Step 1: Write failing test for Platt scaling

```python
# tests/test_calibration.py
import pytest
import numpy as np
from calibmiss.calibration import PlattCalibrator

def test_platt_scaling():
    p_cal = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
    y_cal = np.array([1, 1, 0, 0, 1, 0])
    p_test = np.array([0.85, 0.15, 0.65])
    
    calibrator = PlattCalibrator()
    calibrator.fit(p_cal, y_cal)
    p_test_cal = calibrator.predict_proba(p_test)
    
    assert p_test_cal.shape == (3,)
    assert p_test_cal[0] < 0.85  # Less overconfident
    assert p_test_cal[1] > 0.15  # Less overconfident
```

---

### Step 2: Implement `PlattCalibrator`

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

class PlattCalibrator:
    def __init__(self):
        self.calibrator = CalibratedClassifierCV(
            estimator=LogisticRegression(),
            method="sigmoid",
            cv="prefit"
        )
    
    def fit(self, probas: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        # CalibratedClassifierCV with cv="prefit" expects already-fitted estimator
        # But we just want Platt scaling (logistic regression on probas)
        # Actually, simpler: just fit LogisticRegression on probas directly
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        self.model.fit(probas.reshape(-1, 1), labels)
        return self
    
    def predict_proba(self, probas: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(probas.reshape(-1, 1))[:, 1]
```

---

### Step 3: Write failing test for isotonic regression

```python
from calibmiss.calibration import IsotonicCalibrator

def test_isotonic_regression():
    p_cal = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.4])
    y_cal = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p_test = np.array([0.85, 0.15, 0.65, 0.35])
    
    calibrator = IsotonicCalibrator()
    calibrator.fit(p_cal, y_cal)
    p_test_cal = calibrator.predict_proba(p_test)
    
    # Isotonic should be non-decreasing
    assert np.all(np.diff(p_test_cal) >= -1e-10)
```

---

### Step 4: Implement `IsotonicCalibrator`

```python
from sklearn.isotonic import IsotonicRegression

class IsotonicCalibrator:
    def fit(self, probas: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self.isotonic = IsotonicRegression(out_of_bounds="clip")
        self.isotonic.fit(probas, labels)
        return self
    
    def predict_proba(self, probas: np.ndarray) -> np.ndarray:
        return self.isotonic.predict(probas)
```

---

### Step 5: Write test for no-calibration

```python
from calibmiss.calibration import NoCalibrator

def test_no_calibration_returns_same_probs():
    calibrator = NoCalibrator()
    calibrator.fit(np.array([0.5, 0.5]), np.array([0, 1]))
    p_test = np.array([0.1, 0.9])
    assert np.array_equal(calibrator.predict_proba(p_test), p_test)
```

---

### Step 6: Implement `NoCalibrator`

```python
class NoCalibrator:
    def fit(self, probas: np.ndarray, labels: np.ndarray) -> "NoCalibrator":
        return self
    
    def predict_proba(self, probas: np.ndarray) -> np.ndarray:
        return probas
```

---

### Step 7: Write critical test for fit-on-calibration-split-only

```python
def test_calibrators_fit_only_on_calibration_data():
    p_train = np.array([0.9, 0.1])  # Not used for fitting
    y_train = np.array([1, 0])
    p_cal = np.array([0.8, 0.2])
    y_cal = np.array([1, 0])
    p_test = np.array([0.7])
    
    for CalibratorClass in [PlattCalibrator, IsotonicCalibrator]:
        cal = CalibratorClass()
        cal.fit(p_cal, y_cal)  # Only calibration data
        p_test_cal = cal.predict_proba(p_test)
        # Should not depend on train data at all
```

---

### Step 8: Implement factory function

```python
from calibmiss.config import CalibratorConfig

def get_calibrator(config: CalibratorConfig):
    if config.name == "none":
        return NoCalibrator()
    elif config.name == "platt":
        return PlattCalibrator()
    elif config.name == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibrator: {config.name}")
```

---

### Step 9: Run all calibration tests

**Run:** `pytest tests/test_calibration.py -v`
**Expected:** PASS

---

### Step 10: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 11: Commit

```bash
git add src/calibmiss/calibration.py tests/test_calibration.py
git commit -m "feat: add Platt, isotonic, and no calibration with fit-on-cal-only enforcement"
```

---

## Dependencies

- **Requires:** Task 2 (features.py) for probability structure
- **Blocks:** Task 9 (pipeline.py)

---

## Success Criteria

- [ ] Platt scaling reduces overconfidence
- [ ] Isotonic regression produces monotone calibrated probabilities
- [ ] No calibration returns identical probabilities
- [ ] Fit-on-calibration-split-only test passes for all calibrators
- [ ] Factory function returns correct calibrator
- [ ] All tests pass
- [ ] Linting and type checking pass