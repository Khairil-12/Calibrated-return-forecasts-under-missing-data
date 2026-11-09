# Task 8: Metrics Computation (`metrics.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Brier score, ECE, ROC AUC, and reliability diagram data computation.

**Architecture:** Module `src/calibmiss/metrics.py` with individual metric functions and a combined `compute_metrics` function.

**Tech Stack:** Python 3.11+, numpy, scikit-learn (roc_auc_score).

**Spec:** PRD Sections 10.3, 14.3

## Audit Gap: Metric Input Contract

Validate equal nonzero input lengths, binary labels, finite probabilities within [0, 1], and positive `n_bins`. When labels contain one class, return `nan` ROC AUC while still computing other metrics. Acceptance: invalid arguments raise `ValueError`; one-class data returns finite Brier/ECE/reliability values and `nan` AUC.

## Global Constraints

- **ECE:** 10 equal-width bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
- **Reliability diagram data:** bin_edges, bin_accuracies, bin_confidences, bin_counts

---

### Files
- Create: `src/calibmiss/metrics.py`
- Test: `tests/test_metrics.py` (new)

### Interfaces
- **Consumes:** True labels (y_true), predicted probabilities (y_prob), n_bins=10
- **Produces:** Dictionary with keys: "brier", "ece", "roc_auc", "reliability" (dict with bin_edges, bin_accuracies, bin_confidences, bin_counts)

---

### Step 1: Write failing test for Brier score

```python
# tests/test_metrics.py
import pytest
import numpy as np
from calibmiss.metrics import compute_metrics

def test_brier_score():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    expected = np.mean((y_prob - y_true) ** 2)
    result = compute_metrics(y_true, y_prob, n_bins=10)
    assert abs(result["brier"] - expected) < 1e-10
```

---

### Step 2: Implement `compute_brier` and `compute_metrics`

```python
def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            weight = mask.sum() / len(y_true)
            ece += weight * abs(acc - conf)
    return ece

def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_prob))

def compute_reliability_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracies.append(float(y_true[mask].mean()))
            bin_confidences.append(float(y_prob[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)
    
    return {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts
    }

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    return {
        "brier": compute_brier(y_true, y_prob),
        "ece": compute_ece(y_true, y_prob, n_bins),
        "roc_auc": compute_roc_auc(y_true, y_prob),
        "reliability": compute_reliability_data(y_true, y_prob, n_bins)
    }
```

---

### Step 3: Write failing test for ECE

```python
def test_ece_10_bins():
    y_true = np.array([0, 1] * 50)
    y_prob = np.array([0.5] * 100)
    result = compute_metrics(y_true, y_prob, n_bins=10)
    assert result["ece"] < 0.01
```

---

### Step 4: Write failing test for ROC AUC

```python
def test_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    result = compute_metrics(y_true, y_prob, n_bins=10)
    assert result["roc_auc"] == 1.0
```

---

### Step 5: Write failing test for reliability diagram data

```python
def test_reliability_diagram_data():
    y_true = np.array([0, 1] * 50)
    y_prob = np.array([0.3] * 50 + [0.7] * 50)
    result = compute_metrics(y_true, y_prob, n_bins=10)
    
    assert "reliability" in result
    rel = result["reliability"]
    assert "bin_edges" in rel
    assert "bin_accuracies" in rel
    assert "bin_confidences" in rel
    assert "bin_counts" in rel
    assert len(rel["bin_edges"]) == 11
```

---

### Step 6: Run all metrics tests

**Run:** `pytest tests/test_metrics.py -v`
**Expected:** PASS

---

### Step 7: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 8: Commit

```bash
git add src/calibmiss/metrics.py tests/test_metrics.py
git commit -m "feat: add Brier, ECE, ROC AUC, and reliability diagram metrics"
```

---

## Dependencies

- **Requires:** Task 2 (features.py), Task 6 (calibration.py)
- **Blocks:** Task 9 (pipeline.py)

---

## Success Criteria

- [ ] Brier score matches manual calculation
- [ ] ECE computes correctly with 10 equal-width bins
- [ ] ROC AUC uses sklearn and returns correct values
- [ ] Reliability diagram data has correct structure
- [ ] All tests pass
- [ ] Linting and type checking pass