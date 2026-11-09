# Task 12: Feature Computation Script (`scripts/compute_features.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create executable script to compute features from raw data and save processed features.

**Architecture:** Script `scripts/compute_features.py` using `calibmiss.data` and `calibmiss.features`.

**Tech Stack:** Python 3.11+, pandas, calibmiss.data, calibmiss.features.

**Spec:** PRD Sections 7.1-7.2, 20.5 (make features), 20.6

---

### Files
- Create: `scripts/compute_features.py`

### Interfaces
- **Consumes:** Raw data from `data/raw/`
- **Produces:** Feature matrix and target in `data/processed/` (parquet)

---

### Step 1: Implement script

```python
#!/usr/bin/env python
"""Compute features from raw OHLCV data."""
import sys
sys.path.insert(0, "src")

import pandas as pd
from calibmiss.data import download_ohlcv
from calibmiss.features import compute_features
from calibmiss.config import get_config

def main():
    cfg = get_config()
    
    print("Loading raw data...")
    raw = download_ohlcv(cfg.universe.tickers, cfg.universe.start_date, cfg.universe.end_date)
    
    print("Computing features...")
    X, y = compute_features(raw)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Save processed data
    import os
    os.makedirs("data/processed", exist_ok=True)
    X.to_parquet("data/processed/features.parquet")
    y.to_frame().to_parquet("data/processed/target.parquet")
    
    print("Saved to data/processed/")

if __name__ == "__main__":
    main()
```

---

### Step 2: Test manually

**Run:** `make features`
**Expected:** Files in `data/processed/`

---

### Step 3: Commit

```bash
git add scripts/compute_features.py
git commit -m "feat: add feature computation script"
```

---

## Dependencies

- **Requires:** Tasks 1 (data.py), 2 (features.py)
- **Blocks:** Task 10 (run_experiment.py)

---

## Audit Addendum

Confirmed gaps against PRD Sections 12, 20.5, and 20.8:

- The example re-downloads data but does not explicitly consume the raw cache artifact under `data/raw/` or resolve paths independently of cwd. Load the project cache path and write processed Parquet files under repository `data/processed/`.
- Add offline tests for loading cached OHLCV, preserving feature/target index alignment on disk round-trip, and invoking the script outside repository cwd.
- Persist or expose source data date/version metadata needed by Task 10 reproducibility reporting.

## Success Criteria

- [ ] Script runs without errors
- [ ] Loads raw data from cache
- [ ] Computes features and target
- [ ] Saves to `data/processed/`
- [ ] `make features` works