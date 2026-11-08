# Task 11: Data Download Script (`scripts/download_data.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create executable script to download and cache all universe data.

**Architecture:** Script `scripts/download_data.py` using `calibmiss.data.download_ohlcv`.

**Tech Stack:** Python 3.11+, yaml, calibmiss.data.

**Spec:** PRD Sections 6.1-6.3, 20.5 (make data), 20.6

---

### Files
- Create: `scripts/download_data.py`

### Interfaces
- **Consumes:** `config/universe.yaml`
- **Produces:** Raw parquet files in `data/raw/`

---

### Step 1: Implement script

```python
#!/usr/bin/env python
"""Download and cache OHLCV data for the universe."""
import sys
sys.path.insert(0, "src")

from calibmiss.config import get_config
from calibmiss.data import download_ohlcv

def main():
    cfg = get_config()
    universe = cfg.universe
    
    print(f"Downloading data for {len(universe.tickers)} tickers...")
    print(f"Date range: {universe.start_date} to {universe.end_date}")
    
    df = download_ohlcv(universe.tickers, universe.start_date, universe.end_date)
    
    print(f"Downloaded {len(df)} rows")
    print(f"Tickers: {df.index.get_level_values('ticker').unique().tolist()}")
    print(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    
    # Validate
    from calibmiss.data import validate_ohlcv
    report = validate_ohlcv(df)
    if report.is_valid:
        print("Validation: PASSED")
    else:
        print(f"Validation: FAILED - {report.errors}")
        sys.exit(1)
    
    print(f"Native missing rates: {report.native_missing_rates}")

if __name__ == "__main__":
    main()
```

---

### Step 2: Test manually

**Run:** `make data`
**Expected:** Files in `data/raw/`, validation passes

---

### Step 3: Commit

```bash
git add scripts/download_data.py
git commit -m "feat: add data download script"
```

---

## Dependencies

- **Requires:** Task 1 (data.py)
- **Blocks:** Task 12 (compute_features.py)

---

## Success Criteria

- [ ] Script runs without errors
- [ ] Downloads all 30 tickers
- [ ] Saves cached parquet files with checksums
- [ ] Validation passes
- [ ] `make data` works