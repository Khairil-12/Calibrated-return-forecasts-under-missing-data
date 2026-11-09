# Task 1: Data Download and Validation (`data.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement data download from Yahoo Finance with caching, checksumming, and validation.

**Architecture:** Single module `src/calibmiss/data.py` with functions for downloading OHLCV data, caching to parquet with SHA256 checksums in filenames, and validating raw data quality.

**Tech Stack:** Python 3.11+, pandas, numpy, yfinance, hashlib, pathlib.

**Spec:** PRD Sections 6.1-6.3, 20.1, 20.7 (rules 5, 6)

## Global Constraints (from main plan)

- **Python version:** 3.11+ (pinned in pyproject.toml)
- **Reproducibility:** Fresh clone + `make all` regenerates all results. Seeds, library versions, data download date recorded.
- **Code quality:** Type hints on public functions, docstrings, ruff linting, pytest tests.
- **Checksums:** Raw data files saved with SHA256 in filename.

## Audit Gap: Cache Integrity

Cached parquet files must be re-hashed after loading and compared with the SHA256 encoded in the filename. A corrupt or mismatched cache entry must not be returned; discard it and download fresh data. Acceptance: a valid-but-modified parquet payload triggers a new download and yields verified content.

## Review Focus

1. **Cache correctness:** Same tickers/date range must return cached data without re-downloading.
2. **Checksum integrity:** Filename must contain SHA256 of the DataFrame content for corruption detection.
3. **Validation completeness:** Must catch duplicates, negative prices, negative volume, and record native missing rates.

---

### Files
- Create: `src/calibmiss/data.py`
- Create: `tests/test_data.py`

### Interfaces
- **Consumes:** `config/universe.yaml` (tickers, date range, source)
- **Produces:** 
  - Raw parquet files in `data/raw/` with format `{timestamp}_sha256_{hash}.parquet`
  - `download_ohlcv(tickers: list[str], start: str, end: str, cache_dir: str | Path = "data/raw") -> pd.DataFrame`
  - `validate_ohlcv(df: pd.DataFrame) -> ValidationReport` (named tuple/dataclass)

---

### Step 1: Write failing test for data download

```python
# tests/test_data.py
import pytest
import pandas as pd
import numpy as np
from calibmiss.data import download_ohlcv

def test_download_returns_correct_structure():
    df = download_ohlcv(["AAPL", "MSFT"], "2023-01-01", "2023-12-31")
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["ticker", "date"]
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert len(df) > 0
```

**Run:** `pytest tests/test_data.py::test_download_returns_correct_structure -v`
**Expected:** FAIL with "download_ohlcv not defined"

---

### Step 2: Implement `download_ohlcv` in `src/calibmiss/data.py`

**Signature:**
```python
def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/raw"
) -> pd.DataFrame:
```

**Implementation approach:**
- Use `yfinance.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=False, progress=False)`
- Handle single ticker (returns DataFrame) vs multiple tickers (returns dict of DataFrames)
- Stack into single DataFrame with MultiIndex (ticker, date)
- Select and rename columns: Open→open, High→high, Low→low, Close→close, Adj Close→adj_close, Volume→volume
- Compute SHA256 of serialized parquet bytes
- Save as `{timestamp}_sha256_{hash}.parquet` in cache_dir
- On subsequent calls with same tickers/date range, return cached data

---

### Step 3: Write test for caching and checksum

```python
def test_cache_creates_timestamped_parquet_with_checksum(tmp_path):
    df = download_ohlcv(["AAPL"], "2023-01-01", "2023-12-31", cache_dir=tmp_path)
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    assert "_sha256_" in files[0].name
    # Reloading from cache returns same data
    df2 = download_ohlcv(["AAPL"], "2023-01-01", "2023-12-31", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(df, df2)
```

**Run:** `pytest tests/test_data.py::test_cache_creates_timestamped_parquet_with_checksum -v`
**Expected:** PASS after implementation

---

### Step 4: Write test for data validation

```python
def test_validate_data_catches_issues():
    from calibmiss.data import validate_ohlcv, ValidationReport
    
    # Valid data
    df = pd.DataFrame({
        "open": [100, 101], "high": [102, 103], "low": [99, 100],
        "close": [101, 102], "adj_close": [101, 102], "volume": [1000, 2000]
    }, index=pd.MultiIndex.from_tuples([("AAPL", "2023-01-01"), ("AAPL", "2023-01-02")], names=["ticker", "date"]))
    
    report = validate_ohlcv(df)
    assert report.is_valid
    assert report.native_missing_rates is not None
    
    # Duplicate (ticker, date)
    df_dup = pd.concat([df, df.iloc[[0]]])
    report_dup = validate_ohlcv(df_dup)
    assert not report_dup.is_valid
    assert "duplicate" in str(report_dup.errors).lower()
    
    # Negative price
    df_neg = df.copy()
    df_neg.loc[("AAPL", "2023-01-01"), "close"] = -1
    report_neg = validate_ohlcv(df_neg)
    assert not report_neg.is_valid
    
    # Negative volume
    df_negvol = df.copy()
    df_negvol.loc[("AAPL", "2023-01-01"), "volume"] = -100
    report_negvol = validate_ohlcv(df_negvol)
    assert not report_negvol.is_valid
```

---

### Step 5: Implement `validate_ohlcv` and `ValidationReport`

**ValidationReport (dataclass):**
```python
@dataclass
class ValidationReport:
    is_valid: bool
    errors: list[str]
    native_missing_rates: dict[str, float]  # column -> fraction missing
```

**Implementation:**
- Check no duplicate (ticker, date) rows
- Check all price columns > 0
- Check volume >= 0
- Compute native missing rate per column (fraction of NaN)
- Return ValidationReport with is_valid=True/False, list of error messages, and missing rates

---

### Step 6: Run all tests and verify pass

**Run:** `pytest tests/test_data.py -v`
**Expected:** PASS (all tests)

---

### Step 7: Run linting and type checking

**Run:** `make lint typecheck`
**Expected:** No errors

---

### Step 8: Commit

```bash
git add src/calibmiss/data.py tests/test_data.py
git commit -m "feat: add data download, caching, and validation"
```

---

## Dependencies

- **Requires:** config/universe.yaml (already exists)
- **Blocks:** Task 2 (features.py), Task 11 (download_data.py), Task 12 (compute_features.py)

---

## Success Criteria

- [ ] `download_ohlcv` returns DataFrame with correct MultiIndex and columns
- [ ] Caching works: second call returns same data without network request
- [ ] Filenames contain timestamp and SHA256 checksum
- [ ] `validate_ohlcv` catches duplicates, negative prices, negative volume
- [ ] `validate_ohlcv` reports native missing rates per column
- [ ] All tests pass
- [ ] Linting and type checking pass