# CalibMiss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible research pipeline that trains classifiers to predict next-day stock direction, simulates MCAR/MAR missingness, measures calibration degradation (Brier, ECE, reliability diagrams), and tests whether imputation + calibration methods restore calibration.

**Architecture:** Modular Python package (`src/calibmiss/`) with clear separation: data download → feature engineering → walk-forward splits → missingness simulation → imputation → modeling → calibration → evaluation. Orchestrated via Makefile + config YAMLs. All randomness seeded. No lookahead bias (enforced by tests).

**Tech Stack:** Python 3.11+, pandas/numpy, yfinance, scikit-learn, lightgbm, matplotlib/seaborn, pytest, ruff, mypy, pydantic for config.

**Spec:** `PRD.md` (this directory)

## Global Constraints

- Python 3.11+ (pinned in pyproject.toml)
- All dependencies pinned in requirements.txt
- Walk-forward splits only — no random shuffling
- Purge gap: ≥5 trading days between train and test
- Imputers/calibrators fit on TRAIN ONLY, applied to test
- Seeds fixed via config (5 seeds per config for variance)
- Raw data cached with checksums, never modified in place
- Type hints on public functions; 70%+ test coverage on features/missingness/splits
- Ruff for linting, pytest for tests, mypy for type checking
- Single command `make all` regenerates everything from clean clone

## Review Focus

1. **Lookahead bias** — Any feature using future data (e.g., rolling window including t+1). Test: `test_features_lookahead.py` shifts future data, asserts features unchanged.
2. **Split leakage** — Imputer/calibrator fit on test data or train-test overlap. Test: `test_splits.py` verifies purge gap; `test_imputers.py` verifies fit-on-train-only.
3. **Missingness simulation correctness** — MCAR rate matches target; MAR rule produces expected dependence. Test: `test_missingness.py` checks realized rates and dependence structure.
4. **Calibration fit leakage** — Calibrator fit on test predictions instead of calibration split. Test: `test_calibration.py` verifies calibration split separation.
5. **Reproducibility** — Same config + seed produces identical results. Test: run experiment twice, compare metrics.csv byte-for-byte.

---

### Task 1: Project Scaffold & Config System

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `Makefile`
- Create: `.gitignore`
- Create: `.pre-commit-config.yaml`
- Create: `config/universe.yaml`
- Create: `config/experiment.yaml`
- Create: `config/missingness.yaml`
- Create: `src/calibmiss/__init__.py`
- Create: `src/calibmiss/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config` class (pydantic) loading all three YAMLs; `get_config()` function returning typed config object.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
def test_load_universe_config():
    from calibmiss.config import get_config
    cfg = get_config()
    assert len(cfg.universe.tickers) == 30
    assert cfg.universe.start_date == "2010-01-01"
    assert cfg.universe.end_date == "2024-12-31"

def test_load_experiment_config():
    from calibmiss.config import get_config
    cfg = get_config()
    assert len(cfg.experiment.models) == 2
    assert len(cfg.experiment.imputers) == 3
    assert len(cfg.experiment.calibrators) == 3
    assert cfg.experiment.missingness.mechanisms == ["MCAR", "MAR"]
    assert cfg.experiment.missingness.rates == [0.0, 0.1, 0.2, 0.3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement config system**

Create `src/calibmiss/config.py` with pydantic models for UniverseConfig, ExperimentConfig, MissingnessConfig, and a root Config. Implement `get_config()` that loads YAMLs from `config/` directory.

Create `pyproject.toml` with project metadata, dependencies, and tool configs (ruff, mypy, pytest).

Create `requirements.txt` with pinned versions.

Create `Makefile` with targets: `setup`, `data`, `features`, `experiment`, `figures`, `report`, `all`, `clean`, `clean-all`.

Create `.gitignore` ignoring `data/`, `results/`, `__pycache__/`, `.venv/`, `*.pyc`.

Create `.pre-commit-config.yaml` with ruff, mypy, pytest hooks.

Create three YAML config files per Section 20.3 of PRD.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt Makefile .gitignore .pre-commit-config.yaml config/ src/calibmiss/config.py tests/test_config.py
git commit -m "feat: project scaffold and config system"
```

---

### Task 2: Data Download & Validation

**Files:**
- Create: `src/calibmiss/data.py`
- Create: `scripts/download_data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Config` from `calibmiss.config`
- Produces: `download_ohlcv(tickers, start, end) -> pd.DataFrame` (MultiIndex: ticker, date; columns: open, high, low, close, adj_close, volume); `validate_ohlcv(df) -> ValidationReport`; saves raw parquet to `data/raw/ohlcv_<timestamp>_<checksum>.parquet`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
def test_download_ohlcv_shape():
    from calibmiss.data import download_ohlcv
    df = download_ohlcv(["AAPL", "MSFT"], "2023-01-01", "2023-01-31")
    assert isinstance(df.index, pd.MultiIndex)
    assert list(df.index.names) == ["ticker", "date"]
    assert set(df.columns) == {"open", "high", "low", "close", "adj_close", "volume"}
    assert len(df) > 0

def test_validate_ohlcv():
    from calibmiss.data import validate_ohlcv
    # valid data
    df = pd.DataFrame(...)  # minimal valid fixture
    report = validate_ohlcv(df)
    assert report.is_valid
    # invalid: negative price
    df_bad = df.copy()
    df_bad.loc[0, "close"] = -1
    report = validate_ohlcv(df_bad)
    assert not report.is_valid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL

- [ ] **Step 3: Implement data module**

`src/calibmiss/data.py`:
- `download_ohlcv()`: uses yfinance, handles rate limits, retries, concatenates into MultiIndex DataFrame, saves parquet with timestamp + SHA256 checksum in filename.
- `validate_ohlcv()`: checks no duplicate (ticker, date), prices > 0, volume >= 0, records native missing rate per column. Returns ValidationReport (dataclass with `is_valid`, `errors`, `warnings`, `missing_rates`).
- `load_raw()`: loads latest raw parquet by timestamp.

`scripts/download_data.py`: CLI entry point, loads config, downloads universe, validates, saves.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/data.py scripts/download_data.py tests/test_data.py
git commit -m "feat: data download and validation"
```

---

### Task 3: Feature Engineering + Lookahead Test

**Files:**
- Create: `src/calibmiss/features.py`
- Create: `scripts/compute_features.py`
- Test: `tests/test_features_lookahead.py`

**Interfaces:**
- Consumes: OHLCV DataFrame from `data.py`
- Produces: `compute_features(ohlcv) -> pd.DataFrame` (MultiIndex ticker,date; feature columns per PRD 7.2); `make_target(ohlcv) -> pd.Series` (binary, aligned); saves processed parquet to `data/processed/features_<timestamp>.parquet`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_lookahead.py
def test_no_lookahead_in_features():
    from calibmiss.features import compute_features
    # Create OHLCV where future prices are different
    ohlcv_past = make_ohlcv_fixture(prices=[100, 101, 102, 103, 104])
    ohlcv_future = make_ohlcv_fixture(prices=[200, 201, 202, 203, 204])  # big jump
    # Compute features on past-only
    feats_past = compute_features(ohlcv_past)
    # Compute features on past+future (should not change past features if no lookahead)
    ohlcv_combined = pd.concat([ohlcv_past, ohlcv_future])
    feats_combined = compute_features(ohlcv_combined)
    # Features for past dates must be identical
    pd.testing.assert_frame_equal(feats_past, feats_combined.loc[feats_past.index])

def test_target_definition():
    from calibmiss.features import make_target
    # price goes up -> 1
    # price goes down -> 0
    # zero return -> dropped
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features_lookahead.py -v`
Expected: FAIL

- [ ] **Step 3: Implement features module**

`src/calibmiss/features.py`:
- `compute_features()`: computes all 7 features from PRD 7.2 using only data up to day t. Uses `groupby('ticker')` + rolling/shift to avoid lookahead. Returns feature matrix.
- `make_target()`: `y_t = 1 if adj_close_{t+1} > adj_close_t else 0`, drops zero returns.
- Feature list: `ret_1d`, `ret_5d`, `ret_20d`, `vol_5d`, `vol_20d`, `vol_ratio`, `hl_range`, `price_vs_ma50`.

`scripts/compute_features.py`: loads raw data, computes features + target, saves processed.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features_lookahead.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/features.py scripts/compute_features.py tests/test_features_lookahead.py
git commit -m "feat: feature engineering with lookahead test"
```

---

### Task 4: Walk-Forward Splits with Purge Gap

**Files:**
- Create: `src/calibmiss/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: feature matrix + target (aligned MultiIndex)
- Produces: `WalkForwardSplitter` class with `split(X, y) -> Generator[train_idx, calib_idx, test_idx]`; each split returns integer position indices; purge gap enforced.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
def test_purge_gap_enforced():
    from calibmiss.splits import WalkForwardSplitter
    splitter = WalkForwardSplitter(train_years=3, calibrate_months=6, test_months=6, roll_forward_months=6, purge_days=5)
    # Create dummy index with known dates
    dates = pd.date_range("2010-01-01", periods=1000, freq="B")
    X = pd.DataFrame(index=dates)
    y = pd.Series(index=dates)
    for train_idx, calib_idx, test_idx in splitter.split(X, y):
        # Max train date + purge_days < min calib date
        assert dates[train_idx].max() + pd.Timedelta(days=5) < dates[calib_idx].min()
        # Max calib date + purge_days < min test date
        assert dates[calib_idx].max() + pd.Timedelta(days=5) < dates[test_idx].min()

def test_no_overlap():
    from calibmiss.splits import WalkForwardSplitter
    splitter = WalkForwardSplitter(...)
    for train_idx, calib_idx, test_idx in splitter.split(X, y):
        assert len(set(train_idx) & set(calib_idx)) == 0
        assert len(set(calib_idx) & set(test_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_splits.py -v`
Expected: FAIL

- [ ] **Step 3: Implement splits module**

`src/calibmiss/splits.py`:
- `WalkForwardSplitter` class with config params.
- `split()`: yields (train_idx, calib_idx, test_idx) as integer arrays.
- Uses business day frequency, accounts for holidays implicitly via data index.
- Purge gap: drops `purge_days` rows after train end before calib start, and after calib end before test start.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_splits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/splits.py tests/test_splits.py
git commit -m "feat: walk-forward splits with purge gap"
```

---

### Task 5: Missingness Simulation (MCAR + MAR)

**Files:**
- Create: `src/calibmiss/missingness.py`
- Test: `tests/test_missingness.py`

**Interfaces:**
- Consumes: feature matrix (complete), config (mechanism, rate, seed, MAR rules)
- Produces: `simulate_missingness(X, mechanism, rate, seed, mar_rules=None) -> X_missing, mask` where `mask` is boolean DataFrame (True=missing); `MCAR` and `MAR` classes with `apply(X) -> X_missing, mask`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_missingness.py
def test_mcar_rate():
    from calibmiss.missingness import MCAR
    X = pd.DataFrame(np.random.randn(1000, 10))
    mcar = MCAR(rate=0.2, seed=42)
    X_miss, mask = mcar.apply(X)
    realized = mask.mean().mean()
    assert abs(realized - 0.2) < 0.01  # within 1%

def test_mcar_reproducible():
    from calibmiss.missingness import MCAR
    X = pd.DataFrame(np.random.randn(100, 5))
    mcar1 = MCAR(rate=0.3, seed=42)
    mcar2 = MCAR(rate=0.3, seed=42)
    _, mask1 = mcar1.apply(X)
    _, mask2 = mcar2.apply(X)
    assert mask1.equals(mask2)

def test_mar_dependence():
    from calibmiss.missingness import MAR
    # Create data where feature A depends on feature B
    X = pd.DataFrame({"A": np.random.randn(1000), "B": np.random.randn(1000)})
    # MAR rule: A missing when B > 0
    mar = MAR(rate=0.2, seed=42, rules={"A": {"depends_on": "B", "threshold": 0}})
    X_miss, mask = mar.apply(X)
    # Missing rate of A should be higher when B > 0
    miss_rate_b_pos = mask.loc[X["B"] > 0, "A"].mean()
    miss_rate_b_neg = mask.loc[X["B"] <= 0, "A"].mean()
    assert miss_rate_b_pos > miss_rate_b_neg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_missingness.py -v`
Expected: FAIL

- [ ] **Step 3: Implement missingness module**

`src/calibmiss/missingness.py`:
- `MCAR`: each value independently missing with prob `p`. Uses `np.random.Generator(seed)`.
- `MAR`: probability of missing depends on observed variable per `config/missingness.yaml` rules. Implements quantile-threshold rule: `p_missing = base_prob + (max_prob - base_prob) * I(depends_on > quantile_threshold)`.
- Both return `(X_missing, mask)` where `X_missing` has NaN at masked positions.
- `simulate_missingness()`: dispatcher function.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_missingness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/missingness.py tests/test_missingness.py
git commit -m "feat: MCAR and MAR missingness simulation"
```

---

### Task 6: Imputation Strategies

**Files:**
- Create: `src/calibmiss/imputers.py`
- Test: `tests/test_imputers.py`

**Interfaces:**
- Consumes: training feature matrix (with NaN), test feature matrix (with NaN)
- Produces: `Imputer` base class + `ForwardFillImputer`, `MedianImputer`, `IterativeImputer` subclasses; each has `fit(X_train) -> self`, `transform(X) -> X_imputed`; fit ONLY on train, transform on both.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imputers.py
def test_imputer_fit_on_train_only():
    from calibmiss.imputers import MedianImputer
    X_train = pd.DataFrame({"A": [1, np.nan, 3], "B": [np.nan, 2, 3]})
    X_test = pd.DataFrame({"A": [np.nan, 5], "B": [6, np.nan]})
    imp = MedianImputer()
    imp.fit(X_train)
    X_train_imp = imp.transform(X_train)
    X_test_imp = imp.transform(X_test)
    # Train median used for both
    assert X_train_imp["A"].iloc[1] == 2.0  # median of [1,3]
    assert X_test_imp["A"].iloc[0] == 2.0   # same median

def test_forward_fill_within_ticker():
    from calibmiss.imputers import ForwardFillImputer
    # MultiIndex with ticker
    idx = pd.MultiIndex.from_tuples([("AAPL", 1), ("AAPL", 2), ("AAPL", 3), ("MSFT", 1), ("MSFT", 2)])
    X = pd.DataFrame({"feat": [1, np.nan, 3, np.nan, 5]}, index=idx)
    imp = ForwardFillImputer()
    imp.fit(X)
    X_imp = imp.transform(X)
    # AAPL: 1 -> 1 -> 3 (ffill)
    # MSFT: nan -> 5 (first is nan, stays nan? or use global? PRD says within ticker)
    assert X_imp.loc[("AAPL", 2), "feat"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_imputers.py -v`
Expected: FAIL

- [ ] **Step 3: Implement imputers module**

`src/calibmiss/imputers.py`:
- `BaseImputer` (ABC) with `fit()`, `transform()`.
- `ForwardFillImputer`: groupby ticker, ffill within group. Handles leading NaNs (leave as NaN or fill with global median — document choice).
- `MedianImputer`: computes median per column on train, fills train + test.
- `IterativeImputer`: wraps `sklearn.impute.IterativeImputer` with config params.
- All imputers store fitted statistics (medians, etc.) as attributes.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_imputers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/imputers.py tests/test_imputers.py
git commit -m "feat: three imputation strategies"
```

---

### Task 7: Models (Logistic Regression + LightGBM)

**Files:**
- Create: `src/calibmiss/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: imputed feature matrix (train), target (train)
- Produces: `Model` base class + `LogisticRegressionModel`, `LightGBMModel`; each has `fit(X, y) -> self`, `predict_proba(X) -> np.ndarray` (shape n_samples, 2), `predict(X) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
def test_logistic_regression_proba():
    from calibmiss.models import LogisticRegressionModel
    X = pd.DataFrame({"f1": [1,2,3,4], "f2": [0.1,0.2,0.3,0.4]})
    y = pd.Series([0,0,1,1])
    model = LogisticRegressionModel(C=1.0, max_iter=1000)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()

def test_lightgbm_proba():
    from calibmiss.models import LightGBMModel
    X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
    y = pd.Series(np.random.randint(0, 2, 100))
    model = LightGBMModel(n_estimators=10, max_depth=3)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (100, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL

- [ ] **Step 3: Implement models module**

`src/calibmiss/models.py`:
- `LogisticRegressionModel`: wraps `sklearn.linear_model.LogisticRegression` with StandardScaler in pipeline. Params from config.
- `LightGBMModel`: wraps `lightgbm.LGBMClassifier`. Params from config. Handles categorical features if any.
- Both implement common interface: `fit(X, y)`, `predict_proba(X)`, `predict(X)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/models.py tests/test_models.py
git commit -m "feat: logistic regression and LightGBM models"
```

---

### Task 8: Calibration Methods (Platt + Isotonic)

**Files:**
- Create: `src/calibmiss/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: base model predictions on calibration set (proba, y_true)
- Produces: `Calibrator` base class + `PlattCalibrator`, `IsotonicCalibrator`, `NoCalibration`; each has `fit(proba_calib, y_calib) -> self`, `calibrate(proba_test) -> proba_calibrated`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration.py
def test_platt_calibration():
    from calibmiss.calibration import PlattCalibrator
    # Perfectly calibrated input
    proba = np.array([[0.3, 0.7], [0.6, 0.4], [0.8, 0.2]])
    y = np.array([1, 0, 1])
    cal = PlattCalibrator()
    cal.fit(proba, y)
    out = cal.calibrate(proba)
    assert out.shape == proba.shape

def test_isotonic_monotonic():
    from calibmiss.calibration import IsotonicCalibrator
    proba = np.linspace(0.1, 0.9, 9).reshape(-1, 1)
    proba = np.hstack([1-proba, proba])
    y = (proba[:,1] > 0.5).astype(int)  # step function
    cal = IsotonicCalibrator()
    cal.fit(proba, y)
    out = cal.calibrate(proba)
    # Isotonic should be monotonic
    assert np.all(np.diff(out[:,1]) >= -1e-6)

def test_no_calibration_identity():
    from calibmiss.calibration import NoCalibration
    cal = NoCalibration()
    cal.fit(None, None)
    proba = np.array([[0.3, 0.7], [0.6, 0.4]])
    out = cal.calibrate(proba)
    assert np.allclose(out, proba)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL

- [ ] **Step 3: Implement calibration module**

`src/calibmiss/calibration.py`:
- `NoCalibration`: identity transform.
- `PlattCalibrator`: wraps `sklearn.calibration.CalibratedClassifierCV(method='sigmoid', cv='prefit')` on a dummy estimator, or directly fits logistic regression on logits.
- `IsotonicCalibrator`: wraps `sklearn.calibration.CalibratedClassifierCV(method='isotonic', cv='prefit')`.
- All handle binary classification (use positive class probability).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/calibration.py tests/test_calibration.py
git commit -m "feat: Platt and isotonic calibration"
```

---

### Task 9: Metrics (Brier, ECE, Reliability Diagram, AUC)

**Files:**
- Create: `src/calibmiss/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: predicted probabilities (n, 2), true labels (n,)
- Produces: `compute_brier(proba, y) -> float`; `compute_ece(proba, y, n_bins=10) -> float`; `compute_roc_auc(proba, y) -> float`; `reliability_diagram_data(proba, y, n_bins=10) -> (bin_centers, bin_accuracies, bin_counts, bin_confidences)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
def test_brier_score():
    from calibmiss.metrics import compute_brier
    # Perfect predictions
    proba = np.array([[0, 1], [1, 0], [0, 1]])
    y = np.array([1, 0, 1])
    assert compute_brier(proba, y) == 0.0
    # Worst predictions
    proba = np.array([[1, 0], [0, 1], [1, 0]])
    assert compute_brier(proba, y) == 1.0

def test_ece():
    from calibmiss.metrics import compute_ece
    # Perfect calibration
    proba = np.array([[0.3, 0.7], [0.7, 0.3]])
    y = np.array([1, 0])
    assert compute_ece(proba, y, n_bins=2) == 0.0

def test_reliability_diagram_data():
    from calibmiss.metrics import reliability_diagram_data
    proba = np.array([[0.2, 0.8], [0.3, 0.7], [0.7, 0.3], [0.8, 0.2]])
    y = np.array([1, 1, 0, 0])
    centers, accs, counts, confs = reliability_diagram_data(proba, y, n_bins=2)
    assert len(centers) == 2
    assert np.all(counts > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement metrics module**

`src/calibmiss/metrics.py`:
- `compute_brier()`: mean squared error of positive class probability.
- `compute_ece()`: equal-width bins on positive class probability; weighted avg of |accuracy - confidence| per bin.
- `compute_roc_auc()`: `sklearn.metrics.roc_auc_score`.
- `reliability_diagram_data()`: returns data for plotting.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/metrics.py tests/test_metrics.py
git commit -m "feat: calibration and discrimination metrics"
```

---

### Task 10: Experiment Pipeline Orchestration

**Files:**
- Create: `src/calibmiss/pipeline.py`
- Create: `src/calibmiss/run_experiment.py`
- Create: `scripts/run_grid.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: all config, data, features, splits, missingness, imputers, models, calibration, metrics
- Produces: `run_single_config(config, fold, seed) -> dict` (metrics row); `run_grid()` loops over all combinations, saves `results/metrics.csv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
def test_run_single_config():
    from calibmiss.pipeline import run_single_config
    # Minimal config dict for one combination
    cfg = {...}  # fixture
    result = run_single_config(cfg, fold=0, seed=42)
    assert "brier" in result
    assert "ece" in result
    assert "roc_auc" in result
    assert result["model"] == "logistic_regression"
    assert result["imputer"] == "median"
    # ... all config fields present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement pipeline**

`src/calibmiss/pipeline.py`:
- `run_single_config(config, fold_idx, seed)`: 
  1. Load features + target for this fold (train/calib/test indices from splitter)
  2. Simulate missingness on train/calib/test separately with seed
  3. Fit imputer on train, transform all three
  4. Fit model on imputed train
  5. Get proba on calib set, fit calibrator
  6. Get proba on test set, apply calibrator
  7. Compute metrics on test set
  8. Return dict with all config params + metrics + fold + seed

`src/calibmiss/run_experiment.py`: CLI entry for single config (for debugging).

`scripts/run_grid.py`: 
- Loads full experiment config
- Iterates over all combinations (models × imputers × calibrators × mechanisms × rates × seeds × folds)
- Parallelizes over folds/seeds (joblib or multiprocessing)
- Appends each result row to `results/metrics.csv` (incremental write)
- Logs progress.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/calibmiss/pipeline.py src/calibmiss/run_experiment.py scripts/run_grid.py tests/test_pipeline.py
git commit -m "feat: experiment pipeline orchestration"
```

---

### Task 11: Figure Generation

**Files:**
- Create: `scripts/generate_figures.py`
- Test: `tests/test_figures.py` (smoke test)

**Interfaces:**
- Consumes: `results/metrics.csv`
- Produces: PNG files in `results/figures/` per PRD 20.8.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figures.py
def test_generate_figures_creates_files(tmp_path):
    from scripts.generate_figures import main
    # Create minimal metrics.csv
    metrics = pd.DataFrame([...])  # fixture with all required columns
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    # Run figure generation
    main(input_csv=tmp_path / "metrics.csv", output_dir=tmp_path / "figures")
    # Check all expected files exist
    expected = [
        "brier_vs_missingness.png",
        "ece_vs_missingness.png",
        "reliability_0pct.png",
        "reliability_30pct.png",
        "heatmap_calibration.png"
    ]
    for f in expected:
        assert (tmp_path / "figures" / f).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_figures.py -v`
Expected: FAIL

- [ ] **Step 3: Implement figure generation**

`scripts/generate_figures.py`:
- Loads metrics.csv
- For each mechanism (MCAR, MAR), creates:
  - Line plot: Brier vs missing rate, grouped by imputer, faceted by calibrator
  - Line plot: ECE vs missing rate, same grouping
  - Reliability diagrams at 0% and 30% (average across seeds/folds)
  - Heatmap: mean ECE (rows=imputer, cols=calibrator) for each model×mechanism×rate
- Saves to `results/figures/`
- Uses matplotlib/seaborn with consistent styling.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_figures.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_figures.py tests/test_figures.py
git commit -m "feat: figure generation from results"
```

---

### Task 12: Report Generation & README

**Files:**
- Create: `reports/report.md` (template with placeholders)
- Create: `reports/summary.md` (template)
- Create: `README.md`
- Create: `scripts/generate_report.py` (optional, fills templates)

**Interfaces:**
- Consumes: `results/metrics.csv`, `results/figures/*.png`
- Produces: Final written report (Markdown → PDF via pandoc)

- [ ] **Step 1: Write failing test (smoke)**

```python
# tests/test_report.py
def test_report_exists():
    assert Path("reports/report.md").exists()
    assert Path("reports/summary.md").exists()
    assert Path("README.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL

- [ ] **Step 3: Create documentation**

`README.md`: 
- Project overview, quickstart (`make all`), repo structure, how to reproduce, config explanation.

`reports/report.md`:
- Template with sections: Abstract, Introduction, Data, Methods, Experimental Design, Results, Discussion, Limitations, Conclusion. Placeholders for tables/figures.

`reports/summary.md`:
- 1-page executive summary template.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md reports/report.md reports/summary.md
git commit -m "docs: README and report templates"
```

---

### Task 13: End-to-End Integration Test

**Files:**
- Test: `tests/test_e2e.py`

**Interfaces:**
- Runs `make all` (or subset) on reduced config (1 model, 1 imputer, 1 calibrator, 1 mechanism, 1 rate, 1 seed, 1 fold) and verifies outputs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2e.py
@pytest.mark.slow
def test_e2e_minimal(tmp_path):
    # Run pipeline with minimal config
    # Verify metrics.csv exists and has expected columns
    # Verify figures generated
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_e2e.py -v`
Expected: FAIL

- [ ] **Step 3: Fix any integration issues**

Run `make data`, `make features`, `make experiment` (with minimal config), `make figures`. Debug any failures.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end integration test"
```

---

### Task 14: Final Polish & Full Run

**Files:**
- Modify: various (bug fixes, documentation updates)

**Interfaces:**
- None new

- [ ] **Step 1: Run full pipeline with real config**

```bash
make clean-all
make all
```

- [ ] **Step 2: Verify all outputs**

- `results/metrics.csv` has 144 configs × 5 seeds × n_folds rows
- All figures in `results/figures/`
- Report compiled to PDF
- All tests pass: `pytest -v`

- [ ] **Step 3: Fill in report with actual results**

Edit `reports/report.md` with real numbers, tables, figure references.

- [ ] **Step 4: Commit final version**

```bash
git add -A
git commit -m "feat: complete CalibMiss pipeline with results"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every FR-1 through FR-9 from PRD Section 11 mapped to a task.
- [ ] **Step scan:** Each step writes one reasonable thing; no TBDs; no implementation bodies where signature+test suffice.
- [ ] **Type consistency:** Interfaces use consistent names (`fit`/`transform`/`predict_proba`/`calibrate`); config keys match YAML.
- [ ] **Review Focus:** All 5 focus areas have tests in owning tasks (lookahead→Task 3, split leakage→Task 4, missingness→Task 5, calibration leakage→Task 8, reproducibility→Task 10+13).
- [ ] **Proportion:** Plan is ~14 tasks for ~300-line spec — reasonable ratio.

---

**Plan complete and saved to `docs/superpowers/plans/2026-11-08-calibmiss.md`. Please review the plan. Which execution approach would you prefer?**

- **Subagent-driven** - A fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; costs a fresh context per task and per review.
- **Native** - I implement every task myself in this session, the way this harness runs work, then one fresh reviewer on the most capable model checks the whole branch. Cheapest and fastest; no independent review until the end. Runs well with a mid-tier session model, since the plan carries the design.

**For this plan I recommend Native, because the tasks are highly sequential (each depends on the previous's interfaces) and the plan already specifies exact signatures — independent review per task would add context-switching overhead without much design independence. Does the plan capture what you want, and which approach should we use?**