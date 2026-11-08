# CalibMiss Component Documentation

> **Purpose:** Comprehensive documentation of every module, structure, and component in the CalibMiss project before implementation begins. This serves as the reference architecture for all subsequent implementation tasks.

---

## 1. Project Overview

**Project:** CalibMiss - Calibrated Return Forecasts Under Missing Data  
**Goal:** Build a reproducible research pipeline to study how missing data affects calibration of next-day stock direction classifiers, and whether imputation + calibration methods restore calibration.

**Key Research Questions:**
1. How much does calibration degrade as missingness increases (0%, 10%, 20%, 30%) under MCAR vs MAR?
2. Do imputation methods (forward fill, median, iterative) combined with calibration (Platt, isotonic) recover calibration?
3. What is the trade-off between calibration recovery and discrimination (AUC)?

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CALIBMISS PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐          │
│  │  DATA    │───▶│ FEATURES │───▶│ SPLITS   │───▶│ MISSINGNESS  │          │
│  │ DOWNLOAD │    │ ENGINEER │    │ WALK-FWD │    │ SIMULATION   │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────┬───────┘          │
│                                                          │                  │
│                                               ┌──────────▼───────┐          │
│                                               │   IMPUTATION   │          │
│                                               │ (fit on train) │          │
│                                               └────────┬───────┘          │
│                                                        │                  │
│                                               ┌────────▼───────┐          │
│                                               │    MODELS      │          │
│                                               │ (LR + LightGBM)│          │
│                                               └────────┬───────┘          │
│                                                        │                  │
│                                               ┌────────▼───────┐          │
│                                               │  CALIBRATION   │          │
│                                               │ (fit on calib) │          │
│                                               └────────┬───────┘          │
│                                                        │                  │
│                                               ┌────────▼───────┐          │
│                                               │  EVALUATION    │          │
│                                               │ (Brier, ECE,   │          │
│                                               │  AUC, Reliab.) │          │
│                                               └────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Design Principles:**
- **Modularity:** Each component is a separate Python module with clear interfaces
- **No Lookahead Bias:** Enforced by unit tests; features use only data up to day t
- **Train-Only Fitting:** Imputers, scalers, calibrators fit ONLY on training data
- **Walk-Forward Only:** No random shuffling; time-ordered splits with purge gaps
- **Reproducibility:** All randomness controlled via seeds in config
- **Config-Driven:** Experimental grid defined in YAML configs

---

## 3. Directory Structure

```
calibmiss/
├── README.md                    # Project overview, quickstart, reproduction guide
├── PRD.md                       # Product Requirements Document (this project's spec)
├── Makefile                     # Pipeline orchestration targets
├── pyproject.toml               # Project metadata, dependencies, tool configs
├── requirements.txt             # Pinned dependencies for reproducible installs
├── .gitignore                   # Ignores data/, results/, venv/, caches
├── .pre-commit-config.yaml      # Pre-commit hooks (ruff, mypy, pytest)
├── config/
│   ├── universe.yaml            # 30 tickers, date range, data source
│   ├── experiment.yaml          # Full experimental grid parameters
│   └── missingness.yaml         # MAR dependence rules
├── data/
│   ├── raw/                     # gitignored - timestamped OHLCV parquet files
│   └── processed/               # gitignored - feature matrices, splits
├── src/calibmiss/
│   ├── __init__.py              # Package init, version
│   ├── config.py                # Pydantic config loading with caching
│   ├── data.py                  # Download, cache, validate OHLCV
│   ├── features.py              # Feature engineering + lookahead test
│   ├── missingness.py           # MCAR/MAR simulation
│   ├── imputers.py              # Imputation strategies (fit on train only)
│   ├── calibration.py           # Platt / isotonic calibration
│   ├── splits.py                # Walk-forward splits with purge gap
│   ├── metrics.py               # Brier, ECE, reliability diagrams, AUC
│   ├── models.py                # Model wrappers (LR, LightGBM)
│   ├── pipeline.py              # Orchestration: run one config
│   └── run_experiment.py        # CLI entry for single config
├── tests/
│   ├── __init__.py
│   ├── test_config.py           # Config loading tests
│   ├── test_data.py             # Data download/validation tests
│   ├── test_features_lookahead.py  # Lookahead bias tests
│   ├── test_missingness.py      # MCAR/MAR simulation tests
│   ├── test_splits.py           # Walk-forward split tests
│   ├── test_imputers.py         # Imputer fit-on-train tests
│   ├── test_calibration.py      # Calibration fit-on-calib tests
│   ├── test_metrics.py          # Metric correctness tests
│   ├── test_models.py           # Model interface tests
│   ├── test_pipeline.py         # Single config run test
│   ├── test_figures.py          # Figure generation smoke test
│   ├── test_report.py           # Report files exist test
│   └── test_e2e.py              # End-to-end integration test
├── scripts/
│   ├── download_data.py         # CLI: download OHLCV data
│   ├── compute_features.py      # CLI: compute features + target
│   ├── run_grid.py              # CLI: run full experimental grid
│   └── generate_figures.py      # CLI: generate plots from results
├── notebooks/
│   └── exploration.ipynb        # Ad-hoc exploration
├── results/
│   ├── metrics.csv              # All experiment results (one row per config)
│   └── figures/                 # Auto-generated plots
└── reports/
    ├── report.md                # 6-10 page written report
    └── summary.md               # 1-page executive summary
```

---

## 4. Component Specifications

### 4.1 Config System (`src/calibmiss/config.py`)

**Responsibility:** Load and validate all YAML configuration files using Pydantic models.

**Configuration Files:**
| File | Purpose | Key Contents |
|------|---------|--------------|
| `universe.yaml` | Data universe | 30 tickers, 2010-01-01 to 2024-12-31, source=yfinance |
| `experiment.yaml` | Experimental grid | Models, imputers, calibrators, missingness, splits, evaluation |
| `missingness.yaml` | MAR rules | Dependence structure for MAR simulation |

**Pydantic Models:**
```python
UniverseConfig: tickers, start_date, end_date, source
ModelConfig: name, params
ImputerConfig: name, params
CalibratorConfig: name, params
MissingnessConfig: mechanisms, rates, seeds
SplitsConfig: train_years, calibrate_months, test_months, roll_forward_months, purge_days
EvaluationConfig: n_bins_ece, metrics
ExperimentConfig: models, imputers, calibrators, missingness, splits, evaluation
MARRuleConfig: target_feature, depends_on, rule_type, params
Config: universe, experiment
```

**Functions:**
- `get_config(config_dir="config") -> Config` - Loads universe + experiment configs (cached)
- `get_mar_rules(config_dir="config") -> MissingnessRulesConfig` - Loads MAR rules
- `reset_config()` - Clears cache (for testing)

**Validation:**
- All fields validated by Pydantic
- Ticker count must be 30
- Mechanisms must include "MCAR" and "MAR"
- Rates must be [0.0, 0.1, 0.2, 0.3]
- Seeds must be 5 integers

---

### 4.2 Data Module (`src/calibmiss/data.py`)

**Responsibility:** Download OHLCV data from Yahoo Finance, validate, and cache with checksums.

**Inputs:** 
- `tickers: list[str]`
- `start_date: str` (YYYY-MM-DD)
- `end_date: str` (YYYY-MM-DD)

**Outputs:**
- `pd.DataFrame` with MultiIndex (ticker, date) and columns: open, high, low, close, adj_close, volume
- Saved to `data/raw/ohlcv_<timestamp>_<sha256>.parquet`

**Functions:**
```python
download_ohlcv(tickers, start, end) -> pd.DataFrame
    # Uses yfinance with retries, rate limiting
    # Concatenates into MultiIndex DataFrame
    # Saves parquet with timestamp + SHA256 in filename

validate_ohlcv(df) -> ValidationReport
    # Checks: no duplicate (ticker, date), prices > 0, volume >= 0
    # Records native missing rate per column
    # Returns ValidationReport(is_valid, errors, warnings, missing_rates)

load_raw() -> pd.DataFrame
    # Loads latest raw parquet by timestamp
```

**ValidationReport (dataclass):**
- `is_valid: bool`
- `errors: list[str]`
- `warnings: list[str]`
- `missing_rates: dict[str, float]` (per column)

**Error Handling:**
- Retries on network errors (exponential backoff)
- Skips tickers that fail after max retries (logs warning)
- Validates data integrity before saving

---

### 4.3 Features Module (`src/calibmiss/features.py`)

**Responsibility:** Compute all features and target using only data available at time t (no lookahead).

**Inputs:** OHLCV DataFrame (MultiIndex: ticker, date)

**Outputs:**
- Feature matrix: MultiIndex (ticker, date), 8 feature columns
- Target series: MultiIndex (ticker, date), binary (1=up, 0=down), zero-return days dropped
- Saved to `data/processed/features_<timestamp>.parquet`

**Features (per PRD Section 7.2):**
| Feature | Description | Formula |
|---------|-------------|---------|
| `ret_1d` | 1-day log return | log(close_t / close_{t-1}) |
| `ret_5d` | 5-day log return | log(close_t / close_{t-5}) |
| `ret_20d` | 20-day log return | log(close_t / close_{t-20}) |
| `vol_5d` | 5-day rolling volatility | std(ret_1d) over 5 days |
| `vol_20d` | 20-day rolling volatility | std(ret_1d) over 20 days |
| `vol_ratio` | Volume ratio | volume_t / mean(volume_{t-20:t}) |
| `hl_range` | High-low range | (high_t - low_t) / close_t |
| `price_vs_ma50` | Price vs 50-day MA | close_t / mean(close_{t-50:t}) |

**Target (per PRD Section 7.1):**
- `y_t = 1` if adj_close_{t+1} > adj_close_t, else `0`
- Days with zero return dropped

**Critical Implementation Rules:**
- ALL features computed using `groupby('ticker')` + `shift(1)` for returns
- Rolling windows use `.rolling(window).std()` on shifted returns
- NO future data used - verified by lookahead test
- Features aligned with target (same index)

**Functions:**
```python
compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame
make_target(ohlcv: pd.DataFrame) -> pd.Series
```

**Lookahead Test (tests/test_features_lookahead.py):**
- Create OHLCV with known past prices
- Append different future prices
- Compute features on past-only vs past+future
- Assert features for past dates are IDENTICAL
- If different → lookahead bias detected

---

### 4.4 Splits Module (`src/calibmiss/splits.py`)

**Responsibility:** Generate walk-forward (time-ordered) splits with purge gaps.

**Configuration (from experiment.yaml):**
- `train_years: 3`
- `calibrate_months: 6`
- `test_months: 6`
- `roll_forward_months: 6`
- `purge_days: 5`

**Split Structure:**
```
Time ──────────────────────────────────────────────────────────────────▶
[ TRAIN (3 yrs) ][PURGE 5d][ CALIB (6 mo) ][PURGE 5d][ TEST (6 mo) ]  ◀ Fold 0
                    [ TRAIN (3 yrs) ][PURGE 5d][ CALIB (6 mo) ][PURGE 5d][ TEST (6 mo) ]  ◀ Fold 1
```

**Functions:**
```python
class WalkForwardSplitter:
    def __init__(self, train_years, calibrate_months, test_months, 
                 roll_forward_months, purge_days)
    def split(X, y) -> Generator[train_idx, calib_idx, test_idx]
        # Returns integer position indices
        # Enforces purge gaps between train/calib and calib/test
```

**Purge Gap Logic:**
- After train end: drop `purge_days` rows before calib start
- After calib end: drop `purge_days` rows before test start
- Prevents overlap from rolling-window features (e.g., 20-day vol uses t-20 to t)

**Tests (tests/test_splits.py):**
- Verify purge gap: `train_max + 5d < calib_min`
- Verify no overlap: train ∩ calib = ∅, calib ∩ test = ∅, train ∩ test = ∅
- Verify chronological order: train < calib < test

---

### 4.5 Missingness Module (`src/calibmiss/missingness.py`)

**Responsibility:** Simulate MCAR and MAR missingness on feature matrices.

**MCAR (Missing Completely At Random):**
- Each feature value independently missing with probability `p`
- `p ∈ {0.0, 0.1, 0.2, 0.3}`
- No dependence on any variable

**MAR (Missing At Random):**
- Missing probability depends on OBSERVED variable (not the missing value itself)
- Rule defined in `missingness.yaml`:
  ```yaml
  mar_rules:
    - target_feature: "volatility_20d"
      depends_on: "volume_ratio"
      rule_type: "quantile_threshold"
      params:
        base_prob: 0.05
        max_prob: 0.5
        quantile_threshold: 0.7
  ```
- Interpretation: P(missing | volume_ratio > Q0.7) = max_prob, else base_prob
- Average missing rate calibrated to match target `p`

**Functions:**
```python
class MCAR:
    def __init__(self, rate: float, seed: int)
    def apply(X: pd.DataFrame) -> tuple[X_missing, mask]
        # X_missing: copy with NaN at masked positions
        # mask: boolean DataFrame (True = missing)

class MAR:
    def __init__(self, rate: float, seed: int, rules: list[MARRuleConfig])
    def apply(X: pd.DataFrame) -> tuple[X_missing, mask]

def simulate_missingness(X, mechanism, rate, seed, mar_rules=None) -> tuple[X_missing, mask]
```

**Reproducibility:**
- Uses `np.random.Generator(seed)` for deterministic results
- Same seed + same input = same mask

**Tests (tests/test_missingness.py):**
- MCAR: realized rate ≈ target rate (within 1%)
- MCAR: same seed produces identical mask
- MAR: missing rate higher when depends_on > threshold
- Both: mask shape matches input, NaN only at True positions

---

### 4.6 Imputers Module (`src/calibmiss/imputers.py`)

**Responsibility:** Three imputation strategies, ALL fit on TRAIN only, transform train/calib/test.

**Imputation Strategies:**

| Method | Description | Fit On | Transform On |
|--------|-------------|--------|--------------|
| `forward_fill` | Carry last observed forward within ticker | Train | Train, Calib, Test |
| `median` | Replace with training-set median per column | Train | Train, Calib, Test |
| `iterative` | MICE-style (sklearn IterativeImputer) | Train | Train, Calib, Test |

**Critical Rule:** `fit()` called ONLY on training data. `transform()` applied to all splits.

**Classes:**
```python
class BaseImputer(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> Self
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame

class ForwardFillImputer(BaseImputer):
    # groupby ticker, ffill within group
    # Leading NaNs: leave as NaN (handled by median fallback?)

class MedianImputer(BaseImputer):
    # Stores column medians from fit()
    # transform() fills NaN with stored medians

class IterativeImputer(BaseImputer):
    # Wraps sklearn.impute.IterativeImputer
    # Stores fitted imputer
```

**Tests (tests/test_imputers.py):**
- Median imputer: train median used for test transform
- Forward fill: works within ticker groups
- All imputers: fit on train, transform on test produces no NaN (if train had no all-NaN columns)
- No data leakage: imputer state doesn't change after fit

---

### 4.7 Models Module (`src/calibmiss/models.py`)

**Responsibility:** Wrapper classes for the two model families with unified interface.

**Models:**

| Model | Library | Key Parameters |
|-------|---------|----------------|
| Logistic Regression | sklearn.linear_model | C=1.0, max_iter=1000, class_weight=balanced |
| LightGBM | lightgbm.LGBMClassifier | n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=31 |

**Unified Interface:**
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray  # shape (n, 2)
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray  # shape (n,)

class LogisticRegressionModel(BaseModel):
    # Uses sklearn Pipeline: StandardScaler -> LogisticRegression
    # predict_proba returns probabilities for both classes

class LightGBMModel(BaseModel):
    # Direct LGBMClassifier
    # Handles categorical features if any
```

**Tests (tests/test_models.py):**
- Both models: fit/predict_proba/predict work
- predict_proba: shape (n, 2), sums to 1, values in [0,1]
- Probabilities are calibrated-able (not all 0 or 1)

---

### 4.8 Calibration Module (`src/calibmiss/calibration.py`)

**Responsibility:** Post-hoc calibration methods fitted on CALIBRATION split only.

**Calibrators:**

| Method | Description | sklearn Wrapper |
|--------|-------------|-----------------|
| `none` | Identity (no calibration) | - |
| `platt` | Logistic regression on logits | CalibratedClassifierCV(method='sigmoid') |
| `isotonic` | Non-parametric monotone mapping | CalibratedClassifierCV(method='isotonic') |

**Unified Interface:**
```python
class BaseCalibrator(ABC):
    @abstractmethod
    def fit(self, proba_calib: np.ndarray, y_calib: np.ndarray) -> Self
    @abstractmethod
    def calibrate(self, proba_test: np.ndarray) -> np.ndarray

class NoCalibration(BaseCalibrator):
    def fit(self, proba, y): pass
    def calibrate(self, proba): return proba

class PlattCalibrator(BaseCalibrator):
    # Fits on positive class probability (proba[:, 1])
    # Uses CalibratedClassifierCV with method='sigmoid', cv='prefit'

class IsotonicCalibrator(BaseCalibrator):
    # Fits on positive class probability
    # Uses CalibratedClassifierCV with method='isotonic', cv='prefit'
```

**Critical Rule:** Calibrator fit on CALIBRATION split predictions, NOT training or test.

**Tests (tests/test_calibration.py):**
- NoCalibration: identity transform
- Platt: output shape matches input
- Isotonic: output monotonically increasing
- All: fit on calib split, calibrate on test split

---

### 4.9 Metrics Module (`src/calibmiss/metrics.py`)

**Responsibility:** Compute all evaluation metrics.

**Metrics:**

| Metric | Formula | Purpose |
|--------|---------|---------|
| Brier Score | mean((proba_pos - y)²) | Primary calibration (lower=better) |
| ECE (10 bins) | Σ|bin|/n × |acc_bin - conf_bin| | Primary calibration (lower=better) |
| ROC AUC | sklearn.metrics.roc_auc_score | Discrimination (higher=better) |
| Reliability Diagram Data | bin_centers, accuracies, counts, confidences | Visualization |

**Functions:**
```python
compute_brier(proba: np.ndarray, y: np.ndarray) -> float
    # proba shape (n, 2), uses proba[:, 1] (positive class)

compute_ece(proba: np.ndarray, y: np.ndarray, n_bins=10) -> float
    # Equal-width bins on proba[:, 1]
    # Weighted average of |accuracy - confidence|

compute_roc_auc(proba: np.ndarray, y: np.ndarray) -> float
    # sklearn.metrics.roc_auc_score(y, proba[:, 1])

reliability_diagram_data(proba, y, n_bins=10) -> tuple
    # Returns (bin_centers, bin_accuracies, bin_counts, bin_confidences)
```

**Tests (tests/test_metrics.py):**
- Brier: perfect predictions = 0, worst = 1
- ECE: perfect calibration = 0
- Reliability diagram: returns correct shapes
- All: handle edge cases (single class, all same probability)

---

### 4.10 Pipeline Module (`src/calibmiss/pipeline.py`)

**Responsibility:** Orchestrate single experimental configuration run.

**Function:**
```python
def run_single_config(config: dict, fold_idx: int, seed: int) -> dict:
    """
    Executes one (model, imputer, calibrator, mechanism, rate, fold, seed) combination.
    
    Steps:
    1. Load features + target for this fold (from splitter)
    2. Simulate missingness on train/calib/test separately with seed
    3. Fit imputer on train, transform all three
    4. Fit model on imputed train
    5. Get proba on calib set, fit calibrator
    6. Get proba on test set, apply calibrator
    7. Compute metrics on test set
    8. Return dict with all config params + metrics + fold + seed
    """
```

**Return Dict Keys:**
```python
{
    # Config identifiers
    "model": str,
    "imputer": str,
    "calibrator": str,
    "mechanism": str,  # MCAR or MAR
    "rate": float,
    "seed": int,
    "fold": int,
    # Metrics
    "brier": float,
    "ece": float,
    "roc_auc": float,
    # Optional: reliability diagram data
}
```

**CLI Entry (`src/calibmiss/run_experiment.py`):**
- Accepts args: --model, --imputer, --calibrator, --mechanism, --rate, --seed
- Runs single config for debugging
- Prints metrics to stdout

---

### 4.11 Grid Runner (`scripts/run_grid.py`)

**Responsibility:** Execute full experimental grid, parallelize, save results incrementally.

**Grid Size:**
- 2 models × 3 imputers × 3 calibrators × 2 mechanisms × 4 rates × 5 seeds × n_folds
- = 720 × n_folds configurations (typically 3-5 folds = 2160-3600 runs)

**Implementation:**
```python
def main():
    cfg = get_config()
    splitter = WalkForwardSplitter(...)
    folds = list(splitter.split(X, y))
    
    # Generate all combinations
    for model in models:
        for imputer in imputers:
            for calibrator in calibrators:
                for mechanism in mechanisms:
                    for rate in rates:
                        for seed in seeds:
                            for fold_idx, (train_idx, calib_idx, test_idx) in enumerate(folds):
                                # Run in parallel (joblib.Parallel)
                                result = run_single_config(...)
                                append_to_csv(result)
```

**Parallelization:**
- Uses `joblib.Parallel(n_jobs=-1)` over folds × seeds
- Each combination independent
- Incremental CSV write (append mode) for fault tolerance

**Output:** `results/metrics.csv` with one row per configuration

---

### 4.12 Figure Generation (`scripts/generate_figures.py`)

**Responsibility:** Generate all report figures from `results/metrics.csv`.

**Figures (per PRD Section 20.8):**

| Figure | Description |
|--------|-------------|
| `brier_vs_missingness.png` | Line plot: Brier vs missing rate, grouped by imputer, faceted by calibrator, separate for MCAR/MAR |
| `ece_vs_missingness.png` | Line plot: ECE vs missing rate, same grouping |
| `reliability_0pct.png` | Reliability diagram at 0% missingness (averaged across seeds/folds) |
| `reliability_30pct.png` | Reliability diagram at 30% missingness |
| `heatmap_calibration.png` | Heatmap: mean ECE (rows=imputer, cols=calibrator) for each model×mechanism×rate |

**Implementation:**
- Loads metrics.csv
- Aggregates across seeds/folds (mean ± std)
- Uses matplotlib/seaborn with consistent styling
- Saves to `results/figures/`

---

### 4.13 Report Templates (`reports/report.md`, `reports/summary.md`)

**Report Structure (`reports/report.md`):**
```markdown
# Calibrated Return Forecasts Under Missing Data

## Abstract
## 1. Introduction
## 2. Data
## 3. Methods
   3.1 Features & Target
   3.2 Missingness Simulation
   3.3 Imputation Strategies
   3.4 Models
   3.5 Calibration
   3.6 Evaluation
   3.7 Experimental Design
## 4. Results
   4.1 Calibration Degradation (MCAR vs MAR)
   4.2 Imputation Effects
   4.3 Calibration Recovery
   4.4 Discrimination Trade-offs
## 5. Discussion
## 6. Limitations
## 7. Conclusion
## References
```

**Summary (`reports/summary.md`):**
- 1-page executive summary
- Key findings in plain language
- Main recommendation table

---

## 5. Data Flow Summary

```
┌─────────────┐
│  config/    │
│  *.yaml     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  download   │────▶│  data/raw/   │────▶│  validate      │
│  _data.py   │     │  ohlcv_*.pq  │     │  Validation    │
└─────────────┘     └──────────────┘     └────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│  compute_   │────▶│  data/processed/ │
│  features.py│     │  features_*.pq   │
└─────────────┘     └──────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    run_grid.py (parallel)                    │
│  For each config:                                           │
│  1. Split (walk-forward + purge)                            │
│  2. Missingness (MCAR/MAR)                                  │
│  3. Impute (fit train → transform all)                      │
│  4. Model fit (train) → predict_proba (calib, test)         │
│  5. Calibrator fit (calib) → calibrate (test)               │
│  6. Metrics (test) → append to metrics.csv                  │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│  generate_  │────▶│  results/figures/│
│  figures.py │     │  *.png           │
└─────────────┘     └──────────────────┘
       │
       ▼
┌──────────────────┐
│  reports/*.md    │
│  (filled with    │
│   results)       │
└──────────────────┘
```

---

## 6. Testing Strategy

**Test Categories:**

| Test File | Component | Key Properties Tested |
|-----------|-----------|----------------------|
| `test_config.py` | Config loading | All fields loaded, types correct, caching works |
| `test_data.py` | Data download | Shape, columns, validation catches errors |
| `test_features_lookahead.py` | Features | NO lookahead bias (critical) |
| `test_missingness.py` | Missingness | Rates correct, reproducibility, MAR dependence |
| `test_splits.py` | Splits | Purge gap, no overlap, chronological order |
| `test_imputers.py` | Imputers | Fit on train only, no leakage |
| `test_calibration.py` | Calibration | Fit on calib only, identity/monotonicity |
| `test_metrics.py` | Metrics | Known-value correctness |
| `test_models.py` | Models | Interface, probability outputs |
| `test_pipeline.py` | Pipeline | Single config runs end-to-end |
| `test_figures.py` | Figures | All 5 figures generated |
| `test_report.py` | Reports | Files exist |
| `test_e2e.py` | Integration | Minimal pipeline runs |

**Quality Gates (run before commit):**
1. `pytest -v` - All tests pass
2. `ruff check src tests` - Zero lint errors
3. `mypy src/calibmiss` - Zero type errors
4. `make all` - Full pipeline completes (on clean clone)

---

## 7. Configuration Reference

### 7.1 `config/universe.yaml`
```yaml
universe:
  tickers: [30 large-cap US tickers]
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  source: "yfinance"
```

### 7.2 `config/experiment.yaml`
```yaml
experiment:
  models: [logistic_regression, lightgbm]  # with params
  imputers: [forward_fill, median, iterative]  # with params
  calibrators: [none, platt, isotonic]  # with params
  missingness:
    mechanisms: ["MCAR", "MAR"]
    rates: [0.0, 0.1, 0.2, 0.3]
    seeds: [42, 123, 456, 789, 999]
  splits:
    train_years: 3
    calibrate_months: 6
    test_months: 6
    roll_forward_months: 6
    purge_days: 5
  evaluation:
    n_bins_ece: 10
    metrics: ["brier", "ece", "roc_auc"]
```

### 7.3 `config/missingness.yaml`
```yaml
mar_rules:
  - target_feature: "volatility_20d"
    depends_on: "volume_ratio"
    rule_type: "quantile_threshold"
    params:
      base_prob: 0.05
      max_prob: 0.5
      quantile_threshold: 0.7
```

---

## 8. Expected Outputs

### 8.1 `results/metrics.csv`
Columns:
```
model, imputer, calibrator, mechanism, rate, seed, fold,
brier, ece, roc_auc,
[optional: reliability_bin_centers, reliability_accuracies, reliability_counts, reliability_confidences]
```

### 8.2 `results/figures/`
- `brier_vs_missingness.png`
- `ece_vs_missingness.png`
- `reliability_0pct.png`
- `reliability_30pct.png`
- `heatmap_calibration.png`

### 8.3 `reports/`
- `report.md` - Full 6-10 page report
- `summary.md` - 1-page executive summary
- `report.pdf` - Compiled via pandoc (optional)

---

## 9. Key Invariants (Must Never Break)

1. **NO LOOKAHEAD** - Features at time t use only data ≤ t
2. **FIT ON TRAIN ONLY** - Imputers, scalers, calibrators never see test data during fit
3. **WALK-FORWARD ONLY** - No random splits; time order strictly preserved
4. **PURGE GAP ≥5 DAYS** - Between train/calib and calib/test
5. **SEEDS FIXED** - All randomness controlled via config seeds
6. **RAW DATA IMMUTABLE** - Never modified in place; cached with checksums
7. **REPRODUCIBLE** - Same config + seed = identical results

---

## 10. Implementation Order (Dependencies)

```
Task 1: Config System (COMPLETED) ✓
    ↓
Task 2: Data Download & Validation
    ↓
Task 3: Features + Lookahead Test
    ↓
Task 4: Walk-Forward Splits
    ↓
Task 5: Missingness Simulation
    ↓
Task 6: Imputers
    ↓
Task 7: Models
    ↓
Task 8: Calibration
    ↓
Task 9: Metrics
    ↓
Task 10: Pipeline Orchestration
    ↓
Task 11: Figure Generation
    ↓
Task 12: Report Templates
    ↓
Task 13: E2E Integration Test
    ↓
Task 14: Full Run & Polish
```

Each task depends on the previous task's interfaces. This documentation defines those interfaces precisely.

---

## 11. Stretch Goals (Post-MVP)

- [ ] MNAR simulation with configurable rules
- [ ] Interactive dashboard (Streamlit) for result exploration
- [ ] Crypto universe as alternative dataset
- [ ] Additional models: MLP, TabPFN
- [ ] Additional calibration: temperature scaling, Beta calibration
- [ ] Feature importance stability analysis across missingness levels
- [ ] Bayesian calibration methods

---

*This documentation serves as the single source of truth for all component specifications. All implementation tasks should reference this document for interfaces, invariants, and expected behavior.*