# CalibMiss: Calibrated Return Forecasts Under Missing Data

CalibMiss is an empirical research pipeline evaluating whether probabilistic next-day stock-direction forecasts remain calibrated when feature inputs suffer from missingness. It compares Missing Completely at Random (MCAR) and Missing at Random (MAR) mechanisms across missingness rates, imputers, model families, and post-hoc calibrators using chronological walk-forward splits.

> **Scope notice:** CalibMiss is a calibration research study, not a trading strategy. Profitability analysis, transaction costs, live execution, and position sizing are explicitly outside project scope.

---

## Research Questions

1. How does calibration error (Brier score, Expected Calibration Error) degrade as missingness increases from 0% to 30% under MCAR and MAR?
2. Does calibration degradation differ systematically between MCAR and MAR mechanisms?
3. Can imputation combined with Platt scaling or isotonic regression recover calibration, and what discrimination cost (ROC AUC) does that recovery impose?

---

## Pipeline Architecture

The codebase is organized as a modular Python package under `src/calibmiss/`:

```
src/calibmiss/
├── config.py           # Pydantic schemas and YAML configuration loaders
├── data.py             # yfinance download, Parquet caching with SHA256 checksum, OHLCV validation
├── features.py         # Point-in-time features with lookahead safety (returns, volatility, volume ratio)
├── splits.py           # Walk-forward date-axis splits with chronological purge gaps
├── missingness.py      # MCAR and MAR missingness injection with realized rate recording
├── imputers.py         # Causal forward-fill, training-median, and MICE (fit on train only)
├── models.py           # Standardized Logistic Regression and LightGBM probability classifiers
├── calibration.py      # Identity, Platt scaling, and Isotonic regression (fit on cal split only)
├── metrics.py          # Brier score, 10-bin ECE, ROC AUC, and reliability diagram series
├── pipeline.py         # Single (model, imputer, calibrator, mechanism, rate, seed) fold runner
└── run_experiment.py   # Full grid orchestrator, baselines, paired Wilcoxon tests, Holm correction
```

Supporting directories:
- `config/`: YAML specifications (`universe.yaml`, `features.yaml`, `missingness.yaml`, `experiment.yaml`).
- `scripts/`: CLI entry points for pipeline stages (`download_data.py`, `compute_features.py`, `run_grid.py`, `generate_figures.py`).
- `notebooks/`: Diagnostic data exploration (`exploration.ipynb`).
- `reports/`: Complete written report (`report.md`) and non-specialist executive summary (`summary.md`).
- `tests/`: Unit, integration, lookahead safety, and contract test suite.

---

## Setup and Installation

### Prerequisites

- Python 3.11+
- Virtual environment tool (`venv`)
- Standard build tools (`make`)

### Installation

```bash
# Clone the repository
git clone https://github.com/Khairil-12/Calibrated-return-forecasts-under-missing-data.git
cd Calibrated-return-forecasts-under-missing-data

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Pipeline

The full research pipeline is executable through `make` targets:

```bash
# One-command execution: data -> features -> experiment -> figures -> report
make all
```

### Step-by-Step Execution

1. **Download and cache raw OHLCV data:**
   ```bash
   make data
   # Or: python scripts/download_data.py
   ```
   Downloads 32 large-cap U.S. equities (2010-01-01 to 2024-12-31) from Yahoo Finance, validates OHLCV invariants, and saves to `data/raw/{timestamp}_sha256_{hash}.parquet`.

2. **Compute point-in-time features and target:**
   ```bash
   make features
   # Or: python scripts/compute_features.py
   ```
   Generates 8 lag-only features and next-day direction target into `data/processed/features.parquet` and `target.parquet`.

3. **Run experimental grid:**
   ```bash
   make experiment
   # Or: python scripts/run_grid.py
   ```
   Executes the configured grid across walk-forward folds, writing `results/metrics.csv`, `summary.csv`, `statistical_tests.csv`, and `run_metadata.json`.

4. **Generate figures:**
   ```bash
   make figures
   # Or: python scripts/generate_figures.py
   ```
   Generates line plots (Brier and ECE vs. rate), reliability diagrams at 0% and 30%, and a calibration heatmap in `results/figures/`.

5. **Single configuration test (debugging):**
   ```bash
   make run-single
   # Or: python -m src.calibmiss.run_experiment --model logistic_regression --imputer median --calibrator platt --mechanism MCAR --rate 0.1 --seed 42
   ```

6. **Clean generated artifacts:**
   ```bash
   make clean      # Cleans processed features, metrics, and figures
   make clean-all  # Cleans all generated artifacts including cached raw data
   ```

---

## Experimental Design

| Dimension | Levels |
|---|---|
| **Models** | Logistic Regression (standardized), LightGBM |
| **Imputers** | Forward fill (within-ticker causal), Median (train-only), Iterative/MICE |
| **Calibrators** | None (identity), Platt scaling, Isotonic regression |
| **Mechanisms** | MCAR (independent feature masking), MAR (conditional on volume ratio) |
| **Missing Rates** | 0%, 10%, 20%, 30% |
| **Random Seeds** | 5 seeds per configuration (42, 43, 44, 45, 46) |
| **Splits** | Chronological walk-forward (3-year train, 6-month calibrate, 6-month test, 5-day purge gap) |
| **Baselines** | Complete-data oracle (0% missingness), Naive training-fold base rate |

### Evaluation Metrics

- **Primary:** Brier score (squared probability error), Expected Calibration Error (ECE, 10 equal-width bins), Reliability diagrams.
- **Secondary:** ROC AUC (discrimination).
- **Statistical testing:** Paired Wilcoxon signed-rank tests across walk-forward folds with Holm-Bonferroni correction.

---

## Testing and Code Quality

Run tests, coverage, and linting:

```bash
# Run full test suite
python -m pytest -v

# Run with coverage report
python -m pytest -v --cov=src/calibmiss --cov-report=term-missing

# Lint checks
ruff check .

# Format check
ruff format --check .
```

The test suite enforces:
- Lookahead safety (future raw data mutations cannot alter day $t$ features).
- Leakage prevention (imputers fit on train only; calibrators fit on cal split only).
- Chronological purge enforcement (no overlapping rolling windows).
- Checksum integrity on cached Parquet files.

---

## Limitations and Scope

- Daily equity data for 32 large-cap U.S. stocks; findings may not generalize to intraday frequencies or alternative asset classes.
- MCAR and MAR are synthetic simulations designed to test robustness, not full descriptions of live market feed outages.
- MAR models depend on a single configured trigger variable (`volume_ratio`); sensitivity to other dependencies requires further research.
- Calibration results do not imply economic profitability or alpha generation.
