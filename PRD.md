# PRD: Calibrated Return Forecasts Under Missing Data

**Project codename:** CalibMiss
**Author:** Kairill
**Status:** Draft v1.0
**Last updated:** 2026-11-08
**Estimated duration:** 6 weeks (part-time, ~10 hours/week)

---

## 1. Overview

This project studies whether a probabilistic classifier that predicts the direction of next-day stock returns remains well calibrated when its input data has missing values, and whether common imputation and calibration techniques can restore that calibration.

In quantitative trading, a model that outputs "70% probability the stock goes up" is only useful if the 70% holds up in practice. Real market data is incomplete: prices are missing for halted stocks, volume is missing on some days, and fundamental fields arrive late or with gaps. Most student and beginner projects evaluate models only on accuracy, which hides calibration problems. This project makes calibration under missingness the central question.

The deliverable is a reproducible research repository with code, experiments, figures, and a written report that a hiring manager at a quantitative firm can read in about 15 minutes and rerun in under an hour.

---

## 2. Problem Statement

Models trained on complete data are often deployed on incomplete data. When missingness is not random, the distribution of inputs shifts, and the model's output probabilities can become systematically overconfident or underconfident. Without measuring this, a trader may size positions based on probabilities that are wrong.

The project answers three questions:

1. How much does calibration degrade as the fraction of missing values increases, under different missingness mechanisms?
2. Does the answer change between MCAR (missing completely at random) and MAR (missing at random, where missingness depends on observed variables)?
3. Do imputation methods combined with post-hoc calibration (Platt scaling, isotonic regression) recover calibration, and at what cost to discrimination (AUC)?

---

## 3. Goals

- **G1.** Quantify calibration degradation for at least two model families across at least four missingness rates (0%, 10%, 20%, 30%) and two mechanisms (MCAR, MAR).
- **G2.** Compare at least three imputation strategies and at least two calibration methods on the same data splits.
- **G3.** Produce a clear recommendation: which combination of imputation and calibration is most reliable under each mechanism, with uncertainty estimates.
- **G4.** Deliver a reproducible pipeline where one command regenerates every table and figure in the report.
- **G5.** Write results in a form suitable for a portfolio, a technical blog post, or a research summary.

## 4. Non-Goals

- Building a profitable trading strategy. Profit is not evaluated and not the point of this project.
- Live trading, broker integration, or real-time data feeds.
- Modeling missing-not-at-random (MNAR) mechanisms in full. MNAR is listed as an optional stretch goal only.
- High-frequency data. The project uses daily data only.
- Novel algorithm invention. The project applies and compares existing methods carefully.

---

## 5. Target Audience

- **Primary:** Technical reviewers and recruiters at quantitative firms, who will evaluate rigor, code quality, and clarity of reasoning.
- **Secondary:** Me, as a reference for future research and interview discussion.
- **Tertiary:** Students or practitioners who want a template for evaluating calibration on financial data.

---

## 6. Data Requirements

### 6.1 Source

- Daily adjusted OHLCV (open, high, low, close, adjusted close, volume) from Yahoo Finance via the `yfinance` library.
- Universe: 30 large-cap U.S. stocks with long, continuous histories (for example, a fixed list of constituents of a major index that have traded since at least 2010).
- Date range: 2010-01-01 to 2024-12-31, or the longest continuous range available at project start. The exact list of tickers and date range must be frozen and saved to `config/universe.yaml` before any experiment runs.

### 6.2 Data Storage

- Raw downloads saved to `data/raw/` with a timestamp and a checksum.
- Processed data saved to `data/processed/` in Parquet format.
- Raw data is never modified in place.

### 6.3 Data Quality Checks

- Confirm no duplicate (ticker, date) rows.
- Confirm prices are positive and volume is non-negative.
- Record the native missing-value rate per column. Real missingness in the source data is reported but not used as the experimental variable, so that experiments are controlled.

---

## 7. Feature and Target Definition

### 7.1 Target

- `y_t = 1` if the adjusted close on day t+1 is higher than on day t, otherwise `y_t = 0`.
- Days with zero return are dropped from the target to avoid ambiguity.

### 7.2 Features (computed using only information available at time t)

- 1-day, 5-day, and 20-day log returns.
- 5-day and 20-day rolling volatility of returns.
- Volume ratio: today's volume divided by the 20-day average volume.
- High-low range divided by close.
- Price relative to its 50-day moving average.

**Lookahead rule:** Every feature must be computed using only data up to and including day t. A unit test must verify this by shifting future data and confirming feature values do not change.

---

## 8. Missingness Simulation

Missingness is injected into the feature matrix **after** features are computed, on the test set and training set separately as specified in Section 10.

### 8.1 MCAR (Missing Completely At Random)

- Each feature value is independently set to missing with probability `p`, where `p` is in {0.0, 0.1, 0.2, 0.3}.
- Missingness does not depend on any variable.

### 8.2 MAR (Missing At Random)

- The probability that a feature is missing depends on an observed variable, not on its own unobserved value.
- Example rule: the probability that `volatility_20d` is missing increases with `volume_ratio`. The rule is defined in a config file, and the average missing rate is calibrated to match the target `p`.
- The dependence structure is fixed and documented so that results are reproducible.

### 8.3 Verification

- For each simulated dataset, record the realized missing rate per column.
- Plot realized vs. target missing rate to confirm the simulation is correct.

### 8.4 Stretch Goal: MNAR

- Missingness depends on the unobserved value itself (for example, the largest drawdown days are more likely to be missing). Only pursued if time allows. Clearly labeled as exploratory.

---

## 9. Methods

### 9.1 Imputation Strategies

| Method | Description | Notes |
|---|---|---|
| Forward fill (within ticker) | Carry the last observed value forward | Baseline; simple but can leak if not careful |
| Median imputation | Replace with the training-set median | Baseline |
| Iterative imputation (MICE-style) | Model each feature as a function of others, iterate | Uses scikit-learn `IterativeImputer` |

All imputers must be **fit on the training set only** and then applied to the test set.

### 9.2 Models

- **Logistic regression** (with standardized features) as the interpretable baseline.
- **Gradient-boosted trees** (LightGBM or XGBoost) as the more flexible model.

Both models are trained on complete training data in the baseline condition and on imputed training data in the missingness conditions (see Section 10 for the exact protocol).

### 9.3 Calibration Methods

- **Uncalibrated:** raw model probabilities.
- **Platt scaling:** logistic regression fit on held-out predictions.
- **Isotonic regression:** non-parametric monotone mapping fit on held-out predictions.

Calibration models must be fit on a **separate calibration split** that is not used to train the base model or to evaluate the final metrics.

---

## 10. Experimental Design

### 10.1 Data Splitting

Use walk-forward (time-ordered) splits. No random shuffling.

- Split the timeline into rolling windows: for example, train on 3 years, calibrate on the next 6 months, test on the following 6 months, then roll forward by 6 months.
- Insert a **purge gap** of at least 5 trading days between the training and test windows to avoid overlap from rolling-window features.

### 10.2 Experimental Grid

For each combination of:

- Model: 2 (logistic regression, gradient boosting)
- Imputation: 3 (forward fill, median, iterative)
- Calibration: 3 (none, Platt, isotonic)
- Missingness mechanism: 2 (MCAR, MAR)
- Missingness rate: 4 (0%, 10%, 20%, 30%)

This produces 144 configurations per walk-forward fold. Each configuration is repeated with **at least 5 random seeds** for the missingness simulation to estimate variance.

### 10.3 Evaluation Metrics

**Calibration (primary):**
- Brier score (lower is better)
- Expected Calibration Error (ECE), with 10 equal-width bins
- Reliability diagram (visual)

**Discrimination (secondary):**
- ROC AUC

**Stability:**
- Standard deviation of each metric across walk-forward folds and seeds

**Statistical testing:**
- Use paired comparisons across folds (for example, a Wilcoxon signed-rank test) when comparing methods. Report p-values and effect sizes, and correct for multiple comparisons with Holm or Benjamini-Hochberg.

### 10.4 Baseline Reference

Two reference points are required in every results table:

- **Complete-data oracle:** the same model and calibration trained and tested on 0% missingness.
- **Naive prediction:** always predict the training-set base rate of "up" days.

---

## 11. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Download and cache OHLCV data for the frozen universe | Must |
| FR-2 | Compute all features with a lookahead-safety test | Must |
| FR-3 | Simulate MCAR and MAR missingness with a fixed seed and config | Must |
| FR-4 | Run the full experimental grid from a single config file | Must |
| FR-5 | Fit imputers and calibrators on training/calibration splits only | Must |
| FR-6 | Output a results table (CSV) with all metrics per configuration | Must |
| FR-7 | Generate all report figures from the results table with one command | Must |
| FR-8 | Support MNAR simulation | Could (stretch) |
| FR-9 | Provide an interactive dashboard to explore results | Could (stretch) |

---

## 12. Non-Functional Requirements

- **Reproducibility:** A fresh clone plus `make all` regenerates all results and figures. Seeds, library versions, and data download date are recorded.
- **Environment:** Python 3.11 or later. All dependencies pinned in `requirements.txt` or `pyproject.toml`.
- **Runtime:** The full experimental grid completes in under 2 hours on a standard laptop. Use caching and parallelism where needed.
- **Code quality:** Type hints on public functions, docstrings, `ruff` for linting, `pytest` for tests, and at least 70% test coverage on the feature and missingness modules.
- **Documentation:** README explains how to reproduce results in under 5 minutes of reading.

---

## 13. Proposed Repository Structure

```
calibmiss/
├── README.md
├── PRD.md
├── Makefile
├── pyproject.toml
├── config/
│   ├── universe.yaml
│   ├── experiment.yaml
│   └── missingness.yaml
├── data/
│   ├── raw/            (gitignored)
│   └── processed/      (gitignored)
├── src/calibmiss/
│   ├── data.py
│   ├── features.py
│   ├── missingness.py
│   ├── imputers.py
│   ├── calibration.py
│   ├── splits.py
│   ├── metrics.py
│   └── run_experiment.py
├── tests/
│   ├── test_features_lookahead.py
│   ├── test_missingness.py
│   └── test_splits.py
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── metrics.csv
│   └── figures/
└── report/
    └── report.md
```

---

## 14. Deliverables

1. **Code repository** on GitHub with the structure above.
2. **Results table** (`results/metrics.csv`) with all configurations.
3. **Figures:**
   - Brier score and ECE vs. missingness rate, by imputer and calibrator, for each mechanism.
   - Reliability diagrams at 0% and 30% missingness.
   - Heatmap of calibration error across the experimental grid.
4. **Written report** (6 to 10 pages) covering motivation, methods, results, limitations, and recommendations.
5. **Short summary** (1 page or a blog post) for non-specialist readers.

---

## 15. Milestones and Timeline

| Week | Milestone | Exit criteria |
|---|---|---|
| 1 | Setup and data | Repo created, universe frozen, data downloaded and validated |
| 2 | Features and splits | Feature code complete, lookahead test passing, walk-forward splits implemented |
| 3 | Missingness and imputers | MCAR and MAR simulation verified; three imputers implemented and tested |
| 4 | Models and calibration | Two models and two calibrators running on the baseline (0% missingness) |
| 5 | Full experiment | Full grid run, results table produced, statistical tests done |
| 6 | Report and polish | Figures finalized, report written, README complete, repo cleaned |

---

## 16. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Lookahead bias inflates results | Medium | High | Lookahead unit test; strict fit-on-train-only rule; code review of all pipeline steps |
| Results are noise (small effect sizes) | High | Medium | Multiple seeds, multiple folds, statistical tests; report uncertainty honestly |
| MAR simulation is too artificial | Medium | Medium | Document the dependence rule clearly; include a sensitivity check with a second rule |
| Compute time exceeds budget | Low | Medium | Limit grid if needed; cache imputed datasets; parallelize over folds |
| Data source changes or yfinance breaks | Medium | Low | Cache raw data with checksums; pin library version |
| Scope creep into trading strategy | Medium | Medium | Respect non-goals; any profitability analysis is out of scope |

---

## 17. Success Criteria

The project is considered successful if:

- [ ] The full pipeline runs end-to-end from a clean clone with a single command.
- [ ] Lookahead and split-leakage tests pass.
- [ ] Calibration results are reported for all grid configurations with uncertainty estimates.
- [ ] The report gives a clear, evidence-based answer to the three research questions in Section 2, including cases where the answer is "no effect" or "inconclusive."
- [ ] Limitations are stated explicitly (daily data, 30 stocks, simulated missingness, linear and tree models only).
- [ ] A non-specialist can read the summary and explain the main finding.

---

## 18. Open Questions

1. Which exact 30 tickers form the universe? (Decide in Week 1 and freeze.)
2. Should the MAR dependence rule be a single rule or a family of rules for sensitivity analysis?
3. Is LightGBM or XGBoost preferred for the tree model? (Default: LightGBM.)
4. Will the project be published publicly from the start, or after the report is finished?

---

## 19. Appendix: Glossary

- **MCAR:** Missing completely at random. Missingness is unrelated to any data.
- **MAR:** Missing at random. Missingness depends only on observed data.
- **MNAR:** Missing not at random. Missingness depends on the missing value itself.
- **Calibration:** Agreement between predicted probabilities and observed frequencies.
- **Brier score:** Mean squared error between predicted probabilities and binary outcomes.
- **ECE:** Expected calibration error, the weighted average gap between predicted confidence and observed accuracy across bins.
- **Walk-forward validation:** Evaluation where the model is always trained on the past and tested on the future.
- **Purge gap:** A buffer period between training and test data to prevent information overlap.

---

## 20. Implementation Setup Guide

### 20.1 Tech Stack

| Component | Choice | Version Pinning |
|-----------|--------|-----------------|
| Language | Python | 3.11+ (pin in pyproject.toml) |
| Data download | yfinance | Latest compatible |
| Data processing | pandas, numpy | Pin exact versions |
| ML models | scikit-learn, lightgbm | Pin exact versions |
| Imputation | scikit-learn (IterativeImputer) | Built-in |
| Calibration | scikit-learn (CalibratedClassifierCV) | Built-in |
| Experiment tracking | MLflow or simple CSV/JSON | MLflow optional |
| Plotting | matplotlib, seaborn | Pin exact versions |
| CLI / orchestration | Make + Python scripts | Native |
| Testing | pytest, pytest-cov | Pin exact versions |
| Linting | ruff | Pin exact version |
| Type checking | mypy | Pin exact version |

### 20.2 Environment Setup

```bash
# 1. Clone and enter
git clone <repo-url> calibmiss
cd calibmiss

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 20.3 Configuration Files to Create

Create these files before running experiments:

**config/universe.yaml** — Frozen list of 30 tickers + date range
```yaml
universe:
  tickers:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - NVDA
    - META
    - TSLA
    - BRK.B
    - JPM
    - JNJ
    - V
    - WMT
    - MA
    - PG
    - UNH
    - HD
    - CVX
    - MRK
    - ABBV
    - PEP
    - KO
    - COST
    - AVGO
    - TMO
    - ACN
    - DIS
    - ADBE
    - NKE
    - CRM
    - LIN
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  source: "yfinance"
```

**config/experiment.yaml** — Experimental grid parameters
```yaml
experiment:
  models:
    - name: "logistic_regression"
      params:
        C: 1.0
        max_iter: 1000
        class_weight: "balanced"
    - name: "lightgbm"
      params:
        n_estimators: 200
        max_depth: 5
        learning_rate: 0.05
        num_leaves: 31
        class_weight: "balanced"
        verbosity: -1

  imputers:
    - name: "forward_fill"
      params: {}
    - name: "median"
      params: {}
    - name: "iterative"
      params:
        max_iter: 10
        random_state: 42

  calibrators:
    - name: "none"
      params: {}
    - name: "platt"
      params:
        method: "sigmoid"
    - name: "isotonic"
      params:
        method: "isotonic"

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

**config/missingness.yaml** — MAR dependence rules
```yaml
mar_rules:
  - target_feature: "volatility_20d"
    depends_on: "volume_ratio"
    rule_type: "quantile_threshold"
    params:
      # Missing prob increases with volume_ratio quantile
      base_prob: 0.05
      max_prob: 0.5
      quantile_threshold: 0.7
```

### 20.4 Directory Structure (Complete)

```
calibmiss/
├── README.md
├── PRD.md
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
├── config/
│   ├── universe.yaml
│   ├── experiment.yaml
│   └── missingness.yaml
├── data/
│   ├── raw/              # gitignored - timestamped parquet files
│   └── processed/        # gitignored - feature matrices, splits
├── src/calibmiss/
│   ├── __init__.py
│   ├── config.py         # Config loading with pydantic
│   ├── data.py           # Download, cache, validate OHLCV
│   ├── features.py       # Feature engineering + lookahead test
│   ├── missingness.py    # MCAR/MAR simulation
│   ├── imputers.py       # Imputation strategies (fit on train only)
│   ├── calibration.py    # Platt / isotonic calibration
│   ├── splits.py         # Walk-forward splits with purge gap
│   ├── metrics.py        # Brier, ECE, reliability diagrams
│   ├── models.py         # Model wrappers (LR, LightGBM)
│   ├── pipeline.py       # Orchestration: run one config
│   └── run_experiment.py # Main entry: loops over grid
├── tests/
│   ├── __init__.py
│   ├── test_features_lookahead.py
│   ├── test_missingness.py
│   ├── test_splits.py
│   ├── test_imputers.py
│   └── test_calibration.py
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── metrics.csv       # All experiment results
│   └── figures/          # Auto-generated plots
├── reports/
│   └── report.md         # Final written report
└── scripts/
    ├── download_data.py
    ├── compute_features.py
    ├── run_grid.py
    └── generate_figures.py
```

### 20.5 Makefile Commands

```makefile
.PHONY: all setup data features experiment figures report clean

# Full pipeline
all: setup data features experiment figures report

# Environment
setup:
	pip install -r requirements.txt
	pre-commit install

# Data
data:
	python scripts/download_data.py

features:
	python scripts/compute_features.py

# Experiment
experiment:
	python scripts/run_grid.py

# Figures from results/metrics.csv
figures:
	python scripts/generate_figures.py

# Report compilation
report:
	pandoc reports/report.md -o reports/report.pdf --pdf-engine=weasyprint

# Clean generated files (keep raw data)
clean:
	rm -rf data/processed results/metrics.csv results/figures/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Clean everything including raw data
clean-all: clean
	rm -rf data/raw/*
```

### 20.6 Running the Pipeline

```bash
# One-command full run (after setup)
make all

# Or step by step:
make setup
make data
make features
make experiment
make figures
make report

# Run a single configuration for debugging
python -m src.calibmiss.run_experiment \
  --model logistic_regression \
  --imputer median \
  --calibrator platt \
  --mechanism MCAR \
  --rate 0.1 \
  --seed 42
```

### 20.7 Key Implementation Rules (Enforced by Tests)

1. **No lookahead**: `features.py` functions must pass `test_features_lookahead.py`
2. **Fit on train only**: Imputers, scalers, calibrators fit ONLY on training split
3. **Walk-forward only**: No random shuffling in `splits.py`
4. **Purge gap**: Minimum 5 trading days between train and test
5. **Seeds fixed**: All randomness controlled via config seeds
6. **Checksums**: Raw data files saved with SHA256 in filename

### 20.8 Expected Outputs

| File | Description |
|------|-------------|
| `results/metrics.csv` | One row per (fold, seed, model, imputer, calibrator, mechanism, rate) with all metrics |
| `results/figures/brier_vs_missingness.png` | Line plot: Brier score vs missing rate by imputer/calibrator/mechanism |
| `results/figures/ece_vs_missingness.png` | Line plot: ECE vs missing rate |
| `results/figures/reliability_0pct.png` | Reliability diagram at 0% missingness |
| `results/figures/reliability_30pct.png` | Reliability diagram at 30% missingness |
| `results/figures/heatmap_calibration.png` | Heatmap: mean ECE across grid |
| `reports/report.md` | 6-10 page written report |
| `reports/summary.md` | 1-page executive summary |

### 20.9 Stretch Goals (If Time Permits)

- [ ] MNAR simulation with configurable rules
- [ ] Interactive dashboard (Streamlit) to explore results
- [ ] Crypto universe (BTC, ETH, SOL, etc.) as alternative dataset
- [ ] Additional models: MLP, TabPFN
- [ ] Additional calibration: temperature scaling, Beta calibration
- [ ] Feature importance stability analysis across missingness
