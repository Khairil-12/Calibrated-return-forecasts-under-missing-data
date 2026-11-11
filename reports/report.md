# CalibMiss: Calibrated Return Forecasts Under Missing Data

**Status:** Methods and implementation draft; quantitative results pending the configured data download and experiment run.

## Abstract

CalibMiss studies whether probabilistic next-day stock-direction forecasts remain calibrated when feature values are missing. It compares missing-completely-at-random (MCAR) and missing-at-random (MAR) mechanisms across missingness rates, model families, imputers, and post-hoc calibrators. The primary outcomes are Brier score, expected calibration error (ECE), and reliability diagrams; ROC AUC is secondary. The repository contains a deterministic, cached data pipeline, feature construction, missingness simulation, train-only imputation, calibration on a held-out split, walk-forward evaluation, statistical summaries, and figure generation.

Results are pending: `results/metrics.csv` and generated figures are not present in this checkout. No empirical claim is made here until `make data`, `make features`, `make experiment`, and `make figures` have completed successfully.

## Introduction

A probability forecast is useful only when its stated probabilities match observed frequencies. Missing inputs can shift model predictions and make them overconfident or underconfident. This project asks:

1. How does calibration change as missingness increases under MCAR and MAR?
2. Does calibration degradation differ between MCAR and MAR?
3. Can imputation plus Platt or isotonic calibration recover calibration, and what discrimination cost does that impose?

This is a calibration study, not a trading strategy. Profitability, live execution, and position sizing are outside scope.

## Data

Daily OHLCV data is sourced from Yahoo Finance through `yfinance`. The frozen universe and date range are recorded in `config/universe.yaml`: 32 configured U.S. large-cap tickers, from 2010-01-01 through 2024-12-31. Raw files are cached as Parquet with a timestamp and truncated SHA256 checksum in the filename.

The data validator checks duplicate `(ticker, date)` rows, positive prices, non-negative volume, and native missingness by column. Native source missingness is reported separately from controlled missingness injected after feature construction.

The target is one when next-day adjusted close exceeds current adjusted close. Zero-return days and each ticker's final day are excluded. Features are computed using information available through day *t*: 1-, 5-, and 20-day log returns; 5- and 20-day annualized rolling volatility; volume ratio; high-low range; and price relative to its 50-day moving average.

## Methods

### Missingness

MCAR masks each feature value independently at the configured rate. MAR masks `volatility_20d` using the configured dependence on `volume_ratio`; the probability is calibrated to the requested average rate. Each mechanism receives fixed seeds, and realized rates are recorded.

### Imputation

The grid compares within-ticker causal forward fill, training-set median imputation, and scikit-learn iterative imputation. Every imputer is fit on training rows only, then applied to calibration and test rows.

### Models and calibration

The base models are standardized logistic regression and LightGBM. Calibration options are identity, Platt scaling, and isotonic regression. A calibrator is fit only on held-out calibration predictions; final metrics use a separate test window.

### Walk-forward evaluation

Unique trading dates determine chronological windows: three years of training, six months of calibration, and six months of testing, rolled forward by six months. Configured purge gaps separate windows. Date-axis splits are mapped to all ticker rows for each date, so rows from the same date remain in the same split.

### Metrics and statistical analysis

Brier score measures squared probability error. ECE uses ten equal-width probability bins. Reliability data stores bin counts, mean confidence, and observed frequency. ROC AUC measures discrimination and is secondary; one-class test folds report undefined AUC as `NaN` rather than failing.

The experiment runner emits complete-data oracle and naive training-fold base-rate baselines. Summary tables report means, standard deviations, paired fold comparisons, Wilcoxon p-values, signed-rank effects, pair counts, and Holm-adjusted p-values. Comparisons are paired by fold and configuration; ticker rows are not treated as independent replicates.

## Experimental Design

The configured grid contains:

| Dimension | Levels |
|---|---:|
| Models | 2 |
| Imputers | 3 |
| Calibrators | 3 |
| Mechanisms | 2 |
| Rates | 4: 0%, 10%, 20%, 30% |
| Seeds | 5 |

This gives 144 configurations per walk-forward fold and 720 configuration-seed runs per fold, before baseline rows. Exact fold counts depend on the available trading-date range.

Primary comparisons should use identical fold, seed, model, mechanism, and rate keys. The complete-data oracle is a reference condition, not a claim that missingness can be removed in deployment. The naive baseline is a prevalence reference, not a classifier.

## Results

### Availability

Results are pending. The expected inputs are:

- `results/metrics.csv`
- `results/summary.csv`
- `results/statistical_tests.csv`
- `results/run_metadata.json`
- `results/figures/brier_vs_missingness.png`
- `results/figures/ece_vs_missingness.png`
- `results/figures/reliability_0pct.png`
- `results/figures/reliability_30pct.png`
- `results/figures/heatmap_calibration.png`

After generation, this section must report observed Brier and ECE trends by mechanism and rate, compare imputers and calibrators with uncertainty, state ROC AUC changes, and cite paired tests with corrected p-values. Placeholder text must be replaced by values read from generated outputs; do not infer findings from the configuration alone.

### Required result table

| Mechanism | Rate | Model | Imputer | Calibrator | Brier mean ± SD | ECE mean ± SD | ROC AUC mean ± SD |
|---|---:|---|---|---|---:|---:|---:|
| Pending experiment output | 0%–30% | Pending | Pending | Pending | Pending | Pending | Pending |

## Discussion

No empirical discussion is valid until the full configured grid has run. Interpretation should distinguish calibration from discrimination: a lower Brier or ECE does not establish higher predictive usefulness, and a higher AUC does not establish calibrated probabilities. Results should also distinguish consistent paired effects from noise across folds and seeds.

The intended decision rule is comparative and mechanism-specific: recommend a method only when its calibration improvement is supported across paired folds, does not create unacceptable discrimination loss, and remains interpretable under the stated limitations. Otherwise report the result as inconclusive.

## Limitations

- The universe contains 32 configured U.S. equities and daily observations; findings may not generalize to other assets, markets, frequencies, or regimes.
- MCAR and MAR are simulations, not a complete model of production data collection or market microstructure.
- The MAR design uses one configured dependence rule and one target feature; sensitivity to other dependence structures is not established.
- Only logistic regression and LightGBM are evaluated.
- Walk-forward windows and purge gaps reduce leakage risk but do not remove all dependence among financial observations.
- ECE depends on binning and can be unstable in small test folds.
- Historical data and backtest calibration do not establish profitability or suitability for live trading.
- Current report status is incomplete until data, metrics, figures, and metadata are generated and audited.

## Reproducibility

From a fresh environment with Python 3.11 or later:

```bash
pip install -r requirements.txt
make data
make features
make experiment
make figures
```

The data download date, Python version, package versions, random seeds, and configuration are recorded in `results/run_metadata.json`. Run `pytest -v` and `ruff check src tests scripts` before treating outputs as final.

## Conclusion

CalibMiss defines a reproducible test of probabilistic calibration under controlled MCAR and MAR feature missingness. Implementation and tests are present, but results are pending in this checkout. Final conclusions must be based on generated metrics, uncertainty summaries, corrected paired tests, and figures rather than assumptions about which method should perform best.
