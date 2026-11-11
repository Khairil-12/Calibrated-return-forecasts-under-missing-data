# CalibMiss Executive Summary

## Purpose

CalibMiss tests whether next-day stock-direction probabilities stay calibrated when engineered input features contain missing values. It compares MCAR and MAR missingness, three imputers, two model families, and three calibration choices using chronological walk-forward evaluation.

## What the pipeline measures

- **Brier score:** squared probability error; lower is better.
- **ECE:** ten-bin expected calibration error; lower is better.
- **Reliability diagrams:** observed frequency versus predicted probability.
- **ROC AUC:** secondary discrimination measure.
- **Stability:** fold and seed dispersion, paired tests, signed-rank effects, and Holm-adjusted p-values.

The pipeline includes complete-data oracle and naive training-prevalence baselines. Imputers fit on training data only; calibrators fit on a separate calibration split; final metrics use a held-out test split.

## Current status

Results are pending. This checkout does not contain `results/metrics.csv`, summaries, metadata, or generated figures. Therefore no method is recommended and no claim is made about MCAR versus MAR performance.

## Reproduce

```bash
pip install -r requirements.txt
make data
make features
make experiment
make figures
pytest -v
```

After the run, replace this status with observed Brier/ECE trends, uncertainty estimates, ROC AUC trade-offs, and corrected paired-test results from `results/summary.csv` and `results/statistical_tests.csv`.

## Scope limits

The study uses daily data for the configured U.S. equity universe, simulated MCAR/MAR mechanisms, logistic regression and LightGBM, and a finite historical period. Calibration findings do not demonstrate trading profitability or live deployment suitability.
