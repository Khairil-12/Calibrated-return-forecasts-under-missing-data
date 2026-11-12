# CalibMiss pipeline run report

Run date: 2026-11-12. Machine: Windows 11, Python 3.12.6, 16 logical cores.

`make all` was not invoked as one command. Its stages were run individually in order, because the first full attempt crashed and two code fixes were required before the grid could finish. Every stage below maps to a `make` target.

## Stage results

| Stage | Command | Result |
|---|---|---|
| data | `python scripts/download_data.py` | OK. 113,190 raw rows, 30 tickers, cached to `data/raw/` with SHA256 in filename |
| features | `python scripts/compute_features.py` | OK. 106,028 feature rows written to `data/processed/` |
| experiment | `python scripts/run_grid.py` | OK. Exit code 0, 24.5 min for the parallel grid, 15,994 rows in `results/metrics.csv` |
| figures | `python scripts/generate_figures.py` | OK. 5 PNGs in `results/figures/` |
| report (PDF) | `pandoc reports/report.md -o reports/report.pdf` | NOT RUN. `pandoc` and `weasyprint` are not installed. The Makefile swallows this with `\|\| echo ... skipping PDF`, so `make all` would report success here anyway |
| tests | `python -m pytest` | OK. 87 passed in 13.4 s |

## Errors hit and fixes

1. **Grid crashed immediately on first attempt.**
   `ValueError: depends_on values must be finite` inside `inject_mar` (`src/calibmiss/missingness.py:98`), raised from a joblib worker.
   Cause: `compute_features` keeps the rolling-window warmup rows, so `volume_ratio` (the MAR driver) contained 553 NaN values. MCAR never touches that column's values, so the earlier single-config MCAR run passed and hid this.
   Fix: drop rows with any NaN feature before the matrix is used, in both `scripts/compute_features.py` and `_load_data` in `src/calibmiss/run_experiment.py`. Row count went from 107,444 to 106,028. All 87 tests still pass.

2. **Grid would have taken hours.**
   First benchmark: 2.1 tasks/s, projected ~2 h for 15,840 tasks. LightGBM defaults to `n_jobs=-1`, so each of the 16 worker processes spawned threads for all cores and they fought each other.
   Fix: `LightGBMModel` now defaults `n_jobs=1` (`src/calibmiss/models.py`). A controlled comparison on the same data showed 14.25 s vs 1.59 s for 32 fits. Throughput rose to 5.6 tasks/s and the full grid finished in 24.5 min. Also set `LOKY_MAX_CPU_COUNT` so joblib stops warning about failing to detect physical cores on Windows.

## Warnings (not failures)

- `ConvergenceWarning: [IterativeImputer] Early stopping criterion not reached` fired repeatedly for the `iterative` imputer. It is configured with `max_iter: 10`, which is not enough for these matrices to converge. Imputed values are still produced and all iterative rows completed; results for that imputer are usable but the imputation is under-converged. Raising `max_iter` would slow the slowest cells of the grid further.

## Headline numbers

15,840 experiment rows (22 walk-forward folds x 5 seeds x 2 models x 3 imputers x 3 calibrators x 2 mechanisms x 4 rates) plus 154 baseline rows.

- **Nothing beats the naive base rate.** Predicting the training-fold up-day frequency gives Brier 0.2489, ECE 0.0170, ROC AUC 0.500. The best model configuration (logistic regression, median imputer, Platt, MAR, 30% missing) reaches Brier 0.2488 and ROC AUC 0.517 — a Brier improvement of 0.0001, inside the noise. Discrimination is effectively absent across the whole grid (ROC AUC 0.460 to 0.571).

- **Calibration, not imputation, moves the metrics.** For logistic regression with median imputation under MCAR, going from 0% to 30% missingness changes ECE by -0.001 (0.0397 to 0.0385) and Brier not at all. Adding Platt scaling at 0% missingness cuts ECE in half, 0.0397 to 0.0199, and lowers Brier from 0.2503 to 0.2488. Uncalibrated LightGBM is the worst-calibrated cell in the grid (ECE 0.0512 at 0% missing) and Platt brings it to 0.0191.

- **592 of 1,728 paired comparisons are significant after Holm correction**, but the effects are small and concentrated on the calibrator contrast: calibrated vs uncalibrated LightGBM differs by about 0.004 to 0.005 Brier. Missingness rate and mechanism (MCAR vs MAR) produce no meaningful degradation anywhere in the grid.

## Outputs produced

- `results/metrics.csv` — 10.7 MB, 15,994 rows
- `results/summary.csv` — 144 configuration summaries with mean/std/count
- `results/statistical_tests.csv` — 1,728 paired Wilcoxon tests with Holm-adjusted p-values
- `results/run_metadata.json` — download date and package versions
- `results/figures/brier_vs_missingness.png`, `ece_vs_missingness.png`, `heatmap_calibration.png`, `reliability_0pct.png`, `reliability_30pct.png`

## Sanity check: shuffled labels

The highest AUC in the grid, 0.571, is a single fold (fold 13, seed 999, logistic regression, median imputation, MCAR, 10%) and is identical for all three calibrators, so it comes from the base model rather than from calibration. Rerunning five configurations with the target permuted (seed 20261112):

| Config | Real AUC | Shuffled AUC |
|---|---:|---:|
| fold 13, logistic regression, median, MCAR 10% (the 0.571 cell) | 0.5707 | 0.4986 |
| same, Platt | 0.5707 | 0.4986 |
| fold 0, logistic regression, median, Platt, MAR 30% | 0.5092 | 0.5017 |
| fold 5, LightGBM, iterative, MCAR 0% | 0.5345 | 0.5185 |
| fold 10, LightGBM, forward fill, Platt, MAR 20% | 0.4853 | 0.5156 |

Shuffled AUC sits at 0.5. The 0.571 is a fold-level fluctuation, not label leakage. 481 of 15,840 rows exceed 0.55, which is the expected scatter around a true AUC of 0.5.

## Not committed

The two fixes (`scripts/compute_features.py`, `src/calibmiss/run_experiment.py`, `src/calibmiss/models.py`), the regenerated `data/processed/` parquet files, and everything under `results/` are uncommitted.
