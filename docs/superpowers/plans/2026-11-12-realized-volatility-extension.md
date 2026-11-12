# Realized Volatility Extension

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second experiment where the target has real predictable signal, so the missingness question the direction study could not answer becomes answerable.

**Why this exists:** The 2026-11-12 direction grid (see `reports/pipeline_run.md`) found ROC AUC ≈ 0.5 everywhere and ECE changing by ~0.001 from 0% to 30% missingness. That flat curve is uninformative: the models have no signal to lose, so missingness has nothing to degrade. Research question 1 is unanswered. Realized volatility is predictable from its own past (volatility clustering), which gives missingness something to damage and calibration something to recover.

**Architecture:** A second target computed from the same raw OHLCV cache, reusing `splits`, `missingness`, `imputers`, `models`, `calibration`, `metrics`, and `run_experiment` unchanged. New code is the target definition, its feature set, and a separate config plus output directory so the direction results stay untouched.

**Tech Stack:** Existing stack only. No new dependencies.

---

## Design decisions

- **Target:** next-period realized volatility, defined as the annualized standard deviation of daily log returns over the *next* 5 trading days, computed per ticker. This is known at day t+5, never at day t, so the label uses future data by construction and the features must not.
- **Regression, not classification.** The current pipeline is binary (`predict_proba`, Brier, ECE, ROC AUC) and none of those metrics apply to a continuous target. Two options:
  - (a) Keep the stack: binarize the target at the training-fold median (`1` if next-5-day realized volatility exceeds the training median). All of `metrics.py`, `calibration.py`, and the paired tests work unmodified. Cost: throws away magnitude.
  - (b) Add a regression path: models gain `predict`, metrics gain RMSE and a probability-integral-transform or coverage-based calibration check. This is the version that can actually say "calibration recovers what missingness damaged," but it touches every stage.
  - **Recommendation: (a) first.** It answers the question with the code that already exists and is testable in one pass. Do (b) only if (a) shows a real AUC gap over the base rate, which is the premise worth confirming before building a regression stack.
- **Baseline that must be beaten:** a persistence forecast — today's trailing 5-day realized volatility as the predictor of the next 5 days. If the models cannot beat persistence, volatility has no more signal here than direction did and the extension stops there.
- **Features:** reuse the existing eight, which are all known at day t. Add trailing realized volatility (`volatility_5d` already exists and is the persistence feature). Do not add anything derived from prices after day t.
- **Grid:** same as `config/experiment.yaml` — same imputers, calibrators, mechanisms, rates, seeds, splits. MAR keeps `volume_ratio` as the observed driver and targets `volatility_20d`, since that is the feature whose absence should hurt a volatility forecast most.
- **Outputs:** `results/volatility/` instead of `results/`, so `metrics.csv`, `summary.csv`, and figures for the two targets never overwrite each other. `generate_figures.py` needs the output directory parameterized, not hardcoded.
- **Warmup rows:** the direction run crashed because `volume_ratio` carried rolling-window NaNs into `inject_mar`. The volatility target adds another window, so the NaN-row drop in `scripts/compute_features.py` and `_load_data` must apply to it too. Assert the matrix is finite before the grid starts rather than discovering it 15,000 tasks in.

## Tasks

### Task 1: Realized volatility target

**Files:**
- Modify: `src/calibmiss/features.py`
- Test: `tests/test_features_lookahead.py`

- [ ] Add `compute_volatility_target(raw, horizon=5)` returning a Series aligned to the feature index: annualized std of the next `horizon` daily log returns, per ticker.
- [ ] Test the value against a hand-computed case on a short deterministic price series.
- [ ] Test lookahead safety the same way the direction features are tested: mutating prices inside the forward window must change the target, mutating prices after it must not.
- [ ] Test that the target's index is a subset of the feature index and contains no row whose features are NaN.

### Task 2: Binarized target and persistence baseline

**Files:**
- Modify: `src/calibmiss/run_experiment.py`
- Test: `tests/test_run_experiment.py`

- [ ] Binarize using the training fold only: threshold = median of `y[train_idx]`, applied to calibrate and test. Computing the threshold on anything but the training fold leaks the test distribution.
- [ ] Test that the threshold ignores calibrate and test labels.
- [ ] Add the persistence baseline next to `naive_training_base_rate`: predict from `volatility_5d` at day t. Report its Brier, ECE, and ROC AUC in the same baseline rows.

### Task 3: Separate config and output directory

**Files:**
- Create: `config/experiment_volatility.yaml`
- Modify: `scripts/generate_figures.py`, `src/calibmiss/run_experiment.py`
- Test: `tests/test_generate_figures.py`

- [ ] New config mirrors `experiment.yaml` with the volatility target named and the persistence baseline enabled.
- [ ] `run_grid` and `generate_figures` take the output directory instead of writing to hardcoded `results/`. Test that a run with the volatility config writes `results/volatility/` and leaves `results/metrics.csv` unchanged.

### Task 4: Finite-input guard

**Files:**
- Modify: `src/calibmiss/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] `run_single_config` raises `ValueError` naming the offending column if any feature used as a MAR driver is non-finite, before `inject_mar` does. This turns the failure mode from the direction run into an immediate, readable error.

### Task 5: Run and write up

- [ ] Run the volatility grid. Expect roughly the same 25 minutes as the direction grid.
- [ ] **Stop condition:** if no model beats the persistence baseline on ROC AUC by a margin visible above the fold noise, report that and stop. Do not interpret missingness effects on a target that also has no signal.
- [ ] If a gap exists, compare ECE at 0% and 30% missingness for the best model with and without Platt scaling, and state whether calibration recovers the degradation.
- [ ] Add the comparison to `reports/report.md` and a line to the README Key findings. Keep the direction result as the baseline it is.

## Out of scope

- The regression path (option b above), unless Task 5 shows signal worth a second stack.
- New model families. The point is the target, not the estimator.
- Any claim about trading or profitability. Same scope limit as the direction study.
