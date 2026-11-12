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

## Findings

The full grid (22 folds, 5 seeds, 15,840 runs) shows:

- **No model beats the naive base rate.** Training-fold up-day frequency scores Brier 0.2489 and ECE 0.0170. The best model cell reaches Brier 0.2488 and ROC AUC 0.517. Next-day direction is effectively unpredictable from these features, which is what efficient-market intuition predicts. This is a calibration result, not evidence that the models predict returns.
- **Calibration fixes overconfidence, not predictions.** Platt scaling halves logistic-regression ECE (0.0397 → 0.0199) and cuts uncalibrated LightGBM ECE from 0.0512 to 0.0191, with ROC AUC unchanged. The significant paired tests are calibrator contrasts of ~0.005 Brier.
- **Missingness looks harmless only because there is no signal to lose.** ECE moves ~0.001 from 0% to 30% missingness under both MCAR and MAR. That does not answer whether missingness degrades calibration; a target with real signal (e.g. realized volatility) is needed for that.

## Scope limits on the statistics

Walk-forward folds overlap in time and seeds share the same data, so the paired tests treat dependent observations as independent and the Holm-corrected p-values overstate the evidence. The iterative imputer does not converge at its configured iteration cap, so its numbers are approximate.

## Reproduce

```bash
pip install -r requirements.txt
make data
make features
make experiment
make figures
pytest -v
```

## Scope limits

The study uses daily data for the configured U.S. equity universe, simulated MCAR/MAR mechanisms, logistic regression and LightGBM, and a finite historical period. Calibration findings do not demonstrate trading profitability or live deployment suitability.
