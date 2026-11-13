"""Diagnostic: does class_weight='balanced' drive the Platt ECE result?

Trains LR (balanced vs None) on fold 0, MCAR 0%, median imputer, and compares
mean predicted probability vs train base rate plus ECE with/without Platt.
"""

import sys

sys.path.insert(0, "src")

import numpy as np

from calibmiss.calibration import NoCalibrator, PlattCalibrator
from calibmiss.config import ImputerConfig, get_config
from calibmiss.data import download_ohlcv
from calibmiss.features import compute_features
from calibmiss.imputers import get_imputer
from calibmiss.metrics import compute_metrics
from calibmiss.models import LogisticRegressionModel
from calibmiss.run_experiment import map_date_splits_to_rows
from calibmiss.splits import get_split_indices

cfg = get_config("config")
raw = download_ohlcv(
    cfg.universe.tickers, cfg.universe.start_date, cfg.universe.end_date, cache_dir="data/raw"
)
X, y = compute_features(raw)
valid = ~X.isna().any(axis=1) & X.index.isin(y.index)
X = X[valid]
y = y.reindex(X.index)

dates = X.index.get_level_values("date").unique().sort_values()
splits = map_date_splits_to_rows(X, get_split_indices(dates, cfg.experiment.splits.model_dump()))
split = splits[0]
train_idx, cal_idx, test_idx = split.train_idx, split.calibrate_idx, split.test_idx

base_rate = float(np.mean(y.iloc[train_idx].to_numpy()))
print(f"fold 0 train base rate: {base_rate:.4f}  (n_train={len(train_idx)})")

imputer = get_imputer(ImputerConfig(name="median", params={}))
imputer.fit(X.iloc[train_idx])
X_train = imputer.transform(X.iloc[train_idx])
X_cal = imputer.transform(X.iloc[cal_idx])
X_test = imputer.transform(X.iloc[test_idx])
y_train, y_cal, y_test = (
    y.iloc[train_idx].to_numpy(),
    y.iloc[cal_idx].to_numpy(),
    y.iloc[test_idx].to_numpy(),
)

for cw in ("balanced", None):
    model = LogisticRegressionModel(C=1.0, max_iter=1000, class_weight=cw)
    model.fit(X_train, y_train)
    p_cal = model.predict_proba(X_cal)
    p_test = model.predict_proba(X_test)

    for cal_name, calibrator in (("none", NoCalibrator()), ("platt", PlattCalibrator())):
        calibrator.fit(p_cal, y_cal)
        p = calibrator.predict_proba(p_test)
        m = compute_metrics(y_test, p, n_bins=10)
        print(
            f"class_weight={cw!s:9s} calibrator={cal_name:5s} "
            f"mean_p_cal={np.mean(p_cal):.4f} mean_p_test={np.mean(p):.4f} "
            f"brier={m['brier']:.4f} ece={m['ece']:.4f} auc={m['roc_auc']:.4f}"
        )
print(
    f"\nreference: train base rate {base_rate:.4f}; a calibrated model's mean p should be near it"
)
