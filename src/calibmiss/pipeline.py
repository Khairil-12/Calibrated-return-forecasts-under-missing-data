"""Run one model/imputation/calibration configuration on a walk-forward fold."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from calibmiss.calibration import get_calibrator
from calibmiss.config import CalibratorConfig, ImputerConfig, ModelConfig, get_config, get_mar_rules
from calibmiss.imputers import get_imputer
from calibmiss.metrics import compute_metrics
from calibmiss.missingness import MARRuleConfig, inject_mar, inject_mcar
from calibmiss.models import get_model


def run_single_config(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: NDArray[np.int_],
    cal_idx: NDArray[np.int_],
    test_idx: NDArray[np.int_],
    config: dict[str, Any],
    experiment_config: Any = None,
) -> dict[str, Any]:
    """Run a single configured experiment and return test metrics with metadata."""
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_cal, y_cal = X.iloc[cal_idx], y.iloc[cal_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    rate = float(config["rate"])
    seed = int(config["seed"])
    mechanism = str(config["mechanism"])
    config_dir = Path(__file__).resolve().parents[2] / "config"
    if mechanism not in ("MCAR", "MAR"):
        raise ValueError(f"Unknown mechanism: {mechanism}")
    experiment = (experiment_config or get_config(config_dir)).experiment
    if rate > 0:
        if mechanism == "MCAR":
            X_train, _ = inject_mcar(X_train, rate, seed)
            X_cal, _ = inject_mcar(X_cal, rate, seed + 1)
            X_test, _ = inject_mcar(X_test, rate, seed + 2)
        elif mechanism == "MAR":
            rules = (
                experiment_config.mar_rules
                if experiment_config is not None and hasattr(experiment_config, "mar_rules")
                else get_mar_rules(config_dir)
            )
            rule_cfg = rules.mar_rules[0]
            rule = MARRuleConfig(
                target_feature=rule_cfg.target_feature,
                depends_on=rule_cfg.depends_on,
                rule_type=rule_cfg.rule_type,
                params=rule_cfg.params,
            )
            for label, part in (("train", X_train), ("calibrate", X_cal), ("test", X_test)):
                if rule.depends_on in part.columns:
                    bad = part.loc[~np.isfinite(part[rule.depends_on].to_numpy()), rule.depends_on]
                    if len(bad):
                        raise ValueError(
                            f"MAR driver column '{rule.depends_on}' has {len(bad)} non-finite "
                            f"value(s) in the {label} split (rows: {list(bad.index[:5])}); "
                            "drop warmup rows before injecting missingness"
                        )
            X_train, _ = inject_mar(X_train, rule, rate, seed)
            X_cal, _ = inject_mar(X_cal, rule, rate, seed + 1)
            X_test, _ = inject_mar(X_test, rule, rate, seed + 2)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

    imputer_params = config.get("imputer_params")
    if imputer_params is None:
        imputer_params = next(
            (item.params for item in experiment.imputers if item.name == config["imputer"]),
            {},
        )
    imputer = get_imputer(ImputerConfig(name=str(config["imputer"]), params=imputer_params))
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_cal = imputer.transform(X_cal)
    X_test = imputer.transform(X_test)

    model_params = config.get("model_params")
    if model_params is None:
        model_params = next(
            (item.params for item in experiment.models if item.name == config["model"]),
            {},
        )
    model = get_model(ModelConfig(name=str(config["model"]), params=model_params))
    model.fit(X_train, y_train)

    p_cal = model.predict_proba(X_cal)
    p_test = model.predict_proba(X_test)
    calibrator_params = config.get("calibrator_params")
    if calibrator_params is None:
        calibrator_params = next(
            (item.params for item in experiment.calibrators if item.name == config["calibrator"]),
            {},
        )
    calibrator = get_calibrator(
        CalibratorConfig(name=str(config["calibrator"]), params=calibrator_params)
    )
    calibrator.fit(p_cal, y_cal.to_numpy())
    p_test_calibrated = calibrator.predict_proba(p_test)

    metrics = compute_metrics(
        y_test.to_numpy(),
        p_test_calibrated,
        n_bins=experiment.evaluation.n_bins_ece,
    )
    return {**metrics, **config, "fold": 0}
