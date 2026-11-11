"""Tests for single-configuration experiment orchestration."""

import numpy as np
import pandas as pd
import pytest
from calibmiss.pipeline import run_single_config


def make_data() -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    X = pd.DataFrame(
        rng.normal(size=(120, 4)), columns=["volatility_20d", "volume_ratio", "c", "d"]
    )
    y = pd.Series((X["volatility_20d"] + X["volume_ratio"] > 0).astype(int))
    return X, y, np.arange(60), np.arange(60, 90), np.arange(90, 120)


def make_config(rate: float = 0.1) -> dict[str, object]:
    return {
        "model": "logistic_regression",
        "imputer": "median",
        "calibrator": "platt",
        "mechanism": "MCAR",
        "rate": rate,
        "seed": 42,
    }


def test_run_single_config_returns_metrics_and_configuration() -> None:
    X, y, train, cal, test = make_data()

    result = run_single_config(X, y, train, cal, test, make_config())

    assert {"brier", "ece", "roc_auc", "reliability"} <= result.keys()
    assert result["model"] == "logistic_regression"
    assert result["rate"] == 0.1
    assert result["seed"] == 42
    assert result["fold"] == 0


def test_zero_missingness_baseline_runs() -> None:
    X, y, train, cal, test = make_data()

    result = run_single_config(X, y, train, cal, test, make_config(rate=0.0))

    assert np.isfinite(result["brier"])
    assert result["rate"] == 0.0


def test_unknown_mechanism_is_rejected_even_at_zero_rate() -> None:
    X, y, train, cal, test = make_data()
    config = make_config(rate=0.0)
    config["mechanism"] = "MNAR"

    with pytest.raises(ValueError, match="Unknown mechanism: MNAR"):
        run_single_config(X, y, train, cal, test, config)


def test_mar_missingness_uses_configured_rule() -> None:
    X, y, train, cal, test = make_data()
    config = make_config()
    config["mechanism"] = "MAR"

    result = run_single_config(X, y, train, cal, test, config)

    assert result["mechanism"] == "MAR"
    assert np.isfinite(result["brier"])


def test_explicit_model_params_override_yaml() -> None:
    X, y, train, cal, test = make_data()
    config = make_config(rate=0.0)
    config["model_params"] = {"C": 0.01, "max_iter": 1000, "class_weight": None}

    result = run_single_config(X, y, train, cal, test, config)

    assert np.isfinite(result["brier"])
