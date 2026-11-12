"""Tests for model wrappers."""

import numpy as np
import pandas as pd
import pytest
from calibmiss.config import ModelConfig
from calibmiss.models import LightGBMModel, LogisticRegressionModel, get_model


def make_training_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.normal(size=(100, 5)))
    labels = pd.Series((features[0] + features[1] > 0).astype(int))
    return features, labels


def assert_valid_probabilities(probabilities: np.ndarray, n_samples: int) -> None:
    assert probabilities.shape == (n_samples,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_logistic_regression_standardizes_and_predicts_positive_class() -> None:
    X, y = make_training_data()
    model = LogisticRegressionModel(C=1.0, max_iter=1000, class_weight="balanced")

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    assert_valid_probabilities(probabilities, len(X))
    np.testing.assert_allclose(model.scaler.mean_, X.mean().to_numpy())


def test_lightgbm_predicts_positive_class() -> None:
    X, y = make_training_data()
    model = LightGBMModel(n_estimators=10, max_depth=3, learning_rate=0.1, verbosity=-1)

    model.fit(X, y)
    probabilities = model.predict_proba(X)

    assert_valid_probabilities(probabilities, len(X))


def test_get_model_returns_configured_model() -> None:
    logistic = get_model(ModelConfig(name="logistic_regression", params={"max_iter": 321}))
    lightgbm = get_model(ModelConfig(name="lightgbm", params={"n_estimators": 12}))

    assert isinstance(logistic, LogisticRegressionModel)
    assert logistic.model.max_iter == 321
    assert isinstance(lightgbm, LightGBMModel)
    assert lightgbm.model.n_estimators == 12


def test_get_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown model: other"):
        get_model(ModelConfig(name="other"))


def test_logistic_regression_class_weight_default_is_none() -> None:
    """class_weight default must be None, not 'balanced'.

    'balanced' skews predicted probabilities toward 0.5, which Platt scaling
    then appears to 'correct' — creating an ECE artifact. The grid runs must
    use no class weighting by default.
    """
    model = LogisticRegressionModel()
    # LogisticRegressionModel exposes model.class_weight via the wrapped model
    assert model.model.class_weight is None
