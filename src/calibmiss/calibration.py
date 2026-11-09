"""Probability calibration strategies."""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from calibmiss.config import CalibratorConfig


class PlattCalibrator:
    """Fit logistic regression to calibration-set probabilities."""

    def fit(self, probas: NDArray[np.float64], labels: NDArray[np.int_]) -> "PlattCalibrator":
        self.model = LogisticRegression()
        self.model.fit(probas.reshape(-1, 1), labels)
        return self

    def predict_proba(self, probas: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self.model.predict_proba(probas.reshape(-1, 1))[:, 1])


class IsotonicCalibrator:
    """Fit a monotonic probability mapping on the calibration split."""

    def fit(self, probas: NDArray[np.float64], labels: NDArray[np.int_]) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(probas, labels)
        return self

    def predict_proba(self, probas: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self.model.predict(probas))


class NoCalibrator:
    """Return probabilities unchanged."""

    def fit(self, probas: NDArray[np.float64], labels: NDArray[np.int_]) -> "NoCalibrator":
        return self

    def predict_proba(self, probas: NDArray[np.float64]) -> NDArray[np.float64]:
        return probas


def get_calibrator(
    config: CalibratorConfig,
) -> PlattCalibrator | IsotonicCalibrator | NoCalibrator:
    """Build a calibrator from experiment configuration."""
    if config.name == "none":
        return NoCalibrator()
    if config.name == "platt":
        return PlattCalibrator()
    if config.name == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibrator: {config.name}")
