"""Tests for probability calibration methods."""

import numpy as np
import pytest
from calibmiss.calibration import (
    IsotonicCalibrator,
    NoCalibrator,
    PlattCalibrator,
    get_calibrator,
)
from calibmiss.config import CalibratorConfig


def test_platt_scaling_reduces_extreme_probabilities() -> None:
    p_cal = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
    y_cal = np.array([1, 1, 0, 0, 1, 0])
    p_test = np.array([0.85, 0.15, 0.65])

    calibrated = PlattCalibrator().fit(p_cal, y_cal).predict_proba(p_test)

    assert calibrated.shape == (3,)
    assert calibrated[0] < 0.85
    assert calibrated[1] > 0.15
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_isotonic_regression_is_monotonic_and_clips_bounds() -> None:
    p_cal = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.4])
    y_cal = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p_test = np.array([0.85, 0.15, 0.65, 0.35, -0.1, 1.1])

    calibrated = IsotonicCalibrator().fit(p_cal, y_cal).predict_proba(p_test)

    order = np.argsort(p_test[:4])
    assert np.all(np.diff(calibrated[:4][order]) >= -1e-10)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_no_calibrator_is_identity() -> None:
    p = np.array([0.1, 0.9])
    calibrator = NoCalibrator().fit(np.array([0.5, 0.5]), np.array([0, 1]))

    np.testing.assert_array_equal(calibrator.predict_proba(p), p)


def test_fit_uses_only_supplied_calibration_data() -> None:
    p_cal = np.array([0.8, 0.2, 0.7, 0.3])
    y_cal = np.array([1, 0, 1, 0])
    p_test = np.array([0.7])

    for calibrator_type in (PlattCalibrator, IsotonicCalibrator):
        calibrator = calibrator_type().fit(p_cal, y_cal)
        before = calibrator.predict_proba(p_test)
        calibrator.predict_proba(np.array([0.01, 0.99]))
        after = calibrator.predict_proba(p_test)
        np.testing.assert_array_equal(before, after)


def test_get_calibrator_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown calibrator: other"):
        get_calibrator(CalibratorConfig(name="other"))


def test_get_calibrator_returns_configured_method() -> None:
    expected = {
        "none": NoCalibrator,
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
    }
    for name, calibrator_type in expected.items():
        assert isinstance(get_calibrator(CalibratorConfig(name=name)), calibrator_type)
