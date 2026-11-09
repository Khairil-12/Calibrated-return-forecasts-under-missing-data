"""Tests for evaluation metrics."""

import numpy as np
import pytest
from calibmiss.metrics import compute_metrics


def test_brier_score_matches_manual_calculation() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    result = compute_metrics(y_true, y_prob, n_bins=10)

    assert abs(result["brier"] - np.mean((y_prob - y_true) ** 2)) < 1e-10


def test_ece_is_zero_for_balanced_half_probability_bin() -> None:
    y_true = np.array([0, 1] * 50)
    y_prob = np.full(100, 0.5)

    result = compute_metrics(y_true, y_prob, n_bins=10)

    assert result["ece"] == 0.0


def test_roc_auc_is_one_for_perfect_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    result = compute_metrics(y_true, y_prob)

    assert result["roc_auc"] == 1.0


def test_one_class_labels_return_nan_auc_and_other_metrics() -> None:
    result = compute_metrics(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.8]))

    assert np.isnan(result["roc_auc"])
    assert np.isfinite(result["brier"])
    assert np.isfinite(result["ece"])
    assert sum(result["reliability"]["bin_counts"]) == 3


@pytest.mark.parametrize(
    ("y_true", "y_prob", "n_bins"),
    [
        (np.array([], dtype=int), np.array([]), 10),
        (np.array([0]), np.array([0.5, 0.6]), 10),
        (np.array([0, 2]), np.array([0.2, 0.8]), 10),
        (np.array([0, 1]), np.array([0.2, np.nan]), 10),
        (np.array([0, 1]), np.array([0.2, 1.1]), 10),
        (np.array([0, 1]), np.array([0.2, 0.8]), 0),
    ],
)
def test_metrics_reject_invalid_inputs(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> None:
    with pytest.raises(ValueError):
        compute_metrics(y_true, y_prob, n_bins=n_bins)


def test_reliability_data_counts_boundary_probabilities_and_empty_bins() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.0, 0.1, 0.9, 1.0])

    result = compute_metrics(y_true, y_prob, n_bins=10)
    reliability = result["reliability"]

    assert len(reliability["bin_edges"]) == 11
    assert len(reliability["bin_accuracies"]) == 10
    assert len(reliability["bin_confidences"]) == 10
    assert len(reliability["bin_counts"]) == 10
    assert reliability["bin_counts"] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert reliability["bin_accuracies"] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
