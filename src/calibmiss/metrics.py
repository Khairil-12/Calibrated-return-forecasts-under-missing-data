"""Binary classification metrics and reliability diagram data."""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score


class ReliabilityData(TypedDict):
    bin_edges: list[float]
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


class MetricsResult(TypedDict):
    brier: float
    ece: float
    roc_auc: float
    reliability: ReliabilityData


def _validate_inputs(y_true: NDArray[np.int_], y_prob: NDArray[np.float64], n_bins: int) -> None:
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be non-empty 1D arrays of equal length")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")


def _bin_indices(
    y_prob: NDArray[np.float64], n_bins: int
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    return indices, edges


def compute_brier(y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
    _validate_inputs(y_true, y_prob, 1)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(y_true: NDArray[np.int_], y_prob: NDArray[np.float64], n_bins: int = 10) -> float:
    _validate_inputs(y_true, y_prob, n_bins)
    indices, _ = _bin_indices(y_prob, n_bins)
    ece = 0.0
    for index in range(n_bins):
        mask = indices == index
        if np.any(mask):
            accuracy = np.mean(y_true[mask])
            confidence = np.mean(y_prob[mask])
            ece += np.count_nonzero(mask) / len(y_true) * abs(accuracy - confidence)
    return float(ece)


def compute_roc_auc(y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
    _validate_inputs(y_true, y_prob, 1)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_reliability_data(
    y_true: NDArray[np.int_], y_prob: NDArray[np.float64], n_bins: int = 10
) -> ReliabilityData:
    _validate_inputs(y_true, y_prob, n_bins)
    indices, edges = _bin_indices(y_prob, n_bins)
    accuracies: list[float] = []
    confidences: list[float] = []
    counts: list[int] = []

    for index in range(n_bins):
        mask = indices == index
        counts.append(int(np.count_nonzero(mask)))
        if np.any(mask):
            accuracies.append(float(np.mean(y_true[mask])))
            confidences.append(float(np.mean(y_prob[mask])))
        else:
            accuracies.append(0.0)
            confidences.append(0.0)

    return {
        "bin_edges": edges.tolist(),
        "bin_accuracies": accuracies,
        "bin_confidences": confidences,
        "bin_counts": counts,
    }


def compute_metrics(
    y_true: NDArray[np.int_], y_prob: NDArray[np.float64], n_bins: int = 10
) -> MetricsResult:
    return {
        "brier": compute_brier(y_true, y_prob),
        "ece": compute_ece(y_true, y_prob, n_bins),
        "roc_auc": compute_roc_auc(y_true, y_prob),
        "reliability": compute_reliability_data(y_true, y_prob, n_bins),
    }
