"""Missingness simulation (MCAR and MAR) for CalibMiss."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MARRuleConfig:
    """Configuration for MAR missingness rule."""

    target_feature: str
    depends_on: str
    rule_type: str
    params: dict[str, Any]


def inject_mcar(X: pd.DataFrame, rate: float, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Inject Missing Completely At Random (MCAR) missingness.

    Args:
        X: Complete feature matrix
        rate: Target missing rate (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame with NaN, dict of realized missing rates per column)
    """
    if not 0 <= rate <= 1:
        raise ValueError("Rate must be between 0 and 1")

    rng = np.random.RandomState(seed)
    X_missing = X.copy()

    # Generate missing mask for each value independently
    mask = rng.random(X.shape) < rate
    X_missing = X_missing.mask(mask)

    # Compute realized missing rates per column
    realized = {}
    for col in X.columns:
        realized[col] = float(X_missing[col].isna().mean())

    return X_missing, realized


def inject_mar(
    X: pd.DataFrame, rule: MARRuleConfig, target_rate: float, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Inject Missing At Random (MAR) missingness based on observed variable.

    Uses quantile_threshold rule: missing probability = base_prob +
    (max_prob - base_prob) * I(depends_on > quantile_threshold).
    Calibrates base_prob/max_prob to achieve target_rate overall.

    Args:
        X: Complete feature matrix
        rule: MAR rule configuration
        target_rate: Target missing rate for target_feature (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (DataFrame with NaN, dict of realized missing rates per column)
    """
    if rule.rule_type != "quantile_threshold":
        raise ValueError(f"Unknown rule_type: {rule.rule_type}")
    if not 0 <= target_rate <= 1 or not np.isfinite(target_rate):
        raise ValueError("target_rate must be finite and between 0 and 1")
    if X.empty:
        raise ValueError("X must not be empty")

    if rule.target_feature not in X.columns:
        raise ValueError(f"Target feature {rule.target_feature} not in DataFrame")

    if rule.depends_on not in X.columns:
        raise ValueError(f"Depends on feature {rule.depends_on} not in DataFrame")

    rng = np.random.RandomState(seed)
    X_missing = X.copy()

    # Extract parameters
    base_prob = rule.params.get("base_prob", 0.05)
    max_prob = rule.params.get("max_prob", 0.5)
    quantile_threshold = rule.params.get("quantile_threshold", 0.7)
    if (
        not np.isfinite(base_prob)
        or not np.isfinite(max_prob)
        or not 0 <= base_prob <= max_prob <= 1
    ):
        raise ValueError("base_prob and max_prob must satisfy 0 <= base_prob <= max_prob <= 1")
    if not np.isfinite(quantile_threshold) or not 0 <= quantile_threshold <= 1:
        raise ValueError("quantile_threshold must be finite and between 0 and 1")

    observed = X[rule.depends_on].to_numpy()
    if not np.isfinite(observed).all():
        raise ValueError("depends_on values must be finite")
    threshold = np.quantile(observed, quantile_threshold)
    high_group = observed > threshold
    high_share = float(high_group.mean())
    if target_rate in (0.0, 1.0) or high_share in (0.0, 1.0):
        probs = np.full(len(X), target_rate)
    else:
        separation = min(target_rate / high_share, (1.0 - target_rate) / (1.0 - high_share))
        low_prob = target_rate - high_share * separation
        high_prob = target_rate + (1.0 - high_share) * separation
        probs = np.where(high_group, high_prob, low_prob)

    # Generate missing mask
    missing_mask = rng.random(len(X)) < probs

    # Apply missingness only to target feature
    X_missing[rule.target_feature] = X_missing[rule.target_feature].mask(missing_mask)

    # Compute realized missing rates per column
    realized = {}
    for col in X.columns:
        realized[col] = float(X_missing[col].isna().mean())

    return X_missing, realized


def simulate_missingness_grid(
    X: pd.DataFrame,
    mechanisms: list[str],
    rates: list[float],
    seeds: list[int],
    mar_rules: dict[str, MARRuleConfig],
) -> dict[str, dict[float, dict[int, tuple[pd.DataFrame, dict[str, float]]]]]:
    """Simulate missingness grid across all combinations.

    Args:
        X: Complete feature matrix
        mechanisms: List of mechanisms ("MCAR", "MAR")
        rates: List of missing rates
        seeds: List of random seeds
        mar_rules: Dict mapping mechanism to MAR rule config

    Returns:
        Nested dict: results[mechanism][rate][seed] = (X_missing, realized)
    """
    results: dict[str, dict[float, dict[int, tuple[pd.DataFrame, dict[str, float]]]]] = {}

    for mechanism in mechanisms:
        results[mechanism] = {}
        for rate in rates:
            results[mechanism][rate] = {}
            for seed in seeds:
                if mechanism == "MCAR":
                    X_miss, realized = inject_mcar(X, rate, seed)
                elif mechanism == "MAR":
                    if "MAR" not in mar_rules:
                        raise ValueError("MAR rule config required for MAR mechanism")
                    rule = mar_rules["MAR"]
                    X_miss, realized = inject_mar(X, rule, rate, seed)
                else:
                    raise ValueError(f"Unknown mechanism: {mechanism}")

                results[mechanism][rate][seed] = (X_miss, realized)

    return results
