"""Missingness simulation (MCAR and MAR) for CalibMiss."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


@dataclass
class MARRuleConfig:
    """Configuration for MAR missingness rule."""
    target_feature: str
    depends_on: str
    rule_type: str
    params: dict[str, Any]


def inject_mcar(
    X: pd.DataFrame,
    rate: float,
    seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
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
    X: pd.DataFrame,
    rule: MARRuleConfig,
    target_rate: float,
    seed: int
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

    # Get the observed variable
    observed = X[rule.depends_on].values

    # Compute threshold
    threshold = np.quantile(observed, quantile_threshold)

    # Compute missing probabilities
    # High group: probability = max_prob
    # Low group: probability = base_prob
    high_group = observed > threshold
    probs = np.where(high_group, max_prob, base_prob)

    # Calibrate to achieve target_rate
    # We need to scale probs so that mean(probs) = target_rate
    # Simple approach: adjust max_prob while keeping base_prob fixed
    # Or adjust both proportionally
    current_rate = probs.mean()
    if current_rate > 0:
        scale = target_rate / current_rate
        probs = probs * scale
        # Clip to [0, 1]
        probs = np.clip(probs, 0, 1)

    # Generate missing mask
    missing_mask = rng.random(len(X)) < probs

    # Apply missingness only to target feature
    target_idx = X.columns.get_loc(rule.target_feature)
    X_missing.iloc[missing_mask, target_idx] = np.nan

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
    mar_rules: dict[str, MARRuleConfig]
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
