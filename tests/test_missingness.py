"""Tests for missingness simulation (MCAR and MAR)."""

import numpy as np
import pandas as pd
from calibmiss.missingness import MARRuleConfig, inject_mar, inject_mcar, simulate_missingness_grid


class TestMCAR:
    """Tests for MCAR missingness injection."""

    def test_mcar_produces_target_rate(self) -> None:
        """Test that MCAR produces the target missing rate."""
        X = pd.DataFrame(np.random.randn(1000, 10))
        X_missing, realized = inject_mcar(X, rate=0.2, seed=42)

        # Overall missing rate should be close to target
        overall_rate = X_missing.isna().mean().mean()
        assert abs(overall_rate - 0.2) < 0.02

        # Realized rates recorded per column
        assert len(realized) == 10
        for col, rate in realized.items():
            assert abs(rate - 0.2) < 0.03

    def test_mcar_with_different_rates(self) -> None:
        """Test MCAR at different missing rates."""
        X = pd.DataFrame(np.random.randn(1000, 5))
        for rate in [0.1, 0.2, 0.3]:
            X_missing, realized = inject_mcar(X, rate=rate, seed=42)
            overall = X_missing.isna().mean().mean()
            assert abs(overall - rate) < 0.02

    def test_mcar_reproducibility(self) -> None:
        """Test that MCAR is reproducible with same seed."""
        X = pd.DataFrame(np.random.randn(100, 5))
        X_miss1, _ = inject_mcar(X, rate=0.2, seed=42)
        X_miss2, _ = inject_mcar(X, rate=0.2, seed=42)
        pd.testing.assert_frame_equal(X_miss1, X_miss2)

    def test_mcar_different_seeds(self) -> None:
        """Test that different seeds produce different patterns."""
        X = pd.DataFrame(np.random.randn(100, 5))
        X_miss1, _ = inject_mcar(X, rate=0.2, seed=42)
        X_miss2, _ = inject_mcar(X, rate=0.2, seed=123)
        # Should have different NaN patterns
        assert not X_miss1.isna().equals(X_miss2.isna())


class TestMAR:
    """Tests for MAR missingness injection."""

    def test_mar_depends_on_observed_variable(self) -> None:
        """Test that MAR missingness depends on observed variable."""
        np.random.seed(42)
        X = pd.DataFrame(
            {"volatility_20d": np.random.randn(1000), "volume_ratio": np.random.rand(1000)}
        )

        rule = MARRuleConfig(
            target_feature="volatility_20d",
            depends_on="volume_ratio",
            rule_type="quantile_threshold",
            params={"base_prob": 0.05, "max_prob": 0.5, "quantile_threshold": 0.7},
        )
        X_missing, realized = inject_mar(X, rule, target_rate=0.2, seed=42)

        # Missingness in volatility_20d should correlate with volume_ratio
        missing_mask = X_missing["volatility_20d"].isna()
        high_vol_ratio = X["volume_ratio"] > X["volume_ratio"].quantile(0.7)

        # Higher missing rate in high volume_ratio group
        assert missing_mask[high_vol_ratio].mean() > missing_mask[~high_vol_ratio].mean()

    def test_mar_target_rate_achieved(self) -> None:
        """Test that MAR achieves target rate through calibration."""
        X = pd.DataFrame({"feature_a": np.random.randn(1000), "feature_b": np.random.rand(1000)})

        rule = MARRuleConfig(
            target_feature="feature_a",
            depends_on="feature_b",
            rule_type="quantile_threshold",
            params={"base_prob": 0.05, "max_prob": 0.5, "quantile_threshold": 0.7},
        )
        X_missing, realized = inject_mar(X, rule, target_rate=0.2, seed=42)

        # Overall missing rate in target feature should be close to target
        target_missing_rate = X_missing["feature_a"].isna().mean()
        assert abs(target_missing_rate - 0.2) < 0.03

    def test_mar_calibrates_boundary_and_intermediate_rates(self) -> None:
        X = pd.DataFrame({"target": np.zeros(20000), "driver": np.linspace(0, 1, 20000)})
        rule = MARRuleConfig(
            target_feature="target",
            depends_on="driver",
            rule_type="quantile_threshold",
            params={"quantile_threshold": 0.7},
        )

        for target_rate in (0.0, 0.1, 0.3, 0.8, 1.0):
            masked, realized = inject_mar(X, rule, target_rate=target_rate, seed=42)
            assert abs(realized["target"] - target_rate) < 0.015
            assert not masked["driver"].isna().any()

        masked, _ = inject_mar(X, rule, target_rate=0.3, seed=42)
        missing = masked["target"].isna()
        high = X["driver"] > X["driver"].quantile(0.7)
        assert missing[high].mean() > missing[~high].mean()

    def test_mar_rejects_invalid_rate_and_parameters(self) -> None:
        X = pd.DataFrame({"target": [1.0, 2.0], "driver": [0.0, 1.0]})
        base = MARRuleConfig(
            target_feature="target",
            depends_on="driver",
            rule_type="quantile_threshold",
            params={"base_prob": 0.05, "max_prob": 0.5, "quantile_threshold": 0.7},
        )

        import pytest

        for rate in (-0.1, 1.1):
            with pytest.raises(ValueError):
                inject_mar(X, base, target_rate=rate, seed=1)
        for params in (
            {"base_prob": 0.8, "max_prob": 0.5, "quantile_threshold": 0.7},
            {"base_prob": 0.05, "max_prob": 0.5, "quantile_threshold": 1.1},
        ):
            rule = MARRuleConfig("target", "driver", "quantile_threshold", params)
            with pytest.raises(ValueError):
                inject_mar(X, rule, target_rate=0.2, seed=1)

    def test_mar_accepts_precomputed_threshold(self) -> None:
        """Passing depends_on_threshold must override the quantile of the passed data.

        With a threshold far above every driver value, high_group is empty and
        the per-split quantile path would have produced a different split of
        the data. The realized rates prove the supplied threshold was used.
        """
        X = pd.DataFrame({"target": np.random.randn(20000), "driver": np.random.rand(20000)})
        rule = MARRuleConfig(
            target_feature="target",
            depends_on="driver",
            rule_type="quantile_threshold",
            params={"quantile_threshold": 0.7},
        )

        # Threshold above all driver values: high group empty -> probs = target_rate
        masked, realized = inject_mar(
            X, rule, target_rate=0.2, seed=42, depends_on_threshold=2.0
        )
        assert abs(realized["target"] - 0.2) < 0.02

        # Threshold below all driver values: everyone in high group, same probs
        masked_low, realized_low = inject_mar(
            X, rule, target_rate=0.2, seed=42, depends_on_threshold=-1.0
        )
        assert abs(realized_low["target"] - 0.2) < 0.02

    def test_mar_other_features_unchanged(self) -> None:
        """Test that MAR only affects the target feature."""
        X = pd.DataFrame(
            {
                "feature_a": np.random.randn(1000),
                "feature_b": np.random.rand(1000),
                "feature_c": np.random.randn(1000),
            }
        )

        rule = MARRuleConfig(
            target_feature="feature_a",
            depends_on="feature_b",
            rule_type="quantile_threshold",
            params={"base_prob": 0.05, "max_prob": 0.5, "quantile_threshold": 0.7},
        )
        X_missing, _ = inject_mar(X, rule, target_rate=0.2, seed=42)

        # Only target feature should have missing values
        assert X_missing["feature_a"].isna().any()
        assert not X_missing["feature_b"].isna().any()
        assert not X_missing["feature_c"].isna().any()


class TestMissingnessGrid:
    """Tests for the missingness grid simulation."""

    def test_simulate_missingness_grid_structure(self) -> None:
        """Test that grid returns correct structure."""
        X = pd.DataFrame(np.random.randn(1000, 5))

        mechanisms = ["MCAR", "MAR"]
        rates = [0.1, 0.2]
        seeds = [42, 123]
        mar_rules = {
            "MAR": MARRuleConfig(
                target_feature=X.columns[0],
                depends_on=X.columns[1],
                rule_type="quantile_threshold",
                params={"base_prob": 0.05, "max_prob": 0.5, "quantile_threshold": 0.7},
            )
        }

        results = simulate_missingness_grid(X, mechanisms, rates, seeds, mar_rules)

        # Check structure: results[mechanism][rate][seed] = (X_missing, realized)
        assert "MCAR" in results
        assert "MAR" in results
        for mech in mechanisms:
            for rate in rates:
                assert rate in results[mech]
                for seed in seeds:
                    assert seed in results[mech][rate]
                    X_miss, realized = results[mech][rate][seed]
                    assert isinstance(X_miss, pd.DataFrame)
                    assert isinstance(realized, dict)
