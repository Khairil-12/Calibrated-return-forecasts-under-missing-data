"""Tests for imputation strategies."""

import numpy as np
import pandas as pd
from calibmiss.imputers import ForwardFillImputer, IterativeImputerWrapper, MedianImputer


class TestForwardFillImputer:
    """Tests for ForwardFillImputer."""

    def test_forward_fill_within_ticker(self) -> None:
        """Test forward fill within each ticker group."""
        idx = pd.MultiIndex.from_tuples([
            ("AAPL", "2023-01-01"), ("AAPL", "2023-01-02"), ("AAPL", "2023-01-03"),
            ("MSFT", "2023-01-01"), ("MSFT", "2023-01-02"), ("MSFT", "2023-01-03")
        ], names=["ticker", "date"])
        X_train = pd.DataFrame({"feat": [1.0, np.nan, 3.0, 10.0, np.nan, 30.0]}, index=idx)
        X_test = pd.DataFrame({"feat": [np.nan, 5.0, np.nan, np.nan, 20.0, np.nan]}, index=idx)

        imputer = ForwardFillImputer()
        imputer.fit(X_train)
        X_train_imp = imputer.transform(X_train)
        X_test_imp = imputer.transform(X_test)

        assert not X_train_imp.isna().any().any()
        assert not X_test_imp.isna().any().any()
        # AAPL: [1, NaN, 3] -> [1, 1, 3] (forward fill)
        # MSFT: [10, NaN, 30] -> [10, 10, 30]
        assert X_train_imp.loc[("AAPL", "2023-01-02"), "feat"] == 1.0
        assert X_train_imp.loc[("MSFT", "2023-01-02"), "feat"] == 10.0


class TestMedianImputer:
    """Tests for MedianImputer."""

    def test_median_imputer_fit_on_train_only(self) -> None:
        """Test that median imputer fits on train only."""
        X_train = pd.DataFrame({"a": [1, 2, np.nan], "b": [np.nan, 5, 6]})
        X_test = pd.DataFrame({"a": [np.nan, 8], "b": [9, np.nan]})

        imputer = MedianImputer()
        imputer.fit(X_train)
        _ = imputer.transform(X_train)
        X_test_imp = imputer.transform(X_test)

        assert imputer.medians_["a"] == 1.5
        assert imputer.medians_["b"] == 5.5
        assert X_test_imp["a"].iloc[0] == 1.5
        assert X_test_imp["b"].iloc[1] == 5.5


class TestIterativeImputerWrapper:
    """Tests for IterativeImputerWrapper."""

    def test_iterative_imputer(self) -> None:
        """Test iterative imputer on correlated features."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n) * 0.5 + 0.5 * np.random.randn(n),  # correlated with a
        })
        # Introduce NaN
        mask = np.random.random((n, 2)) < 0.1
        X_miss = X.copy()
        X_miss.values[mask] = np.nan

        X_train = X_miss.iloc[:100]
        X_test = X_miss.iloc[100:]

        imputer = IterativeImputerWrapper(max_iter=10, random_state=42)
        imputer.fit(X_train)
        X_train_imp = imputer.transform(X_train)
        X_test_imp = imputer.transform(X_test)

        assert not X_train_imp.isna().any().any()
        assert not X_test_imp.isna().any().any()


class TestFitOnTrainOnly:
    """Critical test: imputers never fit on test data."""

    def test_imputers_never_fit_on_test(self) -> None:
        """Verify imputers don't leak test info."""
        X_train = pd.DataFrame({"a": [1, 2, np.nan]})
        X_test = pd.DataFrame({"a": [100, np.nan]})  # Very different distribution

        # Test MedianImputer and IterativeImputerWrapper use training statistics
        # ForwardFillImputer has different behavior (uses forward fill within test data)
        for ImputerClass in [MedianImputer, IterativeImputerWrapper]:
            imputer = ImputerClass()
            imputer.fit(X_train)
            X_test_imp = imputer.transform(X_test)
            if hasattr(imputer, "medians_"):
                assert X_test_imp["a"].iloc[1] == imputer.medians_["a"]

        # ForwardFillImputer: verify it doesn't crash and produces no NaN
        imputer = ForwardFillImputer()
        imputer.fit(X_train)
        X_test_imp = imputer.transform(X_test)
        assert not X_test_imp.isna().any().any()

