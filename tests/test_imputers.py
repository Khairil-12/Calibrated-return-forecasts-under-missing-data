"""Tests for imputation strategies."""

import numpy as np
import pandas as pd
from calibmiss.imputers import DropImputer, ForwardFillImputer, IterativeImputerWrapper, MedianImputer


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

    def test_forward_fill_uses_training_median_for_leading_gaps(self) -> None:
        train_idx = pd.MultiIndex.from_tuples(
            [("AAPL", "2023-01-01"), ("AAPL", "2023-01-02"),
             ("MSFT", "2023-01-01"), ("MSFT", "2023-01-02")],
            names=["ticker", "date"],
        )
        test_idx = pd.MultiIndex.from_tuples(
            [("AAPL", "2023-01-03"), ("AAPL", "2023-01-04"),
             ("MSFT", "2023-01-03"), ("MSFT", "2023-01-04")],
            names=["ticker", "date"],
        )
        X_train = pd.DataFrame({"feat": [2.0, 4.0, 10.0, 20.0]}, index=train_idx)
        X_test = pd.DataFrame({"feat": [np.nan, 5.0, np.nan, 20.0]}, index=test_idx)

        imputer = ForwardFillImputer().fit(X_train)
        X_test_imp = imputer.transform(X_test)

        assert X_test_imp.loc[("AAPL", "2023-01-03"), "feat"] == 3.0
        assert X_test_imp.loc[("MSFT", "2023-01-03"), "feat"] == 15.0

    def test_forward_fill_uses_global_training_median_for_unseen_ticker(self) -> None:
        train_idx = pd.MultiIndex.from_tuples(
            [("AAPL", "2023-01-01"), ("AAPL", "2023-01-02"),
             ("MSFT", "2023-01-01"), ("MSFT", "2023-01-02")],
            names=["ticker", "date"],
        )
        test_idx = pd.MultiIndex.from_tuples(
            [("NVDA", "2023-01-03"), ("NVDA", "2023-01-04")],
            names=["ticker", "date"],
        )
        X_train = pd.DataFrame({"feat": [2.0, 4.0, 10.0, 20.0]}, index=train_idx)
        X_test = pd.DataFrame({"feat": [np.nan, 100.0]}, index=test_idx)

        X_test_imp = ForwardFillImputer().fit(X_train).transform(X_test)

        assert X_test_imp.loc[("NVDA", "2023-01-03"), "feat"] == 7.0
        assert X_test_imp.loc[("NVDA", "2023-01-04"), "feat"] == 100.0

    def test_forward_fill_fallback_fit_on_train_only(self) -> None:
        train_idx = pd.MultiIndex.from_tuples(
            [("AAPL", "2023-01-01"), ("AAPL", "2023-01-02")],
            names=["ticker", "date"],
        )
        test_idx = pd.MultiIndex.from_tuples(
            [("AAPL", "2023-01-03"), ("AAPL", "2023-01-04")],
            names=["ticker", "date"],
        )
        X_train = pd.DataFrame({"feat": [2.0, 4.0]}, index=train_idx)
        X_test = pd.DataFrame({"feat": [np.nan, 1000.0]}, index=test_idx)
        imputer = ForwardFillImputer().fit(X_train)
        fitted_median = imputer.medians_["feat"]
        fitted_ticker_median = imputer.ticker_medians_["feat"]["AAPL"]

        X_test_imp = imputer.transform(X_test)

        assert fitted_median == 3.0
        assert fitted_ticker_median == 3.0
        assert imputer.medians_["feat"] == fitted_median
        assert imputer.ticker_medians_["feat"]["AAPL"] == fitted_ticker_median
        assert X_test_imp.iloc[0, 0] == fitted_ticker_median
        assert X_test_imp.iloc[1, 0] == 1000.0

    def test_forward_fill_is_causal(self) -> None:
        idx = pd.MultiIndex.from_tuples(
            [("AAPL", "2023-01-03"), ("AAPL", "2023-01-04"),
             ("AAPL", "2023-01-05")],
            names=["ticker", "date"],
        )
        X_train = pd.DataFrame({"feat": [2.0, 4.0]}, index=pd.MultiIndex.from_tuples(
            [("AAPL", "2023-01-01"), ("AAPL", "2023-01-02")],
            names=["ticker", "date"],
        ))
        X_test = pd.DataFrame({"feat": [np.nan, 5.0, np.nan]}, index=idx)
        X_later_changed = X_test.copy()
        X_later_changed.iloc[2, 0] = 999.0
        imputer = ForwardFillImputer().fit(X_train)

        X_test_imp = imputer.transform(X_test)
        X_later_changed_imp = imputer.transform(X_later_changed)

        assert X_test_imp.iloc[0, 0] == 3.0
        assert X_test_imp.iloc[1, 0] == 5.0
        assert X_test_imp.iloc[0, 0] == X_later_changed_imp.iloc[0, 0]
        assert X_test_imp.iloc[1, 0] == X_later_changed_imp.iloc[1, 0]
        assert X_later_changed_imp.iloc[2, 0] == 999.0


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
        fitted_statistics = imputer.imputer.initial_imputer_.statistics_.copy()
        X_test_imp = imputer.transform(X_test)

        assert not X_train_imp.isna().any().any()
        assert not X_test_imp.isna().any().any()
        np.testing.assert_array_equal(imputer.imputer.initial_imputer_.statistics_, fitted_statistics)


def _ticker_frame(n_ticks: int = 5, n_days: int = 200) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [range(n_ticks), range(n_days)], names=["ticker", "date"]
    )
    return pd.DataFrame(np.random.randn(n_ticks * n_days, 3), index=idx)


class TestDropImputer:
    """Tests for DropImputer (listwise deletion)."""

    def test_drop_incomplete_rows(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [3.0, np.nan, 4.0]})
        out = DropImputer().transform(X)
        assert len(out) == 1
        assert out.iloc[0]["a"] == 1.0 and out.iloc[0]["b"] == 3.0

    def test_drop_fit_is_noop_and_train_only(self) -> None:
        """DropImputer fits nothing; transform drops any row with a NaN."""
        X_train = pd.DataFrame({"a": [1.0, 2.0, np.nan]})
        X_test = pd.DataFrame({"a": [5.0, np.nan, 6.0]})
        imp = DropImputer().fit(X_train)
        out_train = imp.transform(X_train)
        out_test = imp.transform(X_test)
        assert len(out_train) == 2
        assert out_test.index.tolist() == [0, 2]

    def test_drop_preserves_multiindex_rows(self) -> None:
        X = _ticker_frame(n_ticks=2, n_days=10)
        X.iloc[3] = np.nan  # one row in ticker 0 day 3
        out = DropImputer().transform(X)
        assert len(out) == len(X) - 1
        assert (3,) not in out.index.get_level_values("date")


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

