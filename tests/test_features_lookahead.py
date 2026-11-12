"""Tests for feature engineering with lookahead safety."""

import numpy as np
import pandas as pd
from calibmiss.features import compute_features, compute_volatility_target


def create_sample_raw_data() -> pd.DataFrame:
    """Create sample raw OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = ["AAPL", "MSFT"]

    data = []
    for ticker in tickers:
        np.random.seed(hash(ticker) % 2**32)
        base_price = 150.0 if ticker == "AAPL" else 300.0
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        for i, date in enumerate(dates):
            high = prices[i] * (1 + abs(np.random.normal(0, 0.01)))
            low = prices[i] * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i] * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.lognormal(13, 0.5))

            data.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": prices[i],
                    "adj_close": prices[i],
                    "volume": volume,
                }
            )

    df = pd.DataFrame(data)
    df = df.set_index(["ticker", "date"])
    return df


class TestComputeFeatures:
    """Tests for compute_features function."""

    def test_compute_features_returns_expected_columns(self) -> None:
        """Test that compute_features returns expected columns and target."""
        raw = create_sample_raw_data()
        features, target = compute_features(raw)

        expected_cols = [
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_5d",
            "volatility_20d",
            "volume_ratio",
            "hl_range",
            "price_vs_ma50",
        ]
        assert list(features.columns) == expected_cols
        assert target.name == "target"
        assert set(target.unique()) == {0, 1}

    def test_no_lookahead_bias(self) -> None:
        """Modify future data and verify current features don't change."""
        raw = create_sample_raw_data()
        features, _ = compute_features(raw)

        # Create corrupted raw data where FUTURE values are modified
        # Multiply all prices from day 50 onwards by 2 for each ticker
        raw_corrupted = raw.copy()
        for ticker in ["AAPL", "MSFT"]:
            ticker_data = raw_corrupted.loc[ticker]
            future_dates = ticker_data.index[ticker_data.index >= "2023-03-15"]  # ~day 50
            for date in future_dates:
                raw_corrupted.loc[(ticker, date), "adj_close"] *= 2
                raw_corrupted.loc[(ticker, date), "close"] *= 2
                raw_corrupted.loc[(ticker, date), "open"] *= 2
                raw_corrupted.loc[(ticker, date), "high"] *= 2
                raw_corrupted.loc[(ticker, date), "low"] *= 2

        features_corrupted, _ = compute_features(raw_corrupted)

        # Features for dates BEFORE the modification should be IDENTICAL
        # (no lookahead bias - current features don't depend on future data)
        early_dates = features.index.get_level_values("date") < "2023-03-15"
        features_early = features[early_dates]
        features_corrupted_early = features_corrupted[early_dates]

        # Align indices (they should be the same)
        pd.testing.assert_frame_equal(features_early, features_corrupted_early)

    def test_shuffled_input_matches_chronological_output(self) -> None:
        raw = create_sample_raw_data()
        expected_features, expected_target = compute_features(raw)

        shuffled_features, shuffled_target = compute_features(raw.sample(frac=1, random_state=7))

        pd.testing.assert_frame_equal(shuffled_features, expected_features)
        pd.testing.assert_series_equal(shuffled_target, expected_target)

    def test_target_drops_zero_return_days(self) -> None:
        raw = create_sample_raw_data()
        raw.loc[("AAPL", "2023-01-05"), "adj_close"] = raw.loc[("AAPL", "2023-01-04"), "adj_close"]

        _, target = compute_features(raw)

        assert ("AAPL", pd.Timestamp("2023-01-04")) not in target.index
        assert ("AAPL", pd.Timestamp("2023-01-05")) in target.index


class TestVolatilityTarget:
    """Tests for compute_volatility_target."""

    def test_value_matches_hand_computed_case(self) -> None:
        dates = pd.date_range("2023-01-02", periods=60, freq="B")
        index = pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"])
        rng = np.random.default_rng(3)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 60))), index=index)
        raw = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "adj_close": prices, "volume": 1000.0},
            index=index,
        )

        target = compute_volatility_target(raw, horizon=5)
        assert len(target) > 0
        # Hand-computed std of the 5 log returns after a known kept row (day 50):
        # returns at days 51..55 = log(p[52..56] / p[51..55])? No: return at day k
        # is log(p[k]/p[k-1]), so the forward window of day 50 covers returns
        # at days 51..55 = log(p[51..55]/p[50..54]).
        probe = ("AAA", dates[50])
        rets = np.log(prices.values[51:56] / prices.values[50:55])
        expected = rets.std(ddof=1) * np.sqrt(252)
        assert abs(target.loc[probe] - expected) < 1e-10

    def test_lookahead_window_boundaries(self) -> None:
        dates = pd.date_range("2023-01-02", periods=60, freq="B")
        index = pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"])
        prices = pd.Series(100.0 + np.arange(60), index=index)
        raw = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "adj_close": prices, "volume": 1000.0},
            index=index,
        )
        probe = ("AAA", dates[50])

        # Mutate prices AFTER the forward window (day 56 onward): target at day 50 must not change.
        raw_mut = raw.copy()
        raw_mut.iloc[56:] *= 2
        assert compute_volatility_target(raw_mut, horizon=5).loc[probe] == compute_volatility_target(raw, horizon=5).loc[probe]

        # Mutate prices INSIDE the forward window (day 53): target at day 50 must change.
        raw_mut2 = raw.copy()
        raw_mut2.iloc[53] *= 2
        assert compute_volatility_target(raw_mut2, horizon=5).loc[probe] != compute_volatility_target(raw, horizon=5).loc[probe]

    def test_index_subset_of_features_and_no_nan_rows(self) -> None:
        raw = create_sample_raw_data()
        features, _ = compute_features(raw)
        target = compute_volatility_target(raw, horizon=5)

        assert target.index.isin(features.index).all()
        aligned = features.reindex(target.index)
        assert not aligned.isna().any(axis=None)
        # Last `horizon` rows per ticker cannot have a full forward window.
        counts = target.groupby(level="ticker").size()
        assert (counts < 100).all()

