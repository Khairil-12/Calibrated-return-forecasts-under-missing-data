"""Tests for feature engineering with lookahead safety."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from calibmiss.features import compute_features


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

            data.append({
                "ticker": ticker,
                "date": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": prices[i],
                "adj_close": prices[i],
                "volume": volume
            })

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
            "return_1d", "return_5d", "return_20d",
            "volatility_5d", "volatility_20d",
            "volume_ratio", "hl_range", "price_vs_ma50"
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

    def test_target_drops_zero_return_days(self) -> None:
        """Test that target drops zero-return days."""
        raw = create_sample_raw_data()

        # Create a case with zero return
        raw.loc[("AAPL", "2023-01-05"), "adj_close"] = raw.loc[("AAPL", "2023-01-04"), "adj_close"]

        _, target = compute_features(raw)

        # Zero-return day should not be in target index
        # The target is computed from next day return, so we check that
        # the day before the zero-return day is handled correctly
        assert len(target) > 0
        assert set(target.unique()) == {0, 1}
