"""Feature engineering for CalibMiss with lookahead safety."""

import numpy as np
import pandas as pd


def compute_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Compute features and target from raw OHLCV data.

    All features use only data up to day t (no lookahead).
    Target is binary: 1 if next-day adj_close > current adj_close, 0 otherwise.
    Zero-return days are dropped.

    Args:
        raw: DataFrame with MultiIndex (ticker, date) and columns:
            open, high, low, close, adj_close, volume

    Returns:
        Tuple of (features DataFrame, target Series) with same MultiIndex
    """
    # Work with a copy to avoid modifying original
    df = raw.copy()

    # Ensure we have the right columns
    required_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Group by ticker for per-ticker computations
    grouped = df.groupby(level="ticker")

    # 1. Log returns
    df["return_1d"] = grouped["adj_close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df["return_5d"] = grouped["adj_close"].transform(
        lambda x: np.log(x / x.shift(5))
    )
    df["return_20d"] = grouped["adj_close"].transform(
        lambda x: np.log(x / x.shift(20))
    )

    # 2. Rolling volatility (annualized)
    df["volatility_5d"] = grouped["return_1d"].transform(
        lambda x: x.rolling(5).std() * np.sqrt(252)
    )
    df["volatility_20d"] = grouped["return_1d"].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    )

    # 3. Volume ratio
    df["volume_ratio"] = grouped["volume"].transform(
        lambda x: x / x.rolling(20).mean()
    )

    # 4. HL range
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # 5. Price vs MA50
    df["price_vs_ma50"] = grouped["adj_close"].transform(
        lambda x: x / x.rolling(50).mean()
    )

    # Select feature columns
    feature_cols = [
        "return_1d", "return_5d", "return_20d",
        "volatility_5d", "volatility_20d",
        "volume_ratio", "hl_range", "price_vs_ma50"
    ]
    features = df[feature_cols].copy()

    # 6. Target: next-day return direction (binary)
    # 1 if next day's adj_close > current day's adj_close, 0 otherwise
    # Use shift(-1) to get next day's price
    next_day_close = grouped["adj_close"].transform(lambda x: x.shift(-1))
    target = (next_day_close > df["adj_close"]).astype(int)
    target.name = "target"

    # Drop rows where next day's price equals current (zero return)
    # Also drop rows where next day's price is NaN (last day of each ticker)
    zero_return_mask = next_day_close == df["adj_close"]
    nan_mask = next_day_close.isna()
    valid_mask = ~(zero_return_mask | nan_mask)

    features = features[valid_mask]
    target = target[valid_mask]

    return features, target
