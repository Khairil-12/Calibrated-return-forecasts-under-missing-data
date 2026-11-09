"""Data download, caching, and validation for CalibMiss."""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass
class ValidationReport:
    """Report from data validation."""

    is_valid: bool
    errors: list[str]
    native_missing_rates: dict[str, float]


def _get_cache_key(tickers: list[str], start: str, end: str) -> str:
    """Generate a cache key from tickers and date range."""
    tickers_str = "_".join(sorted(tickers))
    return f"{tickers_str}_{start}_{end}"


def _compute_sha256(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of DataFrame serialized to parquet."""
    import io
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/raw"
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance with caching.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        cache_dir: Directory for cached parquet files

    Returns:
        DataFrame with MultiIndex (ticker, date) and columns:
        open, high, low, close, adj_close, volume
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _get_cache_key(tickers, start, end)

    # Check for existing cached file
    existing_files = list(cache_dir.glob(f"*_{cache_key}_sha256_*.parquet"))
    if existing_files:
        # Load from cache
        df = pd.read_parquet(existing_files[0])
        return df

    # Download from Yahoo Finance
    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers
        frames = []
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                ticker_data = data[ticker].copy()
                ticker_data.index = pd.to_datetime(ticker_data.index)
                ticker_data.index.name = "date"
                ticker_data["ticker"] = ticker
                frames.append(ticker_data)

        if not frames:
            raise ValueError(f"No data found for tickers: {tickers}")

        df = pd.concat(frames)
        df = df.reset_index().set_index(["ticker", "date"])
    else:
        # Single ticker
        data.index = pd.to_datetime(data.index)
        data.index.name = "date"
        data["ticker"] = tickers[0]
        df = data.reset_index().set_index(["ticker", "date"])

    # Standardize column names
    column_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }
    df = df.rename(columns=column_mapping)

    # Select and order columns
    expected_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df = df[expected_cols]

    # Sort index
    df = df.sort_index()

    # Save to cache with timestamp and SHA256
    timestamp = int(time.time())
    sha256 = _compute_sha256(df)
    cache_filename = f"{timestamp}_{cache_key}_sha256_{sha256}.parquet"
    cache_path = cache_dir / cache_filename
    df.to_parquet(cache_path)

    return df


def validate_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """Validate OHLCV data quality.

    Args:
        df: DataFrame with MultiIndex (ticker, date) and OHLCV columns

    Returns:
        ValidationReport with validation results
    """
    errors = []

    # Check for duplicate (ticker, date) rows
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        errors.append(f"Found {dup_count} duplicate (ticker, date) rows")

    # Check prices are positive
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        if col in df.columns:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                errors.append(f"Column '{col}' has {neg_count} negative price values")

    # Check volume is non-negative
    if "volume" in df.columns:
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol > 0:
            errors.append(f"Column 'volume' has {neg_vol} negative volume values")

    # Compute native missing rates per column
    missing_rates = {}
    for col in df.columns:
        missing_rates[col] = float(df[col].isna().mean())

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        errors=errors,
        native_missing_rates=missing_rates
    )
