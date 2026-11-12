"""Compute and persist CalibMiss feature data from the raw cache."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibmiss.config import get_config  # noqa: E402
from calibmiss.data import validate_ohlcv  # noqa: E402
from calibmiss.features import compute_features  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _cache_pattern(tickers: list[str], start: str, end: str) -> str:
    ticker_key = "_".join(sorted(tickers))
    return f"*_{ticker_key}_{start}_{end}_sha256_*.parquet"


def _raw_cache_path(tickers: list[str], start: str, end: str) -> Path:
    matches = sorted(RAW_DIR.glob(_cache_pattern(tickers, start, end)))
    if not matches:
        raise FileNotFoundError(
            f"No raw OHLCV cache found for {len(tickers)} tickers from {start} to {end}"
        )
    return matches[-1]


def main() -> None:
    """Compute features and target from the configured raw OHLCV cache."""
    cfg = get_config(ROOT / "config")
    universe = cfg.universe
    raw_path = _raw_cache_path(universe.tickers, universe.start_date, universe.end_date)
    raw = pd.read_parquet(raw_path)
    report = validate_ohlcv(raw)
    if not report.is_valid:
        raise ValueError("Raw OHLCV validation failed: " + "; ".join(report.errors))

    features, target = compute_features(raw)
    valid_features = ~features.isna().any(axis=1)
    features = features[valid_features]
    target = target[valid_features]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_DIR / "features.parquet")
    target.rename("target").to_frame().to_parquet(PROCESSED_DIR / "target.parquet")
    print(f"Computed {len(features)} feature rows from {raw_path.name}.")
    print(f"Saved {PROCESSED_DIR / 'features.parquet'}")
    print(f"Saved {PROCESSED_DIR / 'target.parquet'}")


if __name__ == "__main__":
    main()
