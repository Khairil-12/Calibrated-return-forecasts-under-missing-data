"""Download and validate the configured OHLCV universe."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibmiss.config import get_config  # noqa: E402
from calibmiss.data import download_ohlcv, validate_ohlcv  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    """Download configured tickers to the raw-data cache and validate them."""
    cfg = get_config(ROOT / "config")
    universe = cfg.universe
    print(
        f"Downloading {len(universe.tickers)} tickers "
        f"from {universe.start_date} to {universe.end_date}..."
    )
    data = download_ohlcv(
        universe.tickers,
        universe.start_date,
        universe.end_date,
        cache_dir=RAW_DIR,
    )
    report = validate_ohlcv(data)
    if not report.is_valid:
        raise ValueError("OHLCV validation failed: " + "; ".join(report.errors))

    print(
        f"Downloaded and validated {len(data)} rows for "
        f"{data.index.get_level_values('ticker').nunique()} tickers."
    )
    print("Native missing rates:")
    for column, rate in report.native_missing_rates.items():
        print(f"  {column}: {rate:.2%}")


if __name__ == "__main__":
    main()
