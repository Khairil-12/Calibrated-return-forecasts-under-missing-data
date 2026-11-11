from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import compute_features


def raw_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=55, freq="D")
    index = pd.MultiIndex.from_product([["AAA"], dates], names=["ticker", "date"])
    close = pd.Series(range(1, 56), index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close * 0.9,
            "close": close,
            "adj_close": close,
            "volume": 1000.0,
        },
        index=index,
    )


def test_main_computes_and_saves_processed_features(tmp_path: Path, monkeypatch) -> None:
    cfg = SimpleNamespace(
        universe=SimpleNamespace(
            tickers=["AAA"], start_date="2024-01-01", end_date="2024-03-01"
        )
    )
    raw_path = tmp_path / "123_AAA_2024-01-01_2024-03-01_sha256_test.parquet"
    raw_path.touch()
    monkeypatch.setattr(compute_features, "ROOT", tmp_path)
    monkeypatch.setattr(compute_features, "RAW_DIR", tmp_path)
    monkeypatch.setattr(compute_features, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(compute_features, "get_config", lambda path: cfg)
    real_read_parquet = pd.read_parquet
    monkeypatch.setattr(
        compute_features.pd,
        "read_parquet",
        lambda path: raw_data() if Path(path) == raw_path else real_read_parquet(path),
    )

    compute_features.main()

    features = pd.read_parquet(tmp_path / "processed" / "features.parquet")
    target = pd.read_parquet(tmp_path / "processed" / "target.parquet")["target"]
    assert list(features.columns) == [
        "return_1d", "return_5d", "return_20d", "volatility_5d",
        "volatility_20d", "volume_ratio", "hl_range", "price_vs_ma50",
    ]
    assert len(features) == len(target) > 0


def test_main_rejects_missing_raw_cache(tmp_path: Path, monkeypatch) -> None:
    cfg = SimpleNamespace(
        universe=SimpleNamespace(
            tickers=["AAA"], start_date="2024-01-01", end_date="2024-03-01"
        )
    )
    monkeypatch.setattr(compute_features, "ROOT", tmp_path)
    monkeypatch.setattr(compute_features, "RAW_DIR", tmp_path)
    monkeypatch.setattr(compute_features, "get_config", lambda path: cfg)

    with pytest.raises(FileNotFoundError, match="No raw OHLCV cache found"):
        compute_features.main()
