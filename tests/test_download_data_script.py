import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scripts import download_data  # noqa: E402


def sample_data() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [("AAA", pd.Timestamp("2024-01-02"))], names=["ticker", "date"]
    )
    return pd.DataFrame(
        {
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "adj_close": [1.5],
            "volume": [100],
        },
        index=index,
    )


def test_main_downloads_configured_universe_into_raw_cache(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    cfg = SimpleNamespace(
        universe=SimpleNamespace(
            tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01"
        )
    )
    calls = []
    monkeypatch.setattr(download_data, "get_config", lambda path: cfg)
    monkeypatch.setattr(download_data, "RAW_DIR", tmp_path)

    def fake_download(tickers, start, end, cache_dir):
        calls.append((tickers, start, end, cache_dir))
        return sample_data()

    monkeypatch.setattr(download_data, "download_ohlcv", fake_download)

    download_data.main()

    assert calls == [(["AAA"], "2024-01-01", "2024-02-01", tmp_path)]
    assert "1 rows" in capsys.readouterr().out


def test_main_rejects_invalid_download(tmp_path: Path, monkeypatch) -> None:
    cfg = SimpleNamespace(
        universe=SimpleNamespace(tickers=["AAA"], start_date="2024-01-01", end_date="2024-02-01")
    )
    monkeypatch.setattr(download_data, "get_config", lambda path: cfg)
    monkeypatch.setattr(download_data, "RAW_DIR", tmp_path)
    invalid_data = sample_data()
    invalid_data.loc[("AAA", pd.Timestamp("2024-01-02")), "close"] = -1
    monkeypatch.setattr(download_data, "download_ohlcv", lambda *args, **kwargs: invalid_data)

    with pytest.raises(ValueError, match="OHLCV validation failed"):
        download_data.main()
