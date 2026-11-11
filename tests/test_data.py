"""Tests for data module."""

from pathlib import Path

import pandas as pd
import pytest
from calibmiss.data import download_ohlcv, validate_ohlcv


class TestDownloadOHLCV:
    """Tests for download_ohlcv function."""

    def test_download_returns_correct_structure(self, monkeypatch) -> None:
        """Test that download returns DataFrame with correct MultiIndex and columns."""
        import importlib

        data_module = importlib.import_module("calibmiss.data")
        monkeypatch.setattr(
            data_module.yf,
            "download",
            lambda *args, **kwargs: pd.concat(
                {
                    "AAPL": pd.DataFrame(
                        {
                            "Open": [1.0],
                            "High": [2.0],
                            "Low": [0.5],
                            "Close": [1.5],
                            "Adj Close": [1.5],
                            "Volume": [100],
                        },
                        index=pd.DatetimeIndex(["2023-01-03"], name="Date"),
                    ),
                    "MSFT": pd.DataFrame(
                        {
                            "Open": [2.0],
                            "High": [3.0],
                            "Low": [1.5],
                            "Close": [2.5],
                            "Adj Close": [2.5],
                            "Volume": [200],
                        },
                        index=pd.DatetimeIndex(["2023-01-03"], name="Date"),
                    ),
                },
                axis=1,
            ),
        )
        df = download_ohlcv(["AAPL", "MSFT"], "2023-01-01", "2023-12-31")

        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["ticker", "date"]
        assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
        assert len(df) > 0

    def test_cache_creates_timestamped_parquet_with_checksum(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Test that caching creates parquet file with timestamp and SHA256 in filename."""
        import importlib

        data_module = importlib.import_module("calibmiss.data")
        monkeypatch.setattr(
            data_module.yf,
            "download",
            lambda *args, **kwargs: pd.DataFrame(
                {
                    "Open": [1.0],
                    "High": [2.0],
                    "Low": [0.5],
                    "Close": [1.5],
                    "Adj Close": [1.5],
                    "Volume": [100],
                },
                index=pd.DatetimeIndex(["2023-01-03"], name="Date"),
            ),
        )
        df = download_ohlcv(["AAPL"], "2023-01-01", "2023-12-31", cache_dir=tmp_path)

        files = list(tmp_path.glob("*.parquet"))
        assert len(files) == 1
        assert "_sha256_" in files[0].name

        df2 = download_ohlcv(["AAPL"], "2023-01-01", "2023-12-31", cache_dir=tmp_path)
        pd.testing.assert_frame_equal(df, df2)


class TestValidateOHLCV:
    """Tests for validate_ohlcv function."""

    def test_validate_data_catches_issues(self) -> None:
        """Test that validation catches duplicates, negative prices, negative volume."""
        idx = pd.MultiIndex.from_tuples(
            [("AAPL", "2023-01-01"), ("AAPL", "2023-01-02")], names=["ticker", "date"]
        )
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "adj_close": [101.0, 102.0],
                "volume": [1000, 2000],
            },
            index=idx,
        )

        report = validate_ohlcv(df)
        assert report.is_valid
        assert report.native_missing_rates is not None

        df_dup = pd.concat([df, df.iloc[[0]]])
        report_dup = validate_ohlcv(df_dup)
        assert not report_dup.is_valid
        assert any("duplicate" in e.lower() for e in report_dup.errors)

        df_neg = df.copy()
        df_neg.loc[("AAPL", "2023-01-01"), "close"] = -1.0
        report_neg = validate_ohlcv(df_neg)
        assert not report_neg.is_valid
        assert any("price" in e.lower() or "negative" in e.lower() for e in report_neg.errors)

        df_negvol = df.copy()
        df_negvol.loc[("AAPL", "2023-01-01"), "volume"] = -100
        report_negvol = validate_ohlcv(df_negvol)
        assert not report_negvol.is_valid
        assert any("volume" in e.lower() or "negative" in e.lower() for e in report_negvol.errors)

    def test_empty_ohlcv_is_invalid(self) -> None:
        idx = pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"])
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "adj_close", "volume"], index=idx
        )

        report = validate_ohlcv(df)

        assert not report.is_valid
        assert any("empty" in error.lower() for error in report.errors)

    def test_empty_download_is_not_cached(self, tmp_path: Path, monkeypatch) -> None:
        import importlib

        data_module = importlib.import_module("calibmiss.data")
        monkeypatch.setattr(data_module.yf, "download", lambda *args, **kwargs: pd.DataFrame())

        with pytest.raises(ValueError, match="OHLCV validation failed: OHLCV data is empty"):
            download_ohlcv(["AAPL"], "2023-01-01", "2023-01-31", tmp_path)

        assert not list(tmp_path.glob("*.parquet"))

    def test_invalid_cache_payload_is_redownloaded(self, tmp_path: Path, monkeypatch) -> None:
        import importlib

        data_module = importlib.import_module("calibmiss.data")
        index = pd.MultiIndex.from_tuples(
            [("AAPL", pd.Timestamp("2023-01-03"))], names=["ticker", "date"]
        )
        invalid = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [-1.0],
                "adj_close": [1.5],
                "volume": [100],
            },
            index=index,
        )
        cache_key = data_module._get_cache_key(["AAPL"], "2023-01-01", "2023-01-31")
        checksum = data_module._compute_sha256(invalid)
        invalid.to_parquet(tmp_path / f"123_{cache_key}_sha256_{checksum}.parquet")
        calls = 0

        def fake_download(*args, **kwargs):
            nonlocal calls
            calls += 1
            return pd.DataFrame(
                {
                    "Open": [1.0],
                    "High": [2.0],
                    "Low": [0.5],
                    "Close": [1.5],
                    "Adj Close": [1.5],
                    "Volume": [100],
                },
                index=pd.DatetimeIndex(["2023-01-03"], name="Date"),
            )

        monkeypatch.setattr(data_module.yf, "download", fake_download)

        result = data_module.download_ohlcv(["AAPL"], "2023-01-01", "2023-01-31", tmp_path)

        assert calls == 1
        assert result.iloc[0]["close"] == 1.5

    def test_unreadable_cache_file_is_redownloaded(self, tmp_path: Path, monkeypatch) -> None:
        import importlib

        data_module = importlib.import_module("calibmiss.data")
        cache_key = data_module._get_cache_key(["AAPL"], "2023-01-01", "2023-01-31")
        (tmp_path / f"123_{cache_key}_sha256_deadbeef.parquet").write_bytes(b"invalid parquet")
        monkeypatch.setattr(
            data_module.yf,
            "download",
            lambda *args, **kwargs: pd.DataFrame(
                {
                    "Open": [1.0],
                    "High": [2.0],
                    "Low": [0.5],
                    "Close": [1.5],
                    "Adj Close": [1.5],
                    "Volume": [100],
                },
                index=pd.DatetimeIndex(["2023-01-03"], name="Date"),
            ),
        )

        result = data_module.download_ohlcv(["AAPL"], "2023-01-01", "2023-01-31", tmp_path)

        assert result.iloc[0]["close"] == 1.5

    def test_corrupt_cache_payload_is_redownloaded(self, tmp_path: Path, monkeypatch) -> None:
        import importlib

        data_module = importlib.import_module("calibmiss.data")
        idx = pd.MultiIndex.from_tuples(
            [("AAPL", pd.Timestamp("2023-01-03"))], names=["ticker", "date"]
        )
        downloaded = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "adj_close": [1.5],
                "volume": [100],
            },
            index=idx,
        )
        calls = 0

        def fake_download(*args, **kwargs):
            nonlocal calls
            calls += 1
            return pd.DataFrame(
                {
                    "Open": [1.0],
                    "High": [2.0],
                    "Low": [0.5],
                    "Close": [1.5],
                    "Adj Close": [1.5],
                    "Volume": [100],
                },
                index=pd.DatetimeIndex(["2023-01-03"], name="Date"),
            )

        monkeypatch.setattr(data_module.yf, "download", fake_download)
        data_module.download_ohlcv(["AAPL"], "2023-01-01", "2023-01-31", tmp_path)
        cached = next(tmp_path.glob("*.parquet"))
        corrupted = pd.read_parquet(cached)
        corrupted.loc[("AAPL", pd.Timestamp("2023-01-03")), "close"] = 999.0
        corrupted.to_parquet(cached)

        reloaded = data_module.download_ohlcv(["AAPL"], "2023-01-01", "2023-01-31", tmp_path)

        assert calls == 2
        pd.testing.assert_frame_equal(reloaded, downloaded)


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_validation_report_structure(self) -> None:
        """Test ValidationReport has correct fields."""
        idx = pd.MultiIndex.from_tuples([("AAPL", "2023-01-01")], names=["ticker", "date"])
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "adj_close": [101.0],
                "volume": [1000],
            },
            index=idx,
        )

        report = validate_ohlcv(df)

        assert hasattr(report, "is_valid")
        assert hasattr(report, "errors")
        assert hasattr(report, "native_missing_rates")
        assert isinstance(report.is_valid, bool)
        assert isinstance(report.errors, list)
        assert isinstance(report.native_missing_rates, dict)
