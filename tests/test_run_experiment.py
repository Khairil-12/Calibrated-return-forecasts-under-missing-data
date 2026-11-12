"""Tests for the experiment grid runner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from calibmiss.config import get_config
from calibmiss.run_experiment import (
    main,
    map_date_splits_to_rows,
    run_grid,
    summarize_results,
    write_run_metadata,
)
from calibmiss.splits import SplitIndices


def make_split(dates: pd.DatetimeIndex) -> SplitIndices:
    return SplitIndices(
        train_idx=np.array([0, 1]),
        calibrate_idx=np.array([2]),
        test_idx=np.array([3]),
        train_start=dates[0],
        train_end=dates[1],
        calibrate_start=dates[2],
        calibrate_end=dates[2],
        test_start=dates[3],
        test_end=dates[3],
    )


def tiny_config():
    cfg = get_config(Path(__file__).resolve().parents[1] / "config").model_copy(deep=True)
    cfg.experiment.models = cfg.experiment.models[:1]
    cfg.experiment.imputers = cfg.experiment.imputers[:1]
    cfg.experiment.calibrators = cfg.experiment.calibrators[:1]
    cfg.experiment.missingness.mechanisms = ["MCAR"]
    cfg.experiment.missingness.rates = [0.0]
    cfg.experiment.missingness.seeds = [0]
    return cfg


def make_data() -> tuple[pd.DataFrame, pd.Series, list[SplitIndices]]:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    index = pd.MultiIndex.from_product([["AAA", "BBB"], dates], names=["ticker", "date"])
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(8, 8)),
        index=index,
        columns=[
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_5d",
            "volatility_20d",
            "volume_ratio",
            "hl_range",
            "price_vs_ma50",
        ],
    )
    y = pd.Series([0, 1, 0, 1, 1, 0, 1, 0], index=index, name="target")
    return X, y, [make_split(dates)]


def test_date_splits_map_each_date_to_every_ticker_row() -> None:
    X, _, splits = make_data()
    mapped = map_date_splits_to_rows(X, splits)[0]

    assert mapped.train_idx.tolist() == [0, 1, 4, 5]
    assert mapped.calibrate_idx.tolist() == [2, 6]
    assert mapped.test_idx.tolist() == [3, 7]
    assert (
        X.index.get_level_values("date")[mapped.train_idx].max()
        < X.index.get_level_values("date")[mapped.calibrate_idx].min()
    )


def test_grid_writes_metrics_baselines_summary_and_metadata(tmp_path: Path, monkeypatch) -> None:
    X, y, splits = make_data()
    config = tiny_config()
    seen: list[dict] = []

    def fake_run_single(X, y, train_idx, cal_idx, test_idx, config, **kwargs):
        seen.append({"config": config, **kwargs})
        labels = y.iloc[test_idx].to_numpy()
        return {
            "brier": float(np.mean((labels - 0.5) ** 2)),
            "ece": 0.1,
            "roc_auc": 0.5,
            "reliability": {
                "bin_edges": [0.0, 1.0],
                "bin_accuracies": [0.5],
                "bin_confidences": [0.5],
                "bin_counts": [len(labels)],
            },
            **config,
        }

    monkeypatch.setattr("calibmiss.run_experiment.run_single_config", fake_run_single)
    out = tmp_path / "metrics.csv"
    frame = run_grid(X, y, splits, config, n_jobs=1, output_path=out)
    loaded = pd.read_csv(out)

    required = {
        "fold",
        "seed",
        "model",
        "imputer",
        "calibrator",
        "mechanism",
        "rate",
        "brier",
        "ece",
        "roc_auc",
        "reliability",
        "row_type",
        "baseline",
    }
    assert required <= set(loaded.columns)
    assert len(frame[frame.row_type == "experiment"]) == 1
    assert set(frame.loc[frame.row_type == "baseline", "baseline"]) == {
        "complete_data_oracle",
        "naive_training_base_rate",
    }
    assert json.loads(loaded.loc[loaded.row_type == "experiment", "reliability"].iloc[0])[
        "bin_counts"
    ] == [1]
    assert seen[0]["config"]["seed"] == 0
    assert all(call.get("experiment_config") is config for call in seen)
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "statistical_tests.csv").exists()


def test_volatility_threshold_from_train_fold_only() -> None:
    X, y_raw, splits = make_data()
    config = tiny_config()
    # Volatility-style continuous target: 1.0, 2.0, 3.0, ... 8.0.
    y_vol = pd.Series(np.arange(1, 9, dtype=float), index=y_raw.index, name="target")
    config.experiment.models = []
    frame = run_grid(
        X, y_vol, splits, config, n_jobs=1, output_path=None, volatility_target=True
    )
    naive = frame.loc[frame.baseline == "naive_training_base_rate"].iloc[0]

    # Train rows 0,1 have raw values [1, 2], median 1.5 -> train binary [0, 1],
    # prevalence 0.5. Test rows 3,7 have raw values 4 and 8 -> binarized [1, 1].
    # Naive predicts prevalence 0.5 for both -> Brier 0.25. A threshold computed
    # on ALL data (median 4.5) would binarize train to [0, 0] (prevalence 0.0)
    # and give Brier 0.5 — so 0.25 proves the threshold came from train only.
    assert naive.brier == pytest.approx(0.25)


def test_persistence_baseline_reports_metrics() -> None:
    X, y_raw, splits = make_data()
    config = tiny_config()
    y_vol = pd.Series(np.arange(1, 9, dtype=float), index=y_raw.index, name="target")
    X.iloc[[3, 7], X.columns.get_loc("volatility_5d")] = 0.9
    config.experiment.models = []
    frame = run_grid(
        X, y_vol, splits, config, n_jobs=1, output_path=None, volatility_target=True
    )
    persistence = frame.loc[frame.baseline == "volatility_persistence"].iloc[0]

    assert {"brier", "ece", "roc_auc"} <= set(persistence.index)
    # Persistence prob = 0.9 at both test rows; binarized labels [1, 1].
    assert persistence.brier == pytest.approx(0.01)


def test_naive_baseline_uses_training_prevalence() -> None:
    X, y, splits = make_data()
    config = tiny_config()
    config.experiment.models = []
    frame = run_grid(X, y, splits, config, n_jobs=1, output_path=None)
    naive = frame.loc[frame.baseline == "naive_training_base_rate"].iloc[0]

    assert naive.brier == pytest.approx(0.25)


def test_grid_uses_configured_parameters(monkeypatch) -> None:
    X, y, splits = make_data()
    config = tiny_config()
    config.experiment.models[0].params = {"C": 2.5}
    captured: list[dict] = []

    def fake_run_single(X, y, train_idx, cal_idx, test_idx, config, **kwargs):
        captured.append(config)
        return {"brier": 0.1, "ece": 0.1, "roc_auc": 0.5, "reliability": {}}

    monkeypatch.setattr("calibmiss.run_experiment.run_single_config", fake_run_single)
    run_grid(X, y, splits, config, n_jobs=1, output_path=None)

    assert captured[0]["model_params"] == {"C": 2.5}


def test_summary_pairs_fold_seed_config_and_corrects_p_values() -> None:
    rows = []
    for fold, score in enumerate([0.4, 0.5, 0.6]):
        for seed in [1, 2]:
            for imputer, brier in [("median", score), ("forward_fill", score + 0.1)]:
                rows.append(
                    {
                        "row_type": "experiment",
                        "fold": fold,
                        "seed": seed,
                        "model": "logistic_regression",
                        "imputer": imputer,
                        "calibrator": "none",
                        "mechanism": "MCAR",
                        "rate": 0.1,
                        "brier": brier,
                        "ece": brier,
                        "roc_auc": 1 - brier,
                    }
                )
    summary, tests = summarize_results(pd.DataFrame(rows))

    assert "brier_std" in summary
    assert len(tests) == 3
    assert tests["n_pairs"].eq(3).all()
    assert tests["mean_paired_difference"].abs().gt(0).all()
    assert tests["wilcoxon_rank_effect"].abs().eq(1.0).all()
    assert tests["p_holm"].notna().all()


def test_summary_effect_ignores_zero_differences() -> None:
    rows = []
    for fold, difference in enumerate([0.0, 0.1, 0.2]):
        for imputer, score in [("median", 0.4), ("forward_fill", 0.4 + difference)]:
            rows.append(
                {
                    "row_type": "experiment",
                    "fold": fold,
                    "seed": 1,
                    "model": "logistic_regression",
                    "imputer": imputer,
                    "calibrator": "none",
                    "mechanism": "MCAR",
                    "rate": 0.1,
                    "brier": score,
                    "ece": score,
                    "roc_auc": 1 - score,
                }
            )

    _, tests = summarize_results(pd.DataFrame(rows))
    brier = tests[tests["metric"] == "brier"].iloc[0]

    assert brier["n_pairs"] == 3
    assert brier["wilcoxon_rank_effect"] == pytest.approx(1.0)


def test_main_preserves_zero_rate_and_seed(monkeypatch) -> None:
    cfg = tiny_config()
    cfg.experiment.imputers = [
        item
        for item in get_config(Path(__file__).resolve().parents[1] / "config").experiment.imputers
        if item.name == "median"
    ]
    X, y, splits = make_data()
    captured = []
    monkeypatch.setattr(
        "calibmiss.run_experiment.get_config", lambda path, experiment_file="experiment.yaml": cfg
    )
    monkeypatch.setattr(
        "calibmiss.run_experiment._load_data",
        lambda config, volatility_target=False: (X, y, "2026-11-11"),
    )
    monkeypatch.setattr("calibmiss.run_experiment.get_split_indices", lambda dates, config: splits)
    monkeypatch.setattr(
        "calibmiss.run_experiment.write_run_metadata", lambda download_date, path=None: None
    )
    monkeypatch.setattr(
        "calibmiss.run_experiment.run_single_config",
        lambda *args, **kwargs: captured.append(args[5]) or {"brier": 0.1},
    )

    main(["--model", "logistic_regression", "--rate", "0", "--seed", "0"])

    assert captured[0]["rate"] == 0.0
    assert captured[0]["seed"] == 0


def test_main_volatility_config_writes_to_separate_dir(tmp_path: Path, monkeypatch) -> None:
    cfg = tiny_config()
    X, y, splits = make_data()
    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "calibmiss.run_experiment.get_config", lambda path, experiment_file="experiment.yaml": cfg
    )
    monkeypatch.setattr(
        "calibmiss.run_experiment._load_data",
        lambda config, volatility_target=False: (X, y, "2026-11-12"),
    )
    monkeypatch.setattr("calibmiss.run_experiment.get_split_indices", lambda dates, config: splits)
    monkeypatch.setattr(
        "calibmiss.run_experiment.write_run_metadata", lambda download_date, path=None: None
    )

    def fake_run_grid(X, y, splits, config, n_jobs, output_path, volatility_target=False):
        seen["output_path"] = Path(output_path)
        seen["volatility_target"] = volatility_target
        seen["config_name"] = config.experiment.models[0].name
        return pd.DataFrame([{"row_type": "experiment"}])

    monkeypatch.setattr("calibmiss.run_experiment.run_grid", fake_run_grid)

    main(
        [
            "--experiment-config",
            "experiment_volatility.yaml",
            "--output-dir",
            str(tmp_path / "volatility"),
        ]
    )

    assert seen["output_path"] == tmp_path / "volatility" / "metrics.csv"
    assert seen["volatility_target"] is True
    assert seen["config_name"] == cfg.experiment.models[0].name


def test_write_metadata_records_package_versions(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    write_run_metadata("2026-11-11", path)
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["download_date"] == "2026-11-11"
    assert {"numpy", "pandas", "scikit-learn", "lightgbm", "yfinance", "joblib"} <= set(
        metadata["packages"]
    )
