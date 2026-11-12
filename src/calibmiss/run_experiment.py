"""Run CalibMiss walk-forward experiments and summarize results."""

import argparse
import importlib.metadata
import itertools
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from calibmiss.config import Config, get_config
from calibmiss.data import download_ohlcv
from calibmiss.features import compute_features
from calibmiss.metrics import compute_metrics
from calibmiss.pipeline import run_single_config
from calibmiss.splits import SplitIndices, get_split_indices

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
METRICS_PATH = RESULTS / "metrics.csv"
SUMMARY_PATH = RESULTS / "summary.csv"
STATISTICAL_TESTS_PATH = RESULTS / "statistical_tests.csv"
METADATA_PATH = RESULTS / "run_metadata.json"


def map_date_splits_to_rows(X: pd.DataFrame, date_splits: list[SplitIndices]) -> list[SplitIndices]:
    """Expand unique-date split positions to all matching feature-row positions."""
    dates = pd.DatetimeIndex(X.index.get_level_values("date"))
    unique_dates = dates.unique().sort_values()
    date_positions_by_value = {date: position for position, date in enumerate(unique_dates)}
    row_date_positions = np.fromiter(
        (date_positions_by_value[date] for date in dates), dtype=np.intp, count=len(dates)
    )
    mapped = []
    for split in date_splits:
        indices = [
            np.flatnonzero(np.isin(row_date_positions, date_positions))
            for date_positions in (split.train_idx, split.calibrate_idx, split.test_idx)
        ]
        if any(not index.size for index in indices):
            raise ValueError("Each date split must map to at least one feature row")
        mapped.append(
            SplitIndices(
                train_idx=indices[0],
                calibrate_idx=indices[1],
                test_idx=indices[2],
                train_start=split.train_start,
                train_end=split.train_end,
                calibrate_start=split.calibrate_start,
                calibrate_end=split.calibrate_end,
                test_start=split.test_start,
                test_end=split.test_end,
            )
        )
    return mapped


def _row_config(
    model: Any,
    imputer: Any,
    calibrator: Any,
    mechanism: str,
    rate: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "model": model.name,
        "model_params": model.params,
        "imputer": imputer.name,
        "imputer_params": imputer.params,
        "calibrator": calibrator.name,
        "calibrator_params": calibrator.params,
        "mechanism": mechanism,
        "rate": rate,
        "seed": seed,
    }


def _persistence_row(
    X: pd.DataFrame,
    y: pd.Series,
    split: SplitIndices,
    fold: int,
    cfg: Config,
) -> dict[str, Any]:
    """Persistence baseline: probability = volatility_5d at day t."""
    vol_col = "volatility_5d"
    if vol_col not in X.columns:
        raise ValueError(f"Persistence requires feature column '{vol_col}'")
    test_probs = X.iloc[split.test_idx][vol_col].to_numpy()
    test_y = y.iloc[split.test_idx].to_numpy()
    result = compute_metrics(test_y, test_probs, n_bins=cfg.experiment.evaluation.n_bins_ece)
    return {
        **result,
        "row_type": "baseline",
        "baseline": "volatility_persistence",
        "fold": fold,
        "seed": None,
        "model": "persistence",
        "imputer": "none",
        "calibrator": "none",
        "mechanism": "none",
        "rate": 0.0,
    }


def _baseline_rows(
    X: pd.DataFrame,
    y: pd.Series,
    split: SplitIndices,
    fold: int,
    cfg: Config,
) -> list[dict[str, Any]]:
    train_y = y.iloc[split.train_idx].to_numpy()
    test_y = y.iloc[split.test_idx].to_numpy()
    base_rate = float(np.mean(train_y))
    naive = compute_metrics(
        test_y,
        np.full(len(test_y), base_rate),
        n_bins=cfg.experiment.evaluation.n_bins_ece,
    )
    rows: list[dict[str, Any]] = [
        {
            **naive,
            "row_type": "baseline",
            "baseline": "naive_training_base_rate",
            "fold": fold,
            "seed": None,
            "model": "naive",
            "imputer": "none",
            "calibrator": "none",
            "mechanism": "none",
            "rate": 0.0,
        }
    ]
    for model in cfg.experiment.models:
        for calibrator in cfg.experiment.calibrators:
            result = run_single_config(
                X,
                y,
                split.train_idx,
                split.calibrate_idx,
                split.test_idx,
                _row_config(model, cfg.experiment.imputers[0], calibrator, "MCAR", 0.0, 0),
                experiment_config=cfg,
            )
            rows.append(
                {
                    **result,
                    "row_type": "baseline",
                    "baseline": "complete_data_oracle",
                    "fold": fold,
                    "mechanism": "none",
                    "rate": 0.0,
                }
            )
    return rows


def _binarize_volatility_target(
    y: pd.Series, split: SplitIndices
) -> tuple[pd.Series, float]:
    """Binarize a continuous target at the training-fold median.

    The threshold comes only from `y[train_idx]`; calibrate and test labels are
    binarized with it but never inform it.
    """
    threshold = float(np.median(y.iloc[split.train_idx].to_numpy()))
    binary = (y > threshold).astype(int)
    binary.name = "target"
    return binary, threshold


def run_grid(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[SplitIndices],
    config: Config | None = None,
    n_jobs: int = -1,
    output_path: str | Path | None = METRICS_PATH,
    volatility_target: bool = False,
) -> pd.DataFrame:
    """Run every configured fold/seed/method and write the result table.

    With `volatility_target=True`, `y` is a continuous realized-volatility series;
    each fold binarizes it at the training-fold median and adds a persistence
    baseline (probability = `volatility_5d` at day t) next to the naive base rate.
    """
    cfg = config or get_config(ROOT / "config")
    if volatility_target:
        y_by_fold = {
            fold: _binarize_volatility_target(y, split)[0] for fold, split in enumerate(splits)
        }
    else:
        y_by_fold = {fold: y for fold in range(len(splits))}
    combinations = [
        (fold, seed, model, imputer, calibrator, mechanism, rate)
        for fold in range(len(splits))
        for seed, model, imputer, calibrator, mechanism, rate in itertools.product(
            cfg.experiment.missingness.seeds,
            cfg.experiment.models,
            cfg.experiment.imputers,
            cfg.experiment.calibrators,
            cfg.experiment.missingness.mechanisms,
            cfg.experiment.missingness.rates,
        )
    ]

    def run_one(combo: tuple[Any, ...]) -> dict[str, Any]:
        fold, seed, model, imputer, calibrator, mechanism, rate = combo
        split = splits[fold]
        result = run_single_config(
            X,
            y_by_fold[fold],
            split.train_idx,
            split.calibrate_idx,
            split.test_idx,
            _row_config(model, imputer, calibrator, mechanism, rate, seed),
            experiment_config=cfg,
        )
        return {**result, "row_type": "experiment", "baseline": None, "fold": fold}

    results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_one)(combo) for combo in combinations)
    for fold, split in enumerate(splits):
        results.extend(_baseline_rows(X, y_by_fold[fold], split, fold, cfg))
        if volatility_target:
            results.append(
                _persistence_row(X, y_by_fold[fold], split, fold, cfg)
            )
    frame = pd.DataFrame(results)
    frame["reliability"] = frame["reliability"].map(
        lambda value: json.dumps(value, allow_nan=False) if isinstance(value, dict) else value
    )
    if output_path is not None:
        output = Path(output_path)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        summary, tests = summarize_results(frame)
        summary.to_csv(output.with_name("summary.csv"), index=False)
        tests.to_csv(output.with_name("statistical_tests.csv"), index=False)
    return frame


def summarize_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize fold-level dispersion and paired comparisons with Holm correction."""
    experiments = frame[frame["row_type"] == "experiment"]
    metrics = ["brier", "ece", "roc_auc"]
    summary = (
        experiments.groupby(["model", "imputer", "calibrator", "mechanism", "rate"], dropna=False)[
            metrics
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    comparisons: list[dict[str, Any]] = []
    for metric in metrics:
        for model in experiments["model"].unique():
            for mechanism in experiments["mechanism"].unique():
                for rate in experiments["rate"].unique():
                    subset = experiments[
                        (experiments["model"] == model)
                        & (experiments["mechanism"] == mechanism)
                        & (experiments["rate"] == rate)
                    ]
                    groups = subset.groupby(["imputer", "calibrator"], sort=True)
                    fold_groups = {
                        name: group.groupby("fold", as_index=False)[metric].mean()
                        for name, group in groups
                    }
                    names = list(fold_groups)
                    for left_index, left_name in enumerate(names):
                        left = fold_groups[left_name]
                        for right_name in names[left_index + 1 :]:
                            right = fold_groups[right_name]
                            paired = left.merge(right, on="fold", suffixes=("_left", "_right"))
                            differences = (
                                paired[f"{metric}_left"] - paired[f"{metric}_right"]
                            ).dropna()
                            values = differences.to_numpy()
                            p_value = float("nan")
                            effect = float("nan")
                            if len(values) >= 2 and np.any(values != 0):
                                p_value = float(wilcoxon(values).pvalue)
                                nonzero = values[values != 0]
                                ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
                                effect = float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks))
                            mean_difference = (
                                float(differences.mean()) if len(differences) else float("nan")
                            )
                            comparisons.append(
                                {
                                    "metric": metric,
                                    "model": model,
                                    "mechanism": mechanism,
                                    "rate": rate,
                                    "left": f"{left_name[0]}|{left_name[1]}",
                                    "right": f"{right_name[0]}|{right_name[1]}",
                                    "n_pairs": len(differences),
                                    "mean_paired_difference": mean_difference,
                                    "wilcoxon_rank_effect": effect,
                                    "p_value": p_value,
                                }
                            )
    tests = pd.DataFrame(comparisons)
    if not tests.empty:
        tests["p_holm"] = np.nan
        for _, indices in tests.groupby("metric").groups.items():
            p_values = tests.loc[indices, "p_value"].to_numpy(dtype=float)
            valid = np.isfinite(p_values)
            order = np.argsort(p_values[valid])
            sorted_p = p_values[valid][order]
            adjusted = np.maximum.accumulate(
                np.minimum(1.0, sorted_p * (len(sorted_p) - np.arange(len(sorted_p))))
            )
            valid_indices = np.asarray(indices)[valid]
            tests.loc[valid_indices[order], "p_holm"] = adjusted
    return summary, tests


def write_run_metadata(download_date: str, path: str | Path = METADATA_PATH) -> None:
    """Write download provenance and installed package versions."""
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "yfinance",
        "joblib",
        "scipy",
        "matplotlib",
        "seaborn",
        "pydantic",
        "PyYAML",
    ]
    versions = {name: importlib.metadata.version(name) for name in packages}
    metadata = {
        "download_date": download_date,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": __import__("platform").python_version(),
        "packages": versions,
    }
    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_data(cfg: Config) -> tuple[pd.DataFrame, pd.Series, str]:
    raw_dir = ROOT / "data" / "raw"
    raw = download_ohlcv(
        cfg.universe.tickers,
        cfg.universe.start_date,
        cfg.universe.end_date,
        cache_dir=raw_dir,
    )
    X, y = compute_features(raw)
    valid_rows = ~X.isna().any(axis=1)
    X = X[valid_rows]
    y = y[valid_rows]
    key = f"{'_'.join(sorted(cfg.universe.tickers))}_{cfg.universe.start_date}_{cfg.universe.end_date}_sha256_"
    cache_files = list(raw_dir.glob(f"*_{key}*.parquet"))
    download_timestamp = max(int(path.name.split("_", 1)[0]) for path in cache_files)
    download_date = datetime.fromtimestamp(download_timestamp, UTC).date().isoformat()
    return X, y, download_date


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--imputer")
    parser.add_argument("--calibrator")
    parser.add_argument("--mechanism")
    parser.add_argument("--rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--experiment-config", default="experiment.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = get_config(ROOT / "config", experiment_file=args.experiment_config)
    output_dir = args.output_dir or (
        RESULTS / "volatility" if args.experiment_config != "experiment.yaml" else RESULTS
    )
    X, y, download_date = _load_data(cfg)
    dates = X.index.get_level_values("date").unique().sort_values()
    date_splits = get_split_indices(dates, cfg.experiment.splits.model_dump())
    splits = map_date_splits_to_rows(X, date_splits)
    write_run_metadata(download_date, output_dir / "run_metadata.json")

    if args.model is not None:
        model = next(item for item in cfg.experiment.models if item.name == args.model)
        imputer_name = "median" if args.imputer is None else args.imputer
        imputer = next(item for item in cfg.experiment.imputers if item.name == imputer_name)
        calibrator_name = "none" if args.calibrator is None else args.calibrator
        calibrator = next(
            item for item in cfg.experiment.calibrators if item.name == calibrator_name
        )
        rate = 0.1 if args.rate is None else args.rate
        seed = cfg.experiment.missingness.seeds[0] if args.seed is None else args.seed
        mechanism = "MCAR" if args.mechanism is None else args.mechanism
        split = splits[0]
        print(
            run_single_config(
                X,
                y,
                split.train_idx,
                split.calibrate_idx,
                split.test_idx,
                _row_config(model, imputer, calibrator, mechanism, rate, seed),
                experiment_config=cfg,
            )
        )
        return

    frame = run_grid(
        X,
        y,
        splits,
        cfg,
        n_jobs=args.n_jobs,
        output_path=output_dir / "metrics.csv",
        volatility_target=(args.experiment_config != "experiment.yaml"),
    )
    print(f"Saved {len(frame)} rows to {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
