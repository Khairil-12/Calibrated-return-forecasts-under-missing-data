import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import generate_figures


def metrics_fixture(path: Path) -> None:
    rows = []
    for model, imputer, calibrator, mechanism, rate, brier, ece in [
        ("logistic_regression", "median", "none", "MCAR", 0.0, 0.20, 0.10),
        ("logistic_regression", "median", "platt", "MCAR", 0.0, 0.18, 0.08),
        ("lightgbm", "forward_fill", "none", "MAR", 0.3, 0.25, 0.14),
        ("lightgbm", "forward_fill", "platt", "MAR", 0.3, 0.22, 0.12),
    ]:
        rows.append(
            {
                "row_type": "experiment",
                "model": model,
                "imputer": imputer,
                "calibrator": calibrator,
                "mechanism": mechanism,
                "rate": rate,
                "brier": brier,
                "ece": ece,
                "reliability": json.dumps(
                    {
                        "bin_edges": [0.0, 0.5, 1.0],
                        "bin_accuracies": [0.3, 0.7],
                        "bin_confidences": [0.25, 0.75],
                        "bin_counts": [10, 10],
                    }
                ),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_figures_writes_all_prd_outputs(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    output_dir = tmp_path / "figures"
    metrics_fixture(metrics_path)

    outputs = generate_figures.generate_figures(metrics_path, output_dir)

    assert {path.name for path in outputs} == {
        "brier_vs_missingness.png",
        "ece_vs_missingness.png",
        "reliability_0pct.png",
        "reliability_30pct.png",
        "heatmap_calibration.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_reliability_plot_separates_imputers(tmp_path: Path, monkeypatch) -> None:
    metrics_path = tmp_path / "metrics.csv"
    metrics_fixture(metrics_path)
    frame = pd.read_csv(metrics_path)
    extra = frame.iloc[[0]].copy()
    extra["imputer"] = "forward_fill"
    extra["reliability"] = json.dumps(
        {
            "bin_edges": [0.0, 0.5, 1.0],
            "bin_accuracies": [0.9, 0.1],
            "bin_confidences": [0.2, 0.8],
            "bin_counts": [10, 10],
        }
    )
    frame = pd.concat([frame, extra], ignore_index=True)
    selected = frame[frame["rate"] == 0.0]
    figures = []
    original_subplots = generate_figures.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(generate_figures.plt, "subplots", capture_subplots)

    assert generate_figures._reliability_plot(selected, 0.0, tmp_path / "reliability.png")
    figure = figures[0]
    labels = [line.get_label() for line in figure.axes[0].lines]
    assert any("median" in label for label in labels)
    assert any("forward_fill" in label for label in labels)
    generate_figures.plt.close(figure)


def test_generate_figures_requires_experiment_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame({"row_type": ["baseline"]}).to_csv(metrics_path, index=False)

    with pytest.raises(ValueError, match="No experiment rows"):
        generate_figures.generate_figures(metrics_path, tmp_path / "figures")


def test_generate_figures_requires_reliability_rates(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    metrics_fixture(metrics_path)
    frame = pd.read_csv(metrics_path)
    frame = frame[frame["rate"] != 0.3]
    frame.to_csv(metrics_path, index=False)

    with pytest.raises(ValueError, match="Missing required reliability rate 30%"):
        generate_figures.generate_figures(metrics_path, tmp_path / "figures")
