"""Generate report figures from the experiment metrics table."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]


def _experiments(metrics_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(metrics_path)
    experiments = frame[frame["row_type"] == "experiment"].copy()
    if experiments.empty:
        raise ValueError(f"No experiment rows in {metrics_path}")
    return experiments


def _plot_metric(experiments: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    grouped = (
        experiments.groupby(
            ["mechanism", "imputer", "model", "calibrator", "rate"], as_index=False
        )[metric]
        .mean()
        .sort_values("rate")
    )
    mechanisms = sorted(grouped["mechanism"].unique())
    imputers = sorted(grouped["imputer"].unique())
    fig, axes = plt.subplots(
        len(mechanisms),
        len(imputers),
        figsize=(4.2 * len(imputers), 3.3 * len(mechanisms)),
        squeeze=False,
        sharex=True,
    )
    for row, mechanism in enumerate(mechanisms):
        for col, imputer in enumerate(imputers):
            ax = axes[row, col]
            subset = grouped[(grouped["mechanism"] == mechanism) & (grouped["imputer"] == imputer)]
            sns.lineplot(
                data=subset,
                x="rate",
                y=metric,
                hue="model",
                style="calibrator",
                markers=True,
                dashes=True,
                ax=ax,
            )
            ax.set_title(f"{mechanism} · {imputer}")
            ax.set_xlabel("Missingness rate")
            ax.set_ylabel(metric.upper().replace("_", " "))
            ax.grid(axis="y", color="#e1e0d9", linewidth=0.7)
            ax.set_axisbelow(True)
            if row or col:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
        for ax in axes.flat:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = output_dir / f"{metric}_vs_missingness.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _reliability_plot(experiments: pd.DataFrame, rate: float, path: Path) -> bool:
    selected = experiments[experiments["rate"] == rate]
    if selected.empty:
        return False
    selected = selected.copy()
    selected["reliability"] = selected["reliability"].map(
        lambda value: json.loads(value) if isinstance(value, str) else value
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    for (model, imputer, mechanism, calibrator), group in selected.groupby(
        ["model", "imputer", "mechanism", "calibrator"], sort=True
    ):
        bins = group["reliability"].map(lambda rel: rel["bin_confidences"])
        accuracies = group["reliability"].map(lambda rel: rel["bin_accuracies"])
        counts = group["reliability"].map(lambda rel: rel["bin_counts"])
        edges = group["reliability"].iloc[0]["bin_edges"]
        centers = [(left + right) / 2 for left, right in zip(edges[:-1], edges[1:])]
        weighted_confidence: list[float] = []
        weighted_accuracy: list[float] = []
        for index in range(len(centers)):
            weights = [count[index] for count in counts]
            total = sum(weights)
            if total:
                weighted_confidence.append(
                    sum(row[index] * weight for row, weight in zip(bins, weights)) / total
                )
                weighted_accuracy.append(
                    sum(row[index] * weight for row, weight in zip(accuracies, weights)) / total
                )
            else:
                weighted_confidence.append(float("nan"))
                weighted_accuracy.append(float("nan"))
        ax.plot(
            weighted_confidence,
            weighted_accuracy,
            marker="o",
            label=f"{model} · {imputer} · {mechanism} · {calibrator}",
        )
    ax.plot(
        [0, 1], [0, 1], color="#898781", linewidth=1, linestyle="--", label="Perfect calibration"
    )
    ax.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed frequency"
    )
    ax.set_title(f"Reliability at {rate:.0%} missingness")
    ax.grid(color="#e1e0d9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_figures(metrics_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate the PRD's metric, reliability, and calibration heatmap figures."""
    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir)
    experiments = _experiments(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [_plot_metric(experiments, metric, output_dir) for metric in ("brier", "ece")]
    for rate, label in ((0.0, "0pct"), (0.3, "30pct")):
        path = output_dir / f"reliability_{label}.png"
        if not _reliability_plot(experiments, rate, path):
            raise ValueError(f"Missing required reliability rate {rate:.0%}")
        outputs.append(path)
    heat = (
        experiments.groupby(["model", "imputer", "calibrator", "mechanism", "rate"])["ece"]
        .mean()
        .reset_index()
    )
    condition_columns = ["model", "imputer", "calibrator", "mechanism"]
    heat["condition"] = heat[condition_columns].astype(str).agg(" | ".join, axis=1)
    matrix = heat.pivot(index="condition", columns="rate", values="ece").sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    fig_height = max(4, 0.24 * len(matrix))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=False,
        cbar_kws={"label": "Mean ECE"},
        xticklabels=[f"{rate:.0%}" for rate in matrix.columns],
        ax=ax,
    )
    ax.set_title("Calibration error across experimental grid")
    ax.set_xlabel("Missingness rate")
    ax.set_ylabel("Model · imputer · calibrator · mechanism")
    fig.tight_layout()
    path = output_dir / "heatmap_calibration.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs.append(path)
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(ROOT / "results" / "metrics.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args()
    outputs = generate_figures(args.metrics, args.output_dir)
    print(f"Generated {len(outputs)} figures in {args.output_dir}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
