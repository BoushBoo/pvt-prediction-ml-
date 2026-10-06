"""All plots are derived from the evaluation prediction table; never refit."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import summarize


def plot_predictions(predictions, output_dir, target):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, p in predictions.groupby("model", sort=False):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(p.actual, p.predicted, s=12, alpha=0.6)
        bounds = [
            min(p.actual.min(), p.predicted.min()),
            max(p.actual.max(), p.predicted.max()),
        ]
        axes[0].plot(bounds, bounds, "--", color="black")
        axes[0].set(
            xlabel=f"Observed {target}",
            ylabel=f"Outer-fold predicted {target}",
            title=name,
        )
        axes[1].scatter(p.predicted, p.actual - p.predicted, s=12, alpha=0.6)
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].set(
            xlabel=f"Outer-fold predicted {target}",
            ylabel="Observed − predicted",
            title="Residuals",
        )
        fig.tight_layout()
        fig.savefig(out / f"{name}_oof.png", dpi=160)
        plt.close(fig)
    _, summary, _ = summarize(predictions)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ("rmse", "mae", "r2")):
        values = summary[f"{metric}_mean"]
        errors = summary[f"{metric}_std"].fillna(0)
        ax.bar(summary.index, values, yerr=errors, capsize=4)
        ax.tick_params(axis="x", rotation=60)
        ax.set(ylabel=metric.upper(), title="Outer-fold mean ± sample SD")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
