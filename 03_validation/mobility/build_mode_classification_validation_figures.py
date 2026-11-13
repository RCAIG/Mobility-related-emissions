"""Build held-out validation figures for four- and five-class mode models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(r"${TRANSPORT_WORKSPACE}\\Observed_Urban_Mobility_Emission_Paper_V25")
DATA = ROOT / "supplementary_fig" / "mode_classification_validation"
OUT = ROOT / "supplementary_fig"

INK = "#223034"
GRID = "#DDE5E4"
METRIC_COLORS = ("#277F78", "#D58A68", "#D3A13B")
CLASS_COLORS = ("#2A6F97", "#4D908E", "#72A85A", "#E3A832", "#C8553D")
CMAP = LinearSegmentedColormap.from_list(
    "classification_teal", ("#F4F8F7", "#BBD8D3", "#4B9A91", "#155653")
)


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.045, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11.5, fontweight="bold", color=INK, clip_on=False)


def bootstrap_metric_intervals(
    predictions: pd.DataFrame, n_bootstrap: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Return stratified-bootstrap 95% confidence limits for three metrics."""
    true = predictions.true_label.to_numpy()
    pred = predictions.pred_label.to_numpy()
    classes = np.unique(true)
    class_indices = [np.flatnonzero(true == label) for label in classes]
    rng = np.random.default_rng(42)
    samples = np.empty((n_bootstrap, 3), dtype=float)

    for iteration in range(n_bootstrap):
        chosen = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        sampled_true = true[chosen]
        sampled_pred = pred[chosen]
        recalls = [
            np.mean(sampled_pred[sampled_true == label] == label)
            for label in classes
        ]
        samples[iteration] = (
            np.mean(sampled_true == sampled_pred),
            np.mean(recalls),
            f1_score(sampled_true, sampled_pred, labels=classes, average="macro"),
        )

    return np.quantile(samples, 0.025, axis=0), np.quantile(samples, 0.975, axis=0)


def draw_metrics(ax: plt.Axes, metrics: dict, predictions: pd.DataFrame) -> None:
    labels = ["Accuracy", "Balanced\naccuracy", "Macro-F1"]
    values = np.asarray([
        metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"]
    ])
    lower, upper = bootstrap_metric_intervals(predictions)
    errors = np.vstack((values - lower, upper - values))
    x = np.arange(3)
    ax.bar(x, values, width=0.64, color=METRIC_COLORS, edgecolor="white", linewidth=0.5)
    ax.errorbar(
        x, values, yerr=errors, fmt="none", ecolor=INK,
        elinewidth=0.65, capsize=2.0, capthick=0.65, zorder=4,
    )
    for xpos, value, error in zip(x, values, errors[1]):
        ax.text(xpos, value + error + 0.022, f"{value:.3f}", ha="center", va="bottom",
                fontsize=8.2, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out test score")
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "a")


def draw_confusion(ax: plt.Axes, predictions: pd.DataFrame, n_classes: int) -> None:
    labels = np.arange(n_classes)
    cm = confusion_matrix(predictions.true_label, predictions.pred_label, labels=labels)
    row_total = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    normalized = cm / row_total
    image = ax.imshow(normalized, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    for row in range(n_classes):
        for col in range(n_classes):
            value = normalized[row, col]
            color = "white" if value >= 0.54 else INK
            ax.text(col, row, f"{value * 100:.1f}%", ha="center", va="center",
                    fontsize=6.7 if n_classes == 5 else 7.2, color=color)
    names = [f"Mode {i + 1}" for i in range(n_classes)]
    ax.set_xticks(labels, names, rotation=0)
    ax.set_yticks(labels, names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = inset_axes(
        ax, width="3.5%", height="100%", loc="lower left",
        bbox_to_anchor=(1.045, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0,
    )
    cbar = ax.figure.colorbar(image, cax=cax)
    cbar.set_label("Row-normalized share", fontsize=7.7)
    cbar.ax.tick_params(labelsize=6.8, length=2.5)
    cbar.outline.set_linewidth(0.55)
    panel_label(ax, "b")


def draw_latent(ax: plt.Axes, path: Path) -> None:
    source = Image.open(path).convert("RGB")
    width, height = source.size
    # Keep the source tick labels and legend, but remove its title and embedded
    # x-axis label. The label is redrawn below so the three visible baselines align.
    image = source.crop((int(width * 0.045), int(height * 0.060),
                         int(width * 0.985), int(height * 0.950)))
    # The source x-axis baseline is 4.8% above the cropped image bottom.
    # Extending that small strip below the axes maps the baseline to y=0.
    ax.imshow(image, aspect="auto", extent=(0, 1, -0.0505, 1), clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0.5, -0.125, "t-SNE dimension 1", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color=INK, clip_on=False)
    panel_label(ax, "c")


def build(prefix: str, n_classes: int, output_stem: str) -> None:
    metrics = json.loads((DATA / f"{prefix}_metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(DATA / f"{prefix}_predictions.csv")

    fig = plt.figure(figsize=(10.5, 3.45), facecolor="white")
    grid = fig.add_gridspec(
        1, 3, width_ratios=(1, 1, 1),
        left=0.065, right=0.985, bottom=0.19, top=0.91, wspace=0.30,
    )
    metric_ax = fig.add_subplot(grid[0, 0])
    confusion_ax = fig.add_subplot(grid[0, 1])
    latent_ax = fig.add_subplot(grid[0, 2])

    draw_metrics(metric_ax, metrics, predictions)
    draw_confusion(confusion_ax, predictions, n_classes)
    draw_latent(latent_ax, DATA / f"{prefix}_latent.png")

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{output_stem}.{suffix}", dpi=360, bbox_inches="tight",
                    pad_inches=0.035, facecolor="white")
    plt.close(fig)


def main() -> None:
    set_style()
    build("five_class", 5, "FigS_mode_five_class_validation")
    build("four_class", 4, "FigS_mode_four_class_validation")


if __name__ == "__main__":
    main()
