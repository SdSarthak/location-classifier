"""Matplotlib figures for datasets, clusters and model results.

Every function accepts an optional ``path``; when given, the figure is written
there and the path is returned, otherwise the ``Axes`` is returned so notebooks
can keep working with it. A non-interactive backend is selected automatically
when the process has no display, which is what makes the CLI safe to run on a
headless machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import matplotlib

if not os.environ.get("MPLBACKEND") and "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import must follow the backend choice)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .cluster import NOISE_LABEL  # noqa: E402
from .config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN  # noqa: E402
from .evaluate import confusion_frame  # noqa: E402

Output = Union[Path, plt.Axes]

DEFAULT_FIGSIZE = (9.0, 7.0)


def _finish(fig: plt.Figure, ax: plt.Axes, path: Optional[Path | str], owns_figure: bool) -> Output:
    """Save and close when a path was given, otherwise hand back the axes."""
    fig.tight_layout()
    if path is None:
        return ax
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if owns_figure:
        plt.close(fig)
    return path


def _palette(n: int) -> np.ndarray:
    """Distinguishable colours for up to ~20 categories, cycling beyond that."""
    name = "tab10" if n <= 10 else "tab20"
    colormap = matplotlib.colormaps[name]
    return np.array([colormap(index % colormap.N) for index in range(max(n, 1))])


def plot_locations(
    frame: pd.DataFrame,
    label_column: Optional[str] = LABEL_COLUMN,
    path: Optional[Path | str] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Location dataset",
) -> Output:
    """Scatter the dataset in lat/lon space, coloured by class."""
    for column in (LATITUDE_COLUMN, LONGITUDE_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"frame is missing the '{column}' column")

    owns_figure = ax is None
    fig, ax = (plt.subplots(figsize=DEFAULT_FIGSIZE) if owns_figure else (ax.figure, ax))

    if label_column and label_column in frame.columns:
        groups = sorted(frame[label_column].astype(str).unique())
        colors = _palette(len(groups))
        for index, name in enumerate(groups):
            subset = frame[frame[label_column].astype(str) == name]
            ax.scatter(
                subset[LONGITUDE_COLUMN],
                subset[LATITUDE_COLUMN],
                s=18,
                alpha=0.75,
                color=colors[index],
                label=name,
            )
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.scatter(frame[LONGITUDE_COLUMN], frame[LATITUDE_COLUMN], s=18, alpha=0.75)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle=":")
    return _finish(fig, ax, path, owns_figure)


def plot_clusters(
    frame: pd.DataFrame,
    labels: Sequence,
    centroids: Optional[np.ndarray] = None,
    path: Optional[Path | str] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Discovered clusters",
) -> Output:
    """Scatter points coloured by cluster, with noise drawn in grey."""
    labels = np.asarray(labels)
    if len(labels) != len(frame):
        raise ValueError(
            f"labels length {len(labels)} does not match frame length {len(frame)}"
        )

    owns_figure = ax is None
    fig, ax = (plt.subplots(figsize=DEFAULT_FIGSIZE) if owns_figure else (ax.figure, ax))

    clusters = [value for value in np.unique(labels) if value != NOISE_LABEL]
    colors = _palette(len(clusters))
    for index, value in enumerate(clusters):
        mask = labels == value
        ax.scatter(
            frame.loc[mask, LONGITUDE_COLUMN],
            frame.loc[mask, LATITUDE_COLUMN],
            s=18,
            alpha=0.75,
            color=colors[index],
            label=f"cluster {value}",
        )

    noise = labels == NOISE_LABEL
    if noise.any():
        ax.scatter(
            frame.loc[noise, LONGITUDE_COLUMN],
            frame.loc[noise, LATITUDE_COLUMN],
            s=14,
            color="0.6",
            marker="x",
            label="noise",
        )

    if centroids is not None and len(centroids):
        centroids = np.asarray(centroids, dtype=float)
        ax.scatter(
            centroids[:, 1],
            centroids[:, 0],
            s=160,
            marker="*",
            color="black",
            label="centroids",
            zorder=5,
        )

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="best", fontsize=8)
    return _finish(fig, ax, path, owns_figure)


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    path: Optional[Path | str] = None,
    ax: Optional[plt.Axes] = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> Output:
    """Heatmap of the confusion matrix, annotated with per-cell values."""
    frame = confusion_frame(metrics)
    values = frame.to_numpy(dtype=float)
    if normalize:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)

    owns_figure = ax is None
    fig, ax = (plt.subplots(figsize=(8.0, 7.0)) if owns_figure else (ax.figure, ax))

    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=values.max() or 1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(frame.columns)), labels=frame.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(frame.index)), labels=frame.index)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title + (" (row-normalised)" if normalize else ""))

    threshold = (values.max() or 1.0) / 2.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[i, j] > threshold else "black",
            )
    return _finish(fig, ax, path, owns_figure)


def plot_feature_importances(
    importances: pd.DataFrame,
    path: Optional[Path | str] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Feature importance",
) -> Output:
    """Horizontal bar chart of the frame returned by ``model.feature_importances``."""
    if importances.empty:
        raise ValueError("no feature importances to plot")
    required = {"feature", "importance"}
    if not required.issubset(importances.columns):
        raise ValueError(f"importances frame must have columns {sorted(required)}")

    owns_figure = ax is None
    fig, ax = (plt.subplots(figsize=(8.0, 6.0)) if owns_figure else (ax.figure, ax))

    ordered = importances.sort_values("importance")
    ax.barh(ordered["feature"], ordered["importance"], color="#3b7dd8")
    ax.set_xlabel("importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    return _finish(fig, ax, path, owns_figure)


def plot_model_comparison(
    leaderboard: pd.DataFrame,
    path: Optional[Path | str] = None,
    ax: Optional[plt.Axes] = None,
    metric: str = "accuracy",
    title: str = "Model comparison",
) -> Output:
    """Bar chart of the leaderboard returned by ``model.compare_models``."""
    if leaderboard.empty:
        raise ValueError("no models to compare")
    if metric not in leaderboard.columns:
        raise ValueError(f"leaderboard has no '{metric}' column")

    owns_figure = ax is None
    fig, ax = (plt.subplots(figsize=(8.0, 5.0)) if owns_figure else (ax.figure, ax))

    ordered = leaderboard.sort_values(metric)
    ax.barh(ordered["model"], ordered[metric], color="#4c9a6a")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(metric)
    ax.set_title(title)
    for index, value in enumerate(ordered[metric]):
        ax.text(min(value + 0.01, 0.98), index, f"{value:.3f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    return _finish(fig, ax, path, owns_figure)
