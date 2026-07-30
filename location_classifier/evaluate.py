"""Classification metrics and human-readable reports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def evaluate_predictions(
    y_true: Sequence,
    y_pred: Sequence,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[Sequence] = None,
    top_k: int = 2,
) -> Dict[str, Any]:
    """Compute the metric bundle used by the CLI and the notebook.

    Args:
        y_true: ground-truth labels.
        y_pred: predicted labels.
        y_proba: optional ``(n, n_classes)`` probability matrix enabling top-k
            accuracy.
        classes: class order matching the columns of ``y_proba``.
        top_k: ``k`` for top-k accuracy; skipped when the model has fewer
            classes than ``k``.

    Returns:
        Dict of scalar metrics plus ``per_class``, ``confusion_matrix`` and
        ``labels`` entries.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true/y_pred length mismatch: {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("cannot evaluate an empty set of predictions")

    labels = list(classes) if classes is not None else sorted(set(y_true) | set(y_pred))

    metrics: Dict[str, Any] = {
        "n_samples": int(y_true.size),
        "n_classes": len(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "labels": [str(label) for label in labels],
    }

    if y_proba is not None and len(labels) > top_k >= 2:
        proba = np.asarray(y_proba, dtype=float)
        if proba.shape != (y_true.size, len(labels)):
            raise ValueError(
                f"y_proba shape {proba.shape} does not match "
                f"({y_true.size}, {len(labels)})"
            )
        metrics[f"top_{top_k}_accuracy"] = float(
            top_k_accuracy_score(y_true, proba, k=top_k, labels=labels)
        )

    metrics["per_class"] = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics


def confusion_frame(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = true, cols = predicted)."""
    if "confusion_matrix" not in metrics:
        raise KeyError("metrics has no 'confusion_matrix' entry")
    labels = metrics["labels"]
    return pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)


def per_class_frame(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Per-class precision/recall/F1 as a DataFrame, ordered by F1 ascending."""
    report = metrics.get("per_class", {})
    rows = {
        name: values
        for name, values in report.items()
        if isinstance(values, dict) and name in set(metrics.get("labels", []))
    }
    frame = pd.DataFrame(rows).T
    if frame.empty:
        return frame
    return frame.sort_values("f1-score")


def worst_confusions(metrics: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """The most frequent off-diagonal true/predicted pairs."""
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    labels = metrics["labels"]
    pairs = [
        {"true": labels[i], "predicted": labels[j], "count": int(matrix[i, j])}
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[1])
        if i != j and matrix[i, j] > 0
    ]
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:limit]


SCALAR_KEYS = (
    "n_samples",
    "n_classes",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
    "top_2_accuracy",
)


def format_metrics(metrics: Dict[str, Any], title: str = "Evaluation") -> str:
    """Render the metric bundle as an aligned block of text."""
    lines = [title, "-" * len(title)]
    for key in SCALAR_KEYS:
        if key not in metrics:
            continue
        value = metrics[key]
        lines.append(
            f"{key:<20} {value}" if isinstance(value, int) else f"{key:<20} {value:.4f}"
        )

    frame = per_class_frame(metrics)
    if not frame.empty:
        lines.append("")
        lines.append("per class (worst F1 first)")
        for name, row in frame.iterrows():
            lines.append(
                f"  {str(name):<16} precision={row['precision']:.3f} "
                f"recall={row['recall']:.3f} f1={row['f1-score']:.3f} "
                f"support={int(row['support'])}"
            )

    confusions = worst_confusions(metrics)
    if confusions:
        lines.append("")
        lines.append("most confused pairs")
        for pair in confusions:
            lines.append(
                f"  {pair['true']} -> {pair['predicted']}: {pair['count']}"
            )
    return "\n".join(lines)
