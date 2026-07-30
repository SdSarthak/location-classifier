"""Unsupervised grouping of geographic points.

Clustering runs in a metric space where distance is measured in kilometres, not
degrees, so a cluster radius means the same thing in Mumbai and in Oslo. K-means
operates on unit-sphere cartesian coordinates scaled by the Earth's radius;
DBSCAN uses scikit-learn's native haversine metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from .config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN
from .geo import EARTH_RADIUS_KM, haversine_km, spherical_centroid, to_cartesian

#: Label DBSCAN assigns to points that belong to no cluster.
NOISE_LABEL = -1

METHODS = ("kmeans", "dbscan")


@dataclass
class ClusterResult:
    """Outcome of a clustering run."""

    labels: np.ndarray
    method: str
    n_clusters: int
    silhouette: Optional[float] = None
    centroids: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    #: Cluster ids matching the rows of ``centroids``.
    cluster_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_noise(self) -> int:
        """Points DBSCAN left unassigned (always 0 for k-means)."""
        return int(np.sum(self.labels == NOISE_LABEL))

    def describe(self) -> str:
        """Human-readable summary used by the CLI."""
        lines = [
            f"method: {self.method}",
            f"clusters: {self.n_clusters}",
            f"noise points: {self.n_noise}",
            "silhouette: "
            + ("n/a" if self.silhouette is None else f"{self.silhouette:.4f}"),
        ]
        if not self.summary.empty:
            lines.append("")
            lines.append(self.summary.to_string(index=False))
        return "\n".join(lines)


def _coordinates(frame: pd.DataFrame) -> np.ndarray:
    missing = [c for c in (LATITUDE_COLUMN, LONGITUDE_COLUMN) if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing coordinate column(s): {missing}")
    if len(frame) == 0:
        raise ValueError("cannot cluster an empty frame")
    return frame[[LATITUDE_COLUMN, LONGITUDE_COLUMN]].to_numpy(dtype=float)


def _km_space(coords: np.ndarray) -> np.ndarray:
    """Cartesian coordinates in kilometres, where Euclidean distance is the chord."""
    return to_cartesian(coords[:, 0], coords[:, 1]) * EARTH_RADIUS_KM


def cluster_locations(
    frame: pd.DataFrame,
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    eps_km: float = 3.0,
    min_samples: int = 5,
    random_seed: int = 42,
    label_column: Optional[str] = LABEL_COLUMN,
) -> ClusterResult:
    """Group points into spatial clusters.

    Args:
        frame: points with latitude/longitude columns.
        method: ``"kmeans"`` or ``"dbscan"``.
        n_clusters: k for k-means; ``None`` picks the best k by silhouette.
        eps_km: DBSCAN neighbourhood radius, in kilometres.
        min_samples: DBSCAN minimum neighbourhood size.
        random_seed: seed for k-means.
        label_column: optional ground-truth column used to report cluster purity.

    Raises:
        ValueError: on an unknown method, an empty frame, or invalid parameters.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}'; choose one of {list(METHODS)}")
    coords = _coordinates(frame)
    points = _km_space(coords)

    if method == "kmeans":
        k = n_clusters if n_clusters is not None else estimate_n_clusters(frame, random_seed=random_seed)
        k = int(max(1, min(k, len(coords))))
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_seed).fit_predict(points)
    else:
        if eps_km <= 0:
            raise ValueError(f"eps_km must be > 0, got {eps_km}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        labels = DBSCAN(
            eps=eps_km / EARTH_RADIUS_KM,
            min_samples=min_samples,
            metric="haversine",
        ).fit_predict(np.radians(coords))

    unique = [value for value in np.unique(labels) if value != NOISE_LABEL]
    centroids = np.array(
        [spherical_centroid(coords[labels == value, 0], coords[labels == value, 1]) for value in unique]
    ) if unique else np.empty((0, 2))

    return ClusterResult(
        labels=labels,
        method=method,
        n_clusters=len(unique),
        silhouette=silhouette_km(coords, labels),
        centroids=centroids,
        cluster_ids=np.asarray(unique, dtype=int),
        summary=cluster_summary(frame, labels, label_column=label_column),
    )


def silhouette_km(coords: np.ndarray, labels: Sequence) -> Optional[float]:
    """Silhouette score computed in kilometre space, ignoring noise points.

    Returns ``None`` when the score is undefined (fewer than two clusters, or
    too few points).
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)
    mask = labels != NOISE_LABEL
    if mask.sum() < 3:
        return None
    kept_labels = labels[mask]
    if len(np.unique(kept_labels)) < 2 or len(np.unique(kept_labels)) >= mask.sum():
        return None
    return float(silhouette_score(_km_space(coords[mask]), kept_labels))


def estimate_n_clusters(
    frame: pd.DataFrame,
    k_range: Optional[Sequence[int]] = None,
    random_seed: int = 42,
) -> int:
    """Choose k for k-means by maximising the silhouette score.

    Args:
        frame: points to cluster.
        k_range: candidate values; defaults to ``2..min(10, n_samples - 1)``.

    Returns:
        The best k, or 1 when the data is too small to score.
    """
    coords = _coordinates(frame)
    points = _km_space(coords)
    n_samples = len(coords)
    if n_samples < 3:
        return 1

    candidates = list(k_range) if k_range is not None else range(2, min(10, n_samples - 1) + 1)
    candidates = [k for k in candidates if 2 <= k < n_samples]
    if not candidates:
        return 1

    best_k, best_score = candidates[0], -np.inf
    for k in candidates:
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_seed).fit_predict(points)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(points, labels)
        if score > best_score:
            best_k, best_score = k, score
    return int(best_k)


def cluster_summary(
    frame: pd.DataFrame,
    labels: Sequence,
    label_column: Optional[str] = LABEL_COLUMN,
) -> pd.DataFrame:
    """Per-cluster size, centre, radius and (optionally) label purity.

    Returns:
        DataFrame with one row per cluster, sorted by size descending. Noise
        points are excluded.
    """
    coords = _coordinates(frame)
    labels = np.asarray(labels)
    if len(labels) != len(coords):
        raise ValueError(
            f"labels length {len(labels)} does not match frame length {len(coords)}"
        )

    rows: List[Dict[str, Any]] = []
    for value in sorted(set(labels.tolist()) - {NOISE_LABEL}):
        mask = labels == value
        lat, lon = coords[mask, 0], coords[mask, 1]
        centre_lat, centre_lon = spherical_centroid(lat, lon)
        distances = haversine_km(lat, lon, centre_lat, centre_lon)
        row: Dict[str, Any] = {
            "cluster": int(value),
            "size": int(mask.sum()),
            "centre_lat": round(centre_lat, 6),
            "centre_lon": round(centre_lon, 6),
            "mean_radius_km": round(float(np.mean(distances)), 3),
            "max_radius_km": round(float(np.max(distances)), 3),
        }
        if label_column and label_column in frame.columns:
            counts = frame.loc[mask, label_column].value_counts()
            row["dominant_label"] = str(counts.index[0])
            row["purity"] = round(float(counts.iloc[0] / counts.sum()), 3)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values("size", ascending=False).reset_index(drop=True)


def assign_to_clusters(result: ClusterResult, lat, lon) -> np.ndarray:
    """Assign new points to the nearest cluster centroid.

    Returns:
        Array of cluster ids drawn from ``result.cluster_ids``.

    Raises:
        ValueError: if the result holds no centroids.
    """
    if result.centroids.size == 0:
        raise ValueError("cluster result has no centroids to assign against")
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    distances = np.stack(
        [haversine_km(lat, lon, c_lat, c_lon) for c_lat, c_lon in result.centroids], axis=-1
    )
    nearest = np.argmin(distances, axis=-1)
    ids = result.cluster_ids
    if ids.size != len(result.centroids):
        return np.asarray(nearest, dtype=int)
    return np.asarray(ids[nearest], dtype=int)
