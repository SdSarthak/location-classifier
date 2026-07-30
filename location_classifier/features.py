"""Geospatial feature engineering as a scikit-learn transformer.

Raw latitude/longitude is a poor model input: it is discontinuous at the
antimeridian and its scale changes with latitude. ``GeoFeatureBuilder`` expands
a coordinate pair into a representation tree- and distance-based models can
actually use — unit-sphere cartesian coordinates, cyclical encodings, and
great-circle distances to reference anchors learned from the training set.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted

from .config import LATITUDE_COLUMN, LONGITUDE_COLUMN
from .geo import (
    haversine_km,
    initial_bearing_deg,
    spherical_centroid,
    to_cartesian,
    validate_coordinates,
)


def split_coordinates(X, extra_columns: Optional[Sequence[str]] = None):
    """Split an input table into ``(lat, lon, extras)``.

    Accepts a DataFrame with named coordinate columns, or a 2-D array whose
    first two columns are latitude and longitude.

    Returns:
        ``(lat, lon, extras)`` where ``extras`` is a ``(n, k)`` float array.
    """
    if isinstance(X, pd.DataFrame):
        missing = [c for c in (LATITUDE_COLUMN, LONGITUDE_COLUMN) if c not in X.columns]
        if missing:
            raise ValueError(f"input is missing coordinate column(s): {missing}")
        lat = X[LATITUDE_COLUMN].to_numpy(dtype=float)
        lon = X[LONGITUDE_COLUMN].to_numpy(dtype=float)
        if extra_columns:
            unknown = [c for c in extra_columns if c not in X.columns]
            if unknown:
                raise ValueError(f"input is missing feature column(s): {unknown}")
            extras = X[list(extra_columns)].to_numpy(dtype=float)
        else:
            extras = np.empty((len(X), 0), dtype=float)
        return lat, lon, extras

    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            "array input must be 2-D with at least two columns (latitude, longitude)"
        )
    expected = len(extra_columns) if extra_columns else array.shape[1] - 2
    if array.shape[1] - 2 != expected:
        raise ValueError(
            f"expected {expected + 2} columns, got {array.shape[1]}; "
            "column order must match the training data"
        )
    return array[:, 0], array[:, 1], array[:, 2:]


class GeoFeatureBuilder(BaseEstimator, TransformerMixin):
    """Expand coordinates into geodesy-aware numeric features.

    Args:
        n_anchors: number of reference points learned with k-means on the
            training coordinates; one distance feature is emitted per anchor.
        extra_columns: additional numeric columns to pass through. ``None``
            means "auto-detect every numeric non-coordinate column at fit time".
        include_cartesian: emit unit-sphere x/y/z features.
        random_state: seed for the anchor k-means, keeping output reproducible.

    Attributes:
        anchors_: ``(k, 2)`` array of learned anchor coordinates.
        centroid_: spherical centroid of the training coordinates.
        extra_columns_: columns actually passed through.
        feature_names_: names of the produced columns, in order.
    """

    def __init__(
        self,
        n_anchors: int = 12,
        extra_columns: Optional[Sequence[str]] = None,
        include_cartesian: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_anchors = n_anchors
        self.extra_columns = extra_columns
        self.include_cartesian = include_cartesian
        self.random_state = random_state

    # -- sklearn API ----------------------------------------------------------

    def fit(self, X, y=None) -> "GeoFeatureBuilder":
        if self.n_anchors < 1:
            raise ValueError(f"n_anchors must be >= 1, got {self.n_anchors}")

        extra_columns = self.extra_columns
        if extra_columns is None and isinstance(X, pd.DataFrame):
            extra_columns = [
                column
                for column in X.columns
                if column not in (LATITUDE_COLUMN, LONGITUDE_COLUMN)
                and pd.api.types.is_numeric_dtype(X[column])
            ]
        self.extra_columns_ = list(extra_columns or [])

        lat, lon, _ = split_coordinates(X, self.extra_columns_)
        validate_coordinates(lat, lon)
        if lat.size == 0:
            raise ValueError("cannot fit on an empty dataset")

        self.centroid_ = spherical_centroid(lat, lon)
        self.anchors_ = self._learn_anchors(lat, lon)
        self.feature_names_ = self._build_feature_names()
        self.n_features_in_ = 2 + len(self.extra_columns_)
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "anchors_")
        lat, lon, extras = split_coordinates(X, self.extra_columns_)
        validate_coordinates(lat, lon)

        phi, lam = np.radians(lat), np.radians(lon)
        blocks: List[np.ndarray] = [
            lat[:, None],
            lon[:, None],
            np.sin(phi)[:, None],
            np.cos(phi)[:, None],
            np.sin(lam)[:, None],
            np.cos(lam)[:, None],
        ]

        if self.include_cartesian:
            blocks.append(to_cartesian(lat, lon))

        centre_lat, centre_lon = self.centroid_
        blocks.append(haversine_km(lat, lon, centre_lat, centre_lon)[:, None])
        bearing = np.radians(initial_bearing_deg(lat, lon, centre_lat, centre_lon))
        blocks.append(np.sin(bearing)[:, None])
        blocks.append(np.cos(bearing)[:, None])

        for anchor_lat, anchor_lon in self.anchors_:
            blocks.append(haversine_km(lat, lon, anchor_lat, anchor_lon)[:, None])

        if extras.shape[1]:
            blocks.append(np.nan_to_num(extras, nan=0.0, posinf=0.0, neginf=0.0))

        matrix = np.hstack(blocks)
        if matrix.shape[1] != len(self.feature_names_):
            raise RuntimeError(
                f"feature width mismatch: built {matrix.shape[1]}, "
                f"named {len(self.feature_names_)}"
            )
        return matrix

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)

    # -- internals ------------------------------------------------------------

    def _learn_anchors(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Place anchors at k-means centres of the training points.

        Clustering happens in cartesian space so the centres are not skewed by
        longitude convergence near the poles; the centres are then projected
        back onto the sphere.
        """
        points = np.column_stack([lat, lon])
        unique_points = np.unique(points, axis=0)
        k = int(min(self.n_anchors, len(unique_points)))
        if k <= 1:
            return unique_points[:1] if len(unique_points) else points[:1]

        cartesian = to_cartesian(unique_points[:, 0], unique_points[:, 1])
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        assignments = kmeans.fit_predict(cartesian)

        anchors = [
            spherical_centroid(
                unique_points[assignments == index, 0], unique_points[assignments == index, 1]
            )
            for index in range(k)
            if np.any(assignments == index)
        ]
        return np.asarray(anchors, dtype=float)

    def _build_feature_names(self) -> List[str]:
        names = [
            "latitude",
            "longitude",
            "sin_lat",
            "cos_lat",
            "sin_lon",
            "cos_lon",
        ]
        if self.include_cartesian:
            names += ["cart_x", "cart_y", "cart_z"]
        names += ["dist_centroid_km", "sin_bearing_centroid", "cos_bearing_centroid"]
        names += [f"dist_anchor_{i}_km" for i in range(len(self.anchors_))]
        names += list(self.extra_columns_)
        return names
