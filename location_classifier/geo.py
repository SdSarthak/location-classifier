"""Spherical-geometry helpers used across the project.

Everything is vectorised with numpy so the same functions work on scalars,
1-D arrays and pandas columns without special-casing the caller.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

# Mean Earth radius (IUGG), in kilometres.
EARTH_RADIUS_KM = 6371.0088

# Length of one degree of latitude at the surface, in kilometres.
KM_PER_DEGREE_LAT = np.pi * EARTH_RADIUS_KM / 180.0

LAT_BOUNDS = (-90.0, 90.0)
LON_BOUNDS = (-180.0, 180.0)


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def validate_coordinates(lat, lon) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``lat``/``lon`` as float arrays, raising if they are unusable.

    Raises:
        ValueError: if the shapes disagree, values are not finite, or a value
            falls outside the valid latitude/longitude range.
    """
    lat_arr = _as_array(lat)
    lon_arr = _as_array(lon)
    if lat_arr.shape != lon_arr.shape:
        raise ValueError(
            f"latitude/longitude shape mismatch: {lat_arr.shape} vs {lon_arr.shape}"
        )
    if not np.all(np.isfinite(lat_arr)) or not np.all(np.isfinite(lon_arr)):
        raise ValueError("coordinates contain NaN or infinite values")
    if np.any(lat_arr < LAT_BOUNDS[0]) or np.any(lat_arr > LAT_BOUNDS[1]):
        raise ValueError(f"latitude outside {LAT_BOUNDS}")
    if np.any(lon_arr < LON_BOUNDS[0]) or np.any(lon_arr > LON_BOUNDS[1]):
        raise ValueError(f"longitude outside {LON_BOUNDS}")
    return lat_arr, lon_arr


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres between two sets of points."""
    lat1, lon1 = validate_coordinates(lat1, lon1)
    lat2, lon2 = validate_coordinates(lat2, lon2)

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2) - np.radians(lon1)

    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def pairwise_haversine_km(coords_a, coords_b) -> np.ndarray:
    """Distance matrix between two ``(n, 2)`` / ``(m, 2)`` lat-lon arrays.

    Returns:
        ndarray of shape ``(n, m)`` in kilometres.
    """
    a = np.atleast_2d(_as_array(coords_a))
    b = np.atleast_2d(_as_array(coords_b))
    if a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("coordinate arrays must have shape (n, 2) as (lat, lon)")
    return haversine_km(
        a[:, 0][:, None], a[:, 1][:, None], b[None, :, 0], b[None, :, 1]
    )


def initial_bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial compass bearing from point 1 to point 2, in degrees ``[0, 360)``."""
    lat1, lon1 = validate_coordinates(lat1, lon1)
    lat2, lon2 = validate_coordinates(lat2, lon2)

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_lambda = np.radians(lon2) - np.radians(lon1)

    y = np.sin(d_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lambda)
    return np.degrees(np.arctan2(y, x)) % 360.0


def destination_point(lat, lon, bearing_deg, distance_km) -> Tuple[np.ndarray, np.ndarray]:
    """Point reached by travelling ``distance_km`` along ``bearing_deg``."""
    lat, lon = validate_coordinates(lat, lon)
    theta = np.radians(_as_array(bearing_deg))
    delta = _as_array(distance_km) / EARTH_RADIUS_KM

    phi1, lambda1 = np.radians(lat), np.radians(lon)
    phi2 = np.arcsin(
        np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta)
    )
    lambda2 = lambda1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(phi1),
        np.cos(delta) - np.sin(phi1) * np.sin(phi2),
    )
    return np.degrees(phi2), (np.degrees(lambda2) + 540.0) % 360.0 - 180.0


def to_cartesian(lat, lon) -> np.ndarray:
    """Unit-sphere cartesian coordinates, shape ``(..., 3)``.

    Useful as model features because it removes the wrap-around discontinuity
    at the antimeridian that raw longitude suffers from.
    """
    lat, lon = validate_coordinates(lat, lon)
    phi, lam = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1
    )


def spherical_centroid(lat, lon) -> Tuple[float, float]:
    """Centroid of a set of points computed on the sphere, not in lat-lon space."""
    lat, lon = validate_coordinates(lat, lon)
    if lat.size == 0:
        raise ValueError("cannot compute a centroid of zero points")
    vectors = to_cartesian(lat.ravel(), lon.ravel())
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        # Antipodal / uniformly spread points have no meaningful centroid.
        return float(np.mean(lat)), float(np.mean(lon))
    x, y, z = mean / norm
    return float(np.degrees(np.arcsin(z))), float(np.degrees(np.arctan2(y, x)))


def bounding_box(lat, lon) -> Tuple[float, float, float, float]:
    """Axis-aligned bounds as ``(min_lat, min_lon, max_lat, max_lon)``."""
    lat, lon = validate_coordinates(lat, lon)
    if lat.size == 0:
        raise ValueError("cannot compute a bounding box of zero points")
    return float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max())


def km_per_degree_lon(lat) -> np.ndarray:
    """Kilometres covered by one degree of longitude at the given latitude."""
    return KM_PER_DEGREE_LAT * np.cos(np.radians(_as_array(lat)))


def offset_degrees(lat, north_km, east_km) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a local north/east offset in km to a lat/lon delta in degrees."""
    d_lat = _as_array(north_km) / KM_PER_DEGREE_LAT
    scale = km_per_degree_lon(lat)
    scale = np.where(np.abs(scale) < 1e-9, 1e-9, scale)
    return d_lat, _as_array(east_km) / scale


def anchors_from_labels(lat, lon, labels: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    """Per-label spherical centroids, sorted by label.

    Returns:
        ``(unique_labels, centroids)`` where ``centroids`` has shape ``(k, 2)``.
    """
    lat, lon = validate_coordinates(lat, lon)
    labels = np.asarray(list(labels))
    if labels.shape[0] != lat.ravel().shape[0]:
        raise ValueError("labels and coordinates must have the same length")
    unique = np.unique(labels)
    centroids = np.array(
        [spherical_centroid(lat[labels == value], lon[labels == value]) for value in unique]
    )
    return unique, centroids
