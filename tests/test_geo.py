import numpy as np
import pytest

from location_classifier import geo

MUMBAI = (19.0760, 72.8777)
DELHI = (28.6139, 77.2090)


def test_haversine_matches_known_distance():
    # Mumbai -> Delhi is ~1153 km great-circle.
    distance = geo.haversine_km(*MUMBAI, *DELHI)
    assert distance == pytest.approx(1153.0, abs=10.0)


def test_haversine_is_zero_for_identical_points():
    assert geo.haversine_km(*MUMBAI, *MUMBAI) == pytest.approx(0.0, abs=1e-9)


def test_haversine_is_symmetric():
    forward = geo.haversine_km(*MUMBAI, *DELHI)
    backward = geo.haversine_km(*DELHI, *MUMBAI)
    assert forward == pytest.approx(backward)


def test_haversine_antipodal_is_half_circumference():
    distance = geo.haversine_km(0.0, 0.0, 0.0, 180.0)
    assert distance == pytest.approx(np.pi * geo.EARTH_RADIUS_KM, rel=1e-9)


def test_haversine_is_vectorised():
    lats = np.array([19.0760, 28.6139])
    lons = np.array([72.8777, 77.2090])
    distances = geo.haversine_km(lats, lons, 12.9716, 77.5946)
    assert distances.shape == (2,)
    assert distances[0] > 0 and distances[1] > 0


def test_haversine_rejects_out_of_range_latitude():
    with pytest.raises(ValueError, match="latitude"):
        geo.haversine_km(95.0, 0.0, 0.0, 0.0)


def test_haversine_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        geo.haversine_km(np.nan, 0.0, 0.0, 0.0)


def test_pairwise_distance_matrix_shape_and_diagonal():
    coords = np.array([MUMBAI, DELHI, (12.9716, 77.5946)])
    matrix = geo.pairwise_haversine_km(coords, coords)
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-9)
    assert np.allclose(matrix, matrix.T, atol=1e-9)


def test_pairwise_rejects_wrong_width():
    with pytest.raises(ValueError, match="shape"):
        geo.pairwise_haversine_km(np.zeros((3, 3)), np.zeros((2, 2)))


def test_bearing_due_north_and_east():
    assert geo.initial_bearing_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert geo.initial_bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0, abs=1e-6)
    assert geo.initial_bearing_deg(10.0, 0.0, 0.0, 0.0) == pytest.approx(180.0, abs=1e-6)


def test_bearing_is_always_in_range():
    bearings = geo.initial_bearing_deg(
        np.array([0.0, 10.0, -30.0]), np.array([0.0, 20.0, 100.0]), 5.0, -5.0
    )
    assert np.all(bearings >= 0.0) and np.all(bearings < 360.0)


def test_destination_point_round_trips_with_haversine():
    lat, lon = geo.destination_point(*MUMBAI, 45.0, 100.0)
    assert geo.haversine_km(*MUMBAI, lat, lon) == pytest.approx(100.0, rel=1e-6)
    assert geo.initial_bearing_deg(*MUMBAI, lat, lon) == pytest.approx(45.0, abs=1e-6)


def test_destination_longitude_wraps_into_range():
    _, lon = geo.destination_point(0.0, 179.0, 90.0, 500.0)
    assert -180.0 <= float(lon) <= 180.0


def test_to_cartesian_produces_unit_vectors():
    vectors = geo.to_cartesian(np.array([0.0, 45.0, -80.0]), np.array([0.0, 90.0, 170.0]))
    assert vectors.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_spherical_centroid_of_symmetric_points():
    lat, lon = geo.spherical_centroid([-1.0, 1.0], [-1.0, 1.0])
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert lon == pytest.approx(0.0, abs=1e-9)


def test_spherical_centroid_handles_antimeridian():
    lat, lon = geo.spherical_centroid([0.0, 0.0], [179.0, -179.0])
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert abs(lon) == pytest.approx(180.0, abs=1e-6)


def test_spherical_centroid_rejects_empty_input():
    with pytest.raises(ValueError, match="zero points"):
        geo.spherical_centroid([], [])


def test_bounding_box():
    box = geo.bounding_box([1.0, -2.0, 3.0], [10.0, 20.0, -30.0])
    assert box == (-2.0, -30.0, 3.0, 20.0)


def test_km_per_degree_lon_shrinks_towards_the_poles():
    assert geo.km_per_degree_lon(0.0) == pytest.approx(geo.KM_PER_DEGREE_LAT)
    assert float(geo.km_per_degree_lon(60.0)) < float(geo.km_per_degree_lon(0.0))
    assert float(geo.km_per_degree_lon(90.0)) == pytest.approx(0.0, abs=1e-9)


def test_offset_degrees_produces_requested_distance():
    d_lat, d_lon = geo.offset_degrees(19.0760, 10.0, 0.0)
    moved = geo.haversine_km(19.0760, 72.8777, 19.0760 + float(d_lat), 72.8777 + float(d_lon))
    assert moved == pytest.approx(10.0, rel=1e-3)


def test_anchors_from_labels_returns_sorted_centroids():
    lat = np.array([0.0, 0.2, 10.0, 10.2])
    lon = np.array([0.0, 0.2, 10.0, 10.2])
    labels, centroids = geo.anchors_from_labels(lat, lon, ["b", "b", "a", "a"])
    assert list(labels) == ["a", "b"]
    assert centroids.shape == (2, 2)
    assert centroids[0][0] == pytest.approx(10.1, abs=0.01)
    assert centroids[1][0] == pytest.approx(0.1, abs=0.01)


def test_anchors_from_labels_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        geo.anchors_from_labels([0.0, 1.0], [0.0, 1.0], ["a"])
