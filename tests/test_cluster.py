import numpy as np
import pandas as pd
import pytest

from location_classifier import cluster
from location_classifier.config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN
from location_classifier.data import MUMBAI_ZONES, make_synthetic_dataset


@pytest.fixture(scope="module")
def frame():
    return make_synthetic_dataset(samples_per_class=40, spread_km=4.0, random_seed=31)


@pytest.fixture(scope="module")
def zone_frame():
    return make_synthetic_dataset(
        samples_per_class=40, spread_km=1.5, sites=MUMBAI_ZONES, random_seed=31
    )


def test_kmeans_recovers_the_true_site_count(frame):
    result = cluster.cluster_locations(frame, method="kmeans", n_clusters=8)
    assert result.n_clusters == 8
    assert result.method == "kmeans"
    assert result.n_noise == 0
    assert result.silhouette is not None and result.silhouette > 0.9


def test_kmeans_centroids_land_near_the_real_sites(frame):
    from location_classifier.data import DEFAULT_SITES
    from location_classifier.geo import haversine_km

    result = cluster.cluster_locations(frame, method="kmeans", n_clusters=8)
    for site in DEFAULT_SITES:
        distances = haversine_km(
            result.centroids[:, 0], result.centroids[:, 1], site.latitude, site.longitude
        )
        assert distances.min() < 5.0


def test_kmeans_picks_k_automatically(frame):
    result = cluster.cluster_locations(frame, method="kmeans")
    assert result.n_clusters == 8


def test_estimate_n_clusters_respects_an_explicit_range(frame):
    assert cluster.estimate_n_clusters(frame, k_range=[2, 3, 4]) in (2, 3, 4)


def test_estimate_n_clusters_returns_one_for_tiny_input():
    tiny = pd.DataFrame({LATITUDE_COLUMN: [19.0, 19.1], LONGITUDE_COLUMN: [72.0, 72.1]})
    assert cluster.estimate_n_clusters(tiny) == 1


def test_dbscan_finds_the_dense_groups(zone_frame):
    result = cluster.cluster_locations(
        zone_frame, method="dbscan", eps_km=1.5, min_samples=5
    )
    assert result.method == "dbscan"
    assert result.n_clusters >= 4
    assert result.n_noise < len(zone_frame) // 2


def test_dbscan_marks_far_away_points_as_noise():
    frame = make_synthetic_dataset(samples_per_class=30, spread_km=1.0, random_seed=2)
    outlier = pd.DataFrame(
        {
            LATITUDE_COLUMN: [-45.0],
            LONGITUDE_COLUMN: [170.0],
            "elevation_m": [0.0],
            "avg_temp_c": [10.0],
            "population_density": [1.0],
            LABEL_COLUMN: ["nowhere"],
        }
    )
    combined = pd.concat([frame, outlier], ignore_index=True)
    result = cluster.cluster_locations(combined, method="dbscan", eps_km=3.0, min_samples=5)
    assert result.labels[-1] == cluster.NOISE_LABEL


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "nope"},
        {"method": "dbscan", "eps_km": 0.0},
        {"method": "dbscan", "min_samples": 0},
    ],
)
def test_cluster_rejects_bad_arguments(frame, kwargs):
    with pytest.raises(ValueError):
        cluster.cluster_locations(frame, **kwargs)


def test_cluster_rejects_missing_columns():
    with pytest.raises(ValueError, match="coordinate column"):
        cluster.cluster_locations(pd.DataFrame({"a": [1.0]}))


def test_cluster_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        cluster.cluster_locations(
            pd.DataFrame({LATITUDE_COLUMN: [], LONGITUDE_COLUMN: []})
        )


def test_summary_reports_size_radius_and_purity(frame):
    result = cluster.cluster_locations(frame, method="kmeans", n_clusters=8)
    summary = result.summary
    assert len(summary) == 8
    assert summary["size"].sum() == len(frame)
    assert summary["size"].is_monotonic_decreasing
    assert (summary["mean_radius_km"] <= summary["max_radius_km"]).all()
    assert (summary["purity"] > 0.9).all()


def test_summary_omits_purity_without_labels(frame):
    result = cluster.cluster_locations(frame, n_clusters=4, label_column=None)
    assert "purity" not in result.summary.columns


def test_summary_rejects_label_length_mismatch(frame):
    with pytest.raises(ValueError, match="does not match"):
        cluster.cluster_summary(frame, [0, 1, 2])


def test_describe_mentions_method_and_counts(frame):
    text = cluster.cluster_locations(frame, n_clusters=3).describe()
    assert "method: kmeans" in text
    assert "clusters: 3" in text
    assert "silhouette" in text


def test_silhouette_is_none_for_a_single_cluster(frame):
    coords = frame[[LATITUDE_COLUMN, LONGITUDE_COLUMN]].to_numpy()
    assert cluster.silhouette_km(coords, np.zeros(len(coords), dtype=int)) is None


def test_silhouette_ignores_noise_points(frame):
    coords = frame[[LATITUDE_COLUMN, LONGITUDE_COLUMN]].to_numpy()
    labels = np.where(np.arange(len(coords)) % 2 == 0, 0, 1)
    labels[:5] = cluster.NOISE_LABEL
    assert cluster.silhouette_km(coords, labels) is not None


def test_assign_to_clusters_matches_the_training_assignment(frame):
    result = cluster.cluster_locations(frame, method="kmeans", n_clusters=8)
    assigned = cluster.assign_to_clusters(
        result, frame[LATITUDE_COLUMN].to_numpy(), frame[LONGITUDE_COLUMN].to_numpy()
    )
    agreement = float(np.mean(assigned == result.labels))
    assert agreement > 0.98


def test_assign_to_clusters_needs_centroids():
    empty = cluster.ClusterResult(labels=np.array([]), method="kmeans", n_clusters=0)
    with pytest.raises(ValueError, match="no centroids"):
        cluster.assign_to_clusters(empty, 19.0, 72.0)
