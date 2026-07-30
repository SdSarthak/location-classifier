import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from location_classifier import visualize  # noqa: E402
from location_classifier.cluster import cluster_locations  # noqa: E402
from location_classifier.config import LABEL_COLUMN, LATITUDE_COLUMN  # noqa: E402
from location_classifier.data import make_synthetic_dataset  # noqa: E402
from location_classifier.evaluate import evaluate_predictions  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    return make_synthetic_dataset(samples_per_class=15, random_seed=41)


@pytest.fixture(scope="module")
def metrics():
    truth = ["a"] * 6 + ["b"] * 6 + ["c"] * 6
    predicted = ["a"] * 5 + ["b"] + ["b"] * 6 + ["c"] * 5 + ["a"]
    return evaluate_predictions(truth, predicted)


def _written(path):
    return path.exists() and path.stat().st_size > 0


def test_plot_locations_writes_a_file(frame, tmp_path):
    path = visualize.plot_locations(frame, path=tmp_path / "map.png")
    assert _written(path)


def test_plot_locations_returns_axes_without_a_path(frame):
    ax = visualize.plot_locations(frame)
    assert hasattr(ax, "scatter")
    assert len(ax.collections) == frame[LABEL_COLUMN].nunique()
    matplotlib.pyplot.close(ax.figure)


def test_plot_locations_handles_unlabelled_data(frame, tmp_path):
    path = visualize.plot_locations(
        frame.drop(columns=[LABEL_COLUMN]), label_column=None, path=tmp_path / "plain.png"
    )
    assert _written(path)


def test_plot_locations_rejects_missing_coordinates(tmp_path):
    with pytest.raises(ValueError, match=LATITUDE_COLUMN):
        visualize.plot_locations(pd.DataFrame({"x": [1.0]}), path=tmp_path / "bad.png")


def test_plot_clusters_writes_a_file(frame, tmp_path):
    result = cluster_locations(frame, n_clusters=8)
    path = visualize.plot_clusters(
        frame, result.labels, centroids=result.centroids, path=tmp_path / "clusters.png"
    )
    assert _written(path)


def test_plot_clusters_rejects_label_mismatch(frame, tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        visualize.plot_clusters(frame, [0, 1], path=tmp_path / "bad.png")


def test_plot_confusion_matrix_writes_a_file(metrics, tmp_path):
    path = visualize.plot_confusion_matrix(metrics, path=tmp_path / "cm.png")
    assert _written(path)


def test_plot_confusion_matrix_supports_raw_counts(metrics, tmp_path):
    path = visualize.plot_confusion_matrix(
        metrics, path=tmp_path / "cm_raw.png", normalize=False
    )
    assert _written(path)


def test_plot_feature_importances_writes_a_file(tmp_path):
    importances = pd.DataFrame(
        {"feature": ["lat", "lon", "dist"], "importance": [0.5, 0.3, 0.2]}
    )
    path = visualize.plot_feature_importances(importances, path=tmp_path / "fi.png")
    assert _written(path)


def test_plot_feature_importances_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="no feature importances"):
        visualize.plot_feature_importances(pd.DataFrame(), path=tmp_path / "fi.png")


def test_plot_feature_importances_requires_the_right_columns(tmp_path):
    with pytest.raises(ValueError, match="must have columns"):
        visualize.plot_feature_importances(
            pd.DataFrame({"a": [1]}), path=tmp_path / "fi.png"
        )


def test_plot_model_comparison_writes_a_file(tmp_path):
    board = pd.DataFrame({"model": ["knn", "rf"], "accuracy": [0.8, 0.9]})
    path = visualize.plot_model_comparison(board, path=tmp_path / "cmp.png")
    assert _written(path)


def test_plot_model_comparison_rejects_unknown_metric(tmp_path):
    board = pd.DataFrame({"model": ["knn"], "accuracy": [0.8]})
    with pytest.raises(ValueError, match="no 'macro_f1' column"):
        visualize.plot_model_comparison(board, metric="macro_f1", path=tmp_path / "c.png")


def test_plot_model_comparison_rejects_empty_leaderboard(tmp_path):
    with pytest.raises(ValueError, match="no models"):
        visualize.plot_model_comparison(pd.DataFrame(), path=tmp_path / "c.png")


def test_palette_is_long_enough_for_many_classes():
    assert len(visualize._palette(25)) == 25
    assert np.asarray(visualize._palette(3)).shape[1] == 4
