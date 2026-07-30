import importlib

import pytest

import location_classifier as lc

SUBMODULES = [
    "cli",
    "cluster",
    "config",
    "data",
    "evaluate",
    "features",
    "geo",
    "model",
    "visualize",
]


def test_version_is_exposed():
    assert lc.__version__.count(".") == 2


@pytest.mark.parametrize("name", SUBMODULES)
def test_every_submodule_imports(name):
    assert importlib.import_module(f"location_classifier.{name}") is not None


@pytest.mark.parametrize("name", lc.__all__)
def test_every_exported_name_resolves(name):
    assert getattr(lc, name) is not None


def test_all_is_sorted_and_unique():
    assert lc.__all__ == sorted(lc.__all__)
    assert len(lc.__all__) == len(set(lc.__all__))


def test_end_to_end_through_the_top_level_api(tmp_path):
    frame = lc.make_synthetic_dataset(samples_per_class=20, random_seed=13)
    result = lc.train_model(frame, config=lc.Config(n_anchors=5), run_cv=False)
    assert result.accuracy > 0.9

    path = lc.save_model(result, tmp_path / "m.joblib")
    pipeline, metadata = lc.load_model(path)
    assert metadata["classes"] == result.classes

    points = lc.points_frame(pipeline, [19.076], [72.877])
    predictions = lc.predict_locations(pipeline, points)
    assert predictions.loc[0, "predicted_location"] == "Mumbai"

    clusters = lc.cluster_locations(frame, n_clusters=8)
    assert clusters.n_clusters == 8


def test_main_module_is_runnable():
    module = importlib.import_module("location_classifier.__main__")
    assert callable(module.main)
