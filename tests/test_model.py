import numpy as np
import pandas as pd
import pytest

from location_classifier import model
from location_classifier.config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, Config
from location_classifier.data import MUMBAI_ZONES, make_synthetic_dataset

CONFIG = Config(n_anchors=6, cv_folds=3, samples_per_class=40)


@pytest.fixture(scope="module")
def easy_frame():
    """Well-separated metros — any sane model should nail this."""
    return make_synthetic_dataset(samples_per_class=40, random_seed=21)


@pytest.fixture(scope="module")
def hard_frame():
    """Overlapping city zones — accuracy should be high but not perfect."""
    return make_synthetic_dataset(
        samples_per_class=60, spread_km=2.5, sites=MUMBAI_ZONES, random_seed=21
    )


@pytest.fixture(scope="module")
def trained(easy_frame):
    return model.train_model(easy_frame, config=CONFIG, run_cv=False)


def test_available_models_are_all_buildable():
    names = model.available_models()
    assert "random_forest" in names
    for name in names:
        assert model.build_pipeline(name, extra_columns=[]) is not None


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(KeyError, match="unknown model"):
        model.build_pipeline("does_not_exist")


def test_pipeline_has_the_expected_steps():
    pipeline = model.build_pipeline("knn", extra_columns=[])
    assert list(pipeline.named_steps) == ["features", "scaler", "classifier"]


def test_training_separates_well_separated_classes(trained):
    assert trained.accuracy > 0.95
    assert trained.metrics["macro_f1"] > 0.95
    assert len(trained.classes) == 8


def test_training_records_split_sizes(trained, easy_frame):
    assert trained.n_train + trained.n_test == len(easy_frame)
    assert trained.n_test == pytest.approx(len(easy_frame) * CONFIG.test_size, abs=2)


def test_training_uses_auxiliary_columns(trained):
    assert set(trained.extra_columns) == {"elevation_m", "avg_temp_c", "population_density"}


def test_training_is_reproducible(easy_frame):
    first = model.train_model(easy_frame, config=CONFIG, run_cv=False)
    second = model.train_model(easy_frame, config=CONFIG, run_cv=False)
    assert first.accuracy == pytest.approx(second.accuracy)


def test_overlapping_classes_are_harder_but_learnable(hard_frame):
    result = model.train_model(hard_frame, config=CONFIG, run_cv=False)
    assert 0.75 < result.accuracy < 1.0
    assert result.metrics["top_2_accuracy"] > result.accuracy


def test_training_requires_at_least_two_classes():
    frame = pd.DataFrame(
        {
            LATITUDE_COLUMN: [19.0, 19.1, 19.2],
            LONGITUDE_COLUMN: [72.0, 72.1, 72.2],
            LABEL_COLUMN: ["only", "only", "only"],
        }
    )
    with pytest.raises(ValueError, match="at least 2 classes"):
        model.train_model(frame, config=CONFIG)


def test_cross_validation_returns_one_score_per_fold(easy_frame):
    scores = model.cross_validate_pipeline(easy_frame, config=CONFIG)
    assert len(scores) == CONFIG.cv_folds
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_cross_validation_skips_when_classes_are_too_small():
    frame = pd.DataFrame(
        {
            LATITUDE_COLUMN: [19.0, 28.0],
            LONGITUDE_COLUMN: [72.0, 77.0],
            LABEL_COLUMN: ["a", "b"],
        }
    )
    assert model.cross_validate_pipeline(frame, config=CONFIG) == []


def test_metadata_is_json_serialisable(trained):
    import json

    payload = json.loads(json.dumps(trained.metadata(), default=str))
    assert payload["model_name"] == "random_forest"
    assert len(payload["classes"]) == 8


def test_save_and_load_round_trip_preserves_predictions(trained, easy_frame, tmp_path):
    path = model.save_model(trained, tmp_path / "model.joblib")
    assert path.exists()
    assert path.with_suffix(".json").exists()

    pipeline, metadata = model.load_model(path)
    features = easy_frame.drop(columns=[LABEL_COLUMN])
    assert list(pipeline.predict(features)) == list(trained.pipeline.predict(features))
    assert metadata["model_name"] == trained.model_name


def test_load_missing_model_points_at_the_train_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        model.load_model(tmp_path / "absent.joblib")


def test_load_rejects_a_foreign_file(tmp_path):
    import joblib

    path = tmp_path / "foreign.joblib"
    joblib.dump([1, 2, 3], path)
    with pytest.raises(ValueError, match="not a location-classifier model bundle"):
        model.load_model(path)


def test_predict_locations_adds_ranked_columns(trained, easy_frame):
    sample = easy_frame.drop(columns=[LABEL_COLUMN]).head(10)
    output = model.predict_locations(trained.pipeline, sample, top_k=3)
    assert len(output) == 10
    assert "predicted_location" in output.columns
    for rank in (1, 2, 3):
        assert f"rank_{rank}_location" in output.columns
    assert (output["rank_1_location"] == output["predicted_location"]).all()
    assert (output["rank_1_confidence"] >= output["rank_2_confidence"]).all()


def test_predict_locations_confidence_is_a_probability(trained, easy_frame):
    output = model.predict_locations(
        trained.pipeline, easy_frame.drop(columns=[LABEL_COLUMN]).head(20)
    )
    assert output["confidence"].between(0.0, 1.0).all()


def test_predict_locations_ignores_an_existing_label_column(trained, easy_frame):
    output = model.predict_locations(trained.pipeline, easy_frame.head(5))
    assert LABEL_COLUMN not in output.columns
    assert len(output) == 5


def test_predict_locations_rejects_bad_top_k(trained, easy_frame):
    with pytest.raises(ValueError, match="top_k"):
        model.predict_locations(trained.pipeline, easy_frame.head(2), top_k=0)


def test_points_frame_imputes_training_medians(trained, easy_frame):
    points = model.points_frame(trained.pipeline, 19.076, 72.877)
    assert list(points.columns[:2]) == [LATITUDE_COLUMN, LONGITUDE_COLUMN]
    for column in trained.extra_columns:
        expected = easy_frame[column].median()
        # Medians come from the training split, so allow a little slack.
        assert points.loc[0, column] == pytest.approx(expected, rel=0.5)


def test_points_frame_enables_coordinate_only_prediction(trained):
    points = model.points_frame(trained.pipeline, [19.076, 28.6139], [72.877, 77.2090])
    predicted = list(trained.pipeline.predict(points))
    assert predicted == ["Mumbai", "Delhi"]


def test_points_frame_rejects_mismatched_lengths(trained):
    with pytest.raises(ValueError, match="mismatch"):
        model.points_frame(trained.pipeline, [19.0, 20.0], [72.0])


def test_fill_missing_features_leaves_present_columns_alone(trained, easy_frame):
    sample = easy_frame.drop(columns=[LABEL_COLUMN]).head(3)
    filled = model.fill_missing_features(trained.pipeline, sample)
    pd.testing.assert_frame_equal(filled, sample)


def test_feature_importances_are_named_and_ranked(trained):
    frame = model.feature_importances(trained.pipeline, limit=5)
    assert len(frame) == 5
    assert list(frame.columns) == ["feature", "importance"]
    assert frame["importance"].is_monotonic_decreasing
    known = set(trained.pipeline.named_steps["features"].get_feature_names_out())
    assert set(frame["feature"]).issubset(known)


def test_feature_importances_fall_back_to_coefficients(easy_frame):
    result = model.train_model(
        easy_frame, config=CONFIG, model_name="logistic_regression", run_cv=False
    )
    frame = model.feature_importances(result.pipeline)
    assert not frame.empty
    assert np.all(frame["importance"] >= 0)


def test_feature_importances_empty_for_models_without_any(easy_frame):
    result = model.train_model(easy_frame, config=CONFIG, model_name="knn", run_cv=False)
    assert model.feature_importances(result.pipeline).empty


def test_compare_models_returns_a_sorted_leaderboard(easy_frame):
    board = model.compare_models(
        easy_frame, config=CONFIG, model_names=["knn", "logistic_regression"]
    )
    assert list(board["model"]) and len(board) == 2
    assert board["accuracy"].is_monotonic_decreasing
