import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from location_classifier.config import LATITUDE_COLUMN, LONGITUDE_COLUMN
from location_classifier.data import make_synthetic_dataset
from location_classifier.features import GeoFeatureBuilder, split_coordinates


@pytest.fixture()
def frame():
    return make_synthetic_dataset(samples_per_class=12, random_seed=17).drop(
        columns=["location"]
    )


def test_split_coordinates_from_dataframe(frame):
    lat, lon, extras = split_coordinates(frame, ["elevation_m"])
    assert lat.shape == (len(frame),)
    assert extras.shape == (len(frame), 1)


def test_split_coordinates_rejects_missing_columns():
    with pytest.raises(ValueError, match="coordinate column"):
        split_coordinates(pd.DataFrame({"a": [1.0]}))


def test_split_coordinates_rejects_unknown_feature_column(frame):
    with pytest.raises(ValueError, match="feature column"):
        split_coordinates(frame, ["nope"])


def test_split_coordinates_from_array():
    array = np.array([[19.0, 72.0, 1.0], [28.0, 77.0, 2.0]])
    lat, lon, extras = split_coordinates(array)
    assert list(lat) == [19.0, 28.0]
    assert extras.shape == (2, 1)


def test_split_coordinates_rejects_narrow_array():
    with pytest.raises(ValueError, match="at least two columns"):
        split_coordinates(np.array([[1.0], [2.0]]))


def test_transform_width_matches_feature_names(frame):
    builder = GeoFeatureBuilder(n_anchors=5).fit(frame)
    matrix = builder.transform(frame)
    assert matrix.shape[0] == len(frame)
    assert matrix.shape[1] == len(builder.get_feature_names_out())


def test_auto_detects_numeric_extra_columns(frame):
    builder = GeoFeatureBuilder(n_anchors=3).fit(frame)
    assert set(builder.extra_columns_) == {"elevation_m", "avg_temp_c", "population_density"}
    names = list(builder.get_feature_names_out())
    assert names[:2] == [LATITUDE_COLUMN, LONGITUDE_COLUMN]
    assert names[-3:] == builder.extra_columns_


def test_explicit_extra_columns_are_respected(frame):
    builder = GeoFeatureBuilder(n_anchors=2, extra_columns=["elevation_m"]).fit(frame)
    assert builder.extra_columns_ == ["elevation_m"]
    assert builder.transform(frame).shape[1] == len(builder.feature_names_)


def test_anchor_count_is_capped_by_unique_points():
    small = pd.DataFrame({LATITUDE_COLUMN: [19.0, 19.0], LONGITUDE_COLUMN: [72.0, 72.0]})
    builder = GeoFeatureBuilder(n_anchors=10).fit(small)
    assert len(builder.anchors_) == 1


def test_distance_features_are_zero_at_the_anchor(frame):
    builder = GeoFeatureBuilder(n_anchors=4, extra_columns=[]).fit(frame)
    anchor_lat, anchor_lon = builder.anchors_[0]
    point = pd.DataFrame({LATITUDE_COLUMN: [anchor_lat], LONGITUDE_COLUMN: [anchor_lon]})
    names = list(builder.get_feature_names_out())
    value = builder.transform(point)[0, names.index("dist_anchor_0_km")]
    assert value == pytest.approx(0.0, abs=1e-6)


def test_cyclical_features_are_bounded(frame):
    builder = GeoFeatureBuilder(n_anchors=3, extra_columns=[]).fit(frame)
    matrix = builder.transform(frame)
    names = list(builder.get_feature_names_out())
    for name in ("sin_lat", "cos_lat", "sin_lon", "cos_lon"):
        column = matrix[:, names.index(name)]
        assert column.min() >= -1.0 - 1e-12 and column.max() <= 1.0 + 1e-12


def test_cartesian_can_be_disabled(frame):
    with_cart = GeoFeatureBuilder(n_anchors=3, extra_columns=[], include_cartesian=True).fit(frame)
    without = GeoFeatureBuilder(n_anchors=3, extra_columns=[], include_cartesian=False).fit(frame)
    assert with_cart.transform(frame).shape[1] - without.transform(frame).shape[1] == 3
    assert "cart_x" not in without.get_feature_names_out()


def test_transform_is_deterministic(frame):
    a = GeoFeatureBuilder(n_anchors=6, random_state=3).fit(frame).transform(frame)
    b = GeoFeatureBuilder(n_anchors=6, random_state=3).fit(frame).transform(frame)
    assert np.allclose(a, b)


def test_transform_before_fit_raises(frame):
    with pytest.raises(NotFittedError):
        GeoFeatureBuilder().transform(frame)


def test_fit_rejects_invalid_n_anchors(frame):
    with pytest.raises(ValueError, match="n_anchors"):
        GeoFeatureBuilder(n_anchors=0).fit(frame)


def test_fit_rejects_out_of_range_coordinates():
    bad = pd.DataFrame({LATITUDE_COLUMN: [200.0], LONGITUDE_COLUMN: [0.0]})
    with pytest.raises(ValueError, match="latitude"):
        GeoFeatureBuilder().fit(bad)


def test_extra_medians_are_recorded_at_fit_time(frame):
    builder = GeoFeatureBuilder(n_anchors=3).fit(frame)
    assert set(builder.extra_medians_) == set(builder.extra_columns_)
    assert builder.extra_medians_["avg_temp_c"] == pytest.approx(
        frame["avg_temp_c"].median()
    )


def test_extra_median_falls_back_to_zero_for_an_all_nan_column():
    frame = pd.DataFrame(
        {
            LATITUDE_COLUMN: [19.0, 19.1],
            LONGITUDE_COLUMN: [72.0, 72.1],
            "broken": [np.nan, np.nan],
        }
    )
    builder = GeoFeatureBuilder(n_anchors=2).fit(frame)
    assert builder.extra_medians_["broken"] == 0.0


def test_extra_column_nans_become_zero():
    frame = pd.DataFrame(
        {
            LATITUDE_COLUMN: [19.0, 19.1],
            LONGITUDE_COLUMN: [72.0, 72.1],
            "elevation_m": [10.0, np.nan],
        }
    )
    builder = GeoFeatureBuilder(n_anchors=2).fit(frame)
    matrix = builder.transform(frame)
    assert np.isfinite(matrix).all()
