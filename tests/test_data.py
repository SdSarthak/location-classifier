import numpy as np
import pandas as pd
import pytest

from location_classifier import data
from location_classifier.config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN
from location_classifier.geo import haversine_km


def test_synthetic_dataset_shape_and_columns():
    frame = data.make_synthetic_dataset(samples_per_class=20, random_seed=0)
    assert len(frame) == 20 * len(data.DEFAULT_SITES)
    for column in (LATITUDE_COLUMN, LONGITUDE_COLUMN, LABEL_COLUMN, *data.AUXILIARY_COLUMNS):
        assert column in frame.columns
    assert frame[LABEL_COLUMN].nunique() == len(data.DEFAULT_SITES)


def test_synthetic_dataset_is_reproducible():
    left = data.make_synthetic_dataset(samples_per_class=15, random_seed=7)
    right = data.make_synthetic_dataset(samples_per_class=15, random_seed=7)
    pd.testing.assert_frame_equal(left, right)


def test_synthetic_dataset_differs_across_seeds():
    left = data.make_synthetic_dataset(samples_per_class=15, random_seed=1)
    right = data.make_synthetic_dataset(samples_per_class=15, random_seed=2)
    assert not np.allclose(left[LATITUDE_COLUMN], right[LATITUDE_COLUMN])


def test_synthetic_points_sit_near_their_site():
    frame = data.make_synthetic_dataset(samples_per_class=50, spread_km=5.0, random_seed=3)
    site = data.DEFAULT_SITES[0]
    subset = frame[frame[LABEL_COLUMN] == site.name]
    distances = haversine_km(
        subset[LATITUDE_COLUMN].to_numpy(),
        subset[LONGITUDE_COLUMN].to_numpy(),
        site.latitude,
        site.longitude,
    )
    # Gaussian scatter in two axes: the mean radius is ~1.25 sigma.
    assert distances.mean() < 12.0
    assert distances.max() < 40.0


def test_synthetic_coordinates_are_valid():
    frame = data.make_synthetic_dataset(samples_per_class=10, random_seed=5)
    assert frame[LATITUDE_COLUMN].between(-90, 90).all()
    assert frame[LONGITUDE_COLUMN].between(-180, 180).all()
    assert (frame["population_density"] >= 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples_per_class": 1},
        {"spread_km": 0.0},
        {"noise_scale": -1.0},
        {"sites": []},
    ],
)
def test_synthetic_dataset_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        data.make_synthetic_dataset(**kwargs)


def test_validate_rejects_missing_columns():
    with pytest.raises(data.DatasetError, match="missing required column"):
        data.validate_dataset(pd.DataFrame({"lat": [1.0], "lon": [2.0], LABEL_COLUMN: ["x"]}))


def test_validate_rejects_empty_frame():
    with pytest.raises(data.DatasetError, match="empty"):
        data.validate_dataset(pd.DataFrame())


def test_validate_drops_invalid_rows_by_default():
    frame = pd.DataFrame(
        {
            LATITUDE_COLUMN: [19.0, 999.0, 20.0],
            LONGITUDE_COLUMN: [72.0, 72.0, 400.0],
            LABEL_COLUMN: ["a", "b", "c"],
        }
    )
    cleaned = data.validate_dataset(frame)
    assert len(cleaned) == 1
    assert cleaned.loc[0, LABEL_COLUMN] == "a"


def test_validate_can_raise_instead_of_dropping():
    frame = pd.DataFrame(
        {LATITUDE_COLUMN: [19.0, 999.0], LONGITUDE_COLUMN: [72.0, 72.0], LABEL_COLUMN: ["a", "b"]}
    )
    with pytest.raises(data.DatasetError, match="invalid row"):
        data.validate_dataset(frame, drop_invalid=False)


def test_validate_coerces_numeric_strings():
    frame = pd.DataFrame(
        {LATITUDE_COLUMN: ["19.0"], LONGITUDE_COLUMN: ["72.5"], LABEL_COLUMN: ["a"]}
    )
    cleaned = data.validate_dataset(frame)
    assert cleaned.loc[0, LATITUDE_COLUMN] == pytest.approx(19.0)


def test_validate_raises_when_everything_is_invalid():
    frame = pd.DataFrame(
        {LATITUDE_COLUMN: ["abc"], LONGITUDE_COLUMN: ["def"], LABEL_COLUMN: ["a"]}
    )
    with pytest.raises(data.DatasetError, match="no valid rows"):
        data.validate_dataset(frame)


def test_validate_allows_unlabelled_data():
    frame = pd.DataFrame({LATITUDE_COLUMN: [19.0], LONGITUDE_COLUMN: [72.0]})
    cleaned = data.validate_dataset(frame, label_column=None)
    assert len(cleaned) == 1


def test_save_and_load_round_trip(tmp_path):
    frame = data.make_synthetic_dataset(samples_per_class=8, random_seed=11)
    path = data.save_dataset(frame, tmp_path / "nested" / "locations.csv")
    assert path.exists()
    loaded = data.load_dataset(path)
    assert len(loaded) == len(frame)
    assert set(loaded[LABEL_COLUMN]) == set(frame[LABEL_COLUMN])


def test_load_missing_file_mentions_the_generator(tmp_path):
    with pytest.raises(FileNotFoundError, match="make-data"):
        data.load_dataset(tmp_path / "nope.csv")


def test_feature_columns_excludes_coordinates_and_label():
    frame = data.make_synthetic_dataset(samples_per_class=5, random_seed=2)
    columns = data.feature_columns(frame)
    assert set(columns) == set(data.AUXILIARY_COLUMNS)


def test_describe_dataset_reports_counts():
    frame = data.make_synthetic_dataset(samples_per_class=6, random_seed=4)
    text = data.describe_dataset(frame)
    assert "rows: 48" in text
    assert "Mumbai: 6" in text


def test_stratified_split_preserves_all_classes():
    frame = data.make_synthetic_dataset(samples_per_class=20, random_seed=9)
    train, test = data.stratified_split(frame, test_size=0.25, random_seed=9)
    assert len(train) + len(test) == len(frame)
    assert set(train[LABEL_COLUMN]) == set(frame[LABEL_COLUMN])
    assert set(test[LABEL_COLUMN]) == set(frame[LABEL_COLUMN])


def test_stratified_split_falls_back_for_singleton_classes():
    frame = pd.DataFrame(
        {
            LATITUDE_COLUMN: [19.0, 19.1, 28.0, 12.0],
            LONGITUDE_COLUMN: [72.0, 72.1, 77.0, 77.5],
            LABEL_COLUMN: ["a", "a", "b", "c"],
        }
    )
    train, test = data.stratified_split(frame, test_size=0.5, random_seed=1)
    assert len(train) + len(test) == 4
