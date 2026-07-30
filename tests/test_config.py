from pathlib import Path

import pytest

from location_classifier.config import ENV_PREFIX, Config


def test_defaults_are_absolute_paths():
    config = Config()
    assert config.data_path.is_absolute()
    assert config.model_path.suffix == ".joblib"
    assert config.model_name in {"random_forest"}


def test_string_paths_are_coerced():
    config = Config(data_path="some/where.csv")
    assert isinstance(config.data_path, Path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0.0},
        {"test_size": 1.0},
        {"cv_folds": 1},
        {"n_anchors": 0},
        {"samples_per_class": 1},
        {"spread_km": 0.0},
    ],
)
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_from_env_reads_typed_values(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_PREFIX + "DATA_PATH", str(tmp_path / "x.csv"))
    monkeypatch.setenv(ENV_PREFIX + "RANDOM_SEED", "7")
    monkeypatch.setenv(ENV_PREFIX + "TEST_SIZE", "0.35")
    monkeypatch.setenv(ENV_PREFIX + "MODEL_NAME", "knn")
    config = Config.from_env()
    assert config.data_path == tmp_path / "x.csv"
    assert config.random_seed == 7
    assert config.test_size == pytest.approx(0.35)
    assert config.model_name == "knn"


def test_from_env_ignores_blank_values(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "MODEL_NAME", "   ")
    assert Config.from_env().model_name == Config().model_name


def test_explicit_overrides_beat_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "RANDOM_SEED", "7")
    assert Config.from_env(random_seed=99).random_seed == 99


def test_none_overrides_are_ignored(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "RANDOM_SEED", "7")
    assert Config.from_env(random_seed=None).random_seed == 7


def test_from_env_reports_a_bad_integer(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "RANDOM_SEED", "not-a-number")
    with pytest.raises(ValueError, match="RANDOM_SEED"):
        Config.from_env()


def test_from_env_reports_a_bad_float(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "TEST_SIZE", "high")
    with pytest.raises(ValueError, match="TEST_SIZE"):
        Config.from_env()


def test_with_overrides_returns_a_copy():
    config = Config()
    updated = config.with_overrides(random_seed=123, model_name=None)
    assert updated.random_seed == 123
    assert config.random_seed != 123
    assert updated.model_name == config.model_name


def test_with_overrides_rejects_unknown_options():
    with pytest.raises(ValueError, match="unknown config option"):
        Config().with_overrides(nonsense=1)


def test_ensure_directories_creates_everything(tmp_path):
    config = Config(
        data_path=tmp_path / "d" / "locations.csv",
        model_path=tmp_path / "m" / "model.joblib",
        output_dir=tmp_path / "o",
    )
    config.ensure_directories()
    assert config.data_path.parent.is_dir()
    assert config.model_path.parent.is_dir()
    assert config.output_dir.is_dir()
