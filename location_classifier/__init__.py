"""Location Classifier — geospatial classification, clustering and evaluation."""

from __future__ import annotations

from .config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, Config
from .data import (
    DEFAULT_SITES,
    DatasetError,
    Site,
    describe_dataset,
    load_dataset,
    make_synthetic_dataset,
    save_dataset,
    stratified_split,
    validate_dataset,
)
from .features import GeoFeatureBuilder

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DEFAULT_SITES",
    "DatasetError",
    "GeoFeatureBuilder",
    "LABEL_COLUMN",
    "LATITUDE_COLUMN",
    "LONGITUDE_COLUMN",
    "Site",
    "__version__",
    "describe_dataset",
    "load_dataset",
    "make_synthetic_dataset",
    "save_dataset",
    "stratified_split",
    "validate_dataset",
]
