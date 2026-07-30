"""Location Classifier — geospatial classification, clustering and evaluation.

The common entry points are re-exported here::

    from location_classifier import make_synthetic_dataset, train_model

Anything not listed in ``__all__`` lives in the submodules: ``geo``, ``data``,
``features``, ``model``, ``evaluate``, ``cluster``, ``visualize``, ``cli``.
"""

from __future__ import annotations

from .cluster import ClusterResult, cluster_locations, estimate_n_clusters
from .config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, Config
from .data import (
    AUXILIARY_COLUMNS,
    DEFAULT_SITES,
    MUMBAI_ZONES,
    SITE_COLLECTIONS,
    DatasetError,
    Site,
    describe_dataset,
    get_sites,
    load_dataset,
    make_synthetic_dataset,
    save_dataset,
    stratified_split,
    validate_dataset,
)
from .evaluate import evaluate_predictions, format_metrics
from .features import GeoFeatureBuilder
from .model import (
    MODEL_REGISTRY,
    TrainingResult,
    available_models,
    build_pipeline,
    compare_models,
    feature_importances,
    load_model,
    points_frame,
    predict_locations,
    save_model,
    train_model,
)

__version__ = "0.1.0"

__all__ = [
    "AUXILIARY_COLUMNS",
    "ClusterResult",
    "Config",
    "DEFAULT_SITES",
    "DatasetError",
    "GeoFeatureBuilder",
    "LABEL_COLUMN",
    "LATITUDE_COLUMN",
    "LONGITUDE_COLUMN",
    "MODEL_REGISTRY",
    "MUMBAI_ZONES",
    "SITE_COLLECTIONS",
    "Site",
    "TrainingResult",
    "__version__",
    "available_models",
    "build_pipeline",
    "cluster_locations",
    "compare_models",
    "describe_dataset",
    "estimate_n_clusters",
    "evaluate_predictions",
    "feature_importances",
    "format_metrics",
    "get_sites",
    "load_dataset",
    "load_model",
    "make_synthetic_dataset",
    "points_frame",
    "predict_locations",
    "save_dataset",
    "save_model",
    "stratified_split",
    "train_model",
    "validate_dataset",
]
