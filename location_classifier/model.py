"""Model construction, training, persistence and inference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, Config
from .data import feature_columns, stratified_split, validate_dataset
from .evaluate import evaluate_predictions
from .features import GeoFeatureBuilder

#: Available estimators, keyed by the name accepted on the command line.
MODEL_REGISTRY: Dict[str, Callable[[int], BaseEstimator]] = {
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=1, random_state=seed, n_jobs=-1
    ),
    "extra_trees": lambda seed: ExtraTreesClassifier(
        n_estimators=300, random_state=seed, n_jobs=-1
    ),
    "gradient_boosting": lambda seed: HistGradientBoostingClassifier(random_state=seed),
    "logistic_regression": lambda seed: LogisticRegression(
        max_iter=2000, random_state=seed
    ),
    "knn": lambda seed: KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "svm": lambda seed: SVC(kernel="rbf", C=10.0, probability=True, random_state=seed),
}


def available_models() -> List[str]:
    """Sorted list of model names accepted by :func:`build_pipeline`."""
    return sorted(MODEL_REGISTRY)


def build_pipeline(
    model_name: str = "random_forest",
    extra_columns: Optional[Sequence[str]] = None,
    n_anchors: int = 12,
    random_seed: int = 42,
) -> Pipeline:
    """Assemble the feature builder, scaler and estimator into one pipeline.

    Raises:
        KeyError: if ``model_name`` is not registered.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model '{model_name}'; choose one of {available_models()}"
        )
    return Pipeline(
        [
            (
                "features",
                GeoFeatureBuilder(
                    n_anchors=n_anchors,
                    extra_columns=extra_columns,
                    random_state=random_seed,
                ),
            ),
            ("scaler", StandardScaler()),
            ("classifier", MODEL_REGISTRY[model_name](random_seed)),
        ]
    )


@dataclass
class TrainingResult:
    """Everything produced by a training run."""

    pipeline: Pipeline
    model_name: str
    classes: List[str]
    metrics: Dict[str, Any]
    cv_scores: List[float] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    extra_columns: List[str] = field(default_factory=list)
    random_seed: int = 42
    trained_at: str = ""

    @property
    def accuracy(self) -> float:
        return float(self.metrics.get("accuracy", float("nan")))

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else float("nan")

    def metadata(self) -> Dict[str, Any]:
        """JSON-serialisable summary stored next to the model file."""
        return {
            "model_name": self.model_name,
            "classes": self.classes,
            "extra_columns": self.extra_columns,
            "random_seed": self.random_seed,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "trained_at": self.trained_at,
            "cv_scores": self.cv_scores,
            "cv_mean": None if not self.cv_scores else self.cv_mean,
            "metrics": self.metrics,
        }


def train_model(
    frame: pd.DataFrame,
    config: Optional[Config] = None,
    model_name: Optional[str] = None,
    label_column: str = LABEL_COLUMN,
    run_cv: bool = True,
) -> TrainingResult:
    """Train a classifier and score it on a held-out split.

    Args:
        frame: labelled dataset.
        config: pipeline settings; defaults to :class:`Config` defaults.
        model_name: overrides ``config.model_name``.
        label_column: column holding the class label.
        run_cv: also run stratified cross-validation on the full dataset.

    Raises:
        ValueError: if fewer than two classes are present.
    """
    config = config or Config()
    model_name = model_name or config.model_name

    frame = validate_dataset(frame, label_column=label_column)
    classes = sorted(frame[label_column].unique())
    if len(classes) < 2:
        raise ValueError(
            f"need at least 2 classes to train, found {len(classes)}: {classes}"
        )

    extras = feature_columns(frame, label_column=label_column)
    train, test = stratified_split(
        frame,
        label_column=label_column,
        test_size=config.test_size,
        random_seed=config.random_seed,
    )

    pipeline = build_pipeline(
        model_name=model_name,
        extra_columns=extras,
        n_anchors=config.n_anchors,
        random_seed=config.random_seed,
    )
    X_train = train.drop(columns=[label_column])
    X_test = test.drop(columns=[label_column])
    pipeline.fit(X_train, train[label_column])

    predictions = pipeline.predict(X_test)
    proba = predict_proba_or_none(pipeline, X_test)
    metrics = evaluate_predictions(
        test[label_column],
        predictions,
        y_proba=proba,
        classes=list(pipeline.named_steps["classifier"].classes_),
    )

    cv_scores: List[float] = []
    if run_cv:
        cv_scores = cross_validate_pipeline(
            frame, config=config, model_name=model_name, label_column=label_column
        )

    return TrainingResult(
        pipeline=pipeline,
        model_name=model_name,
        classes=[str(value) for value in classes],
        metrics=metrics,
        cv_scores=cv_scores,
        n_train=len(train),
        n_test=len(test),
        extra_columns=extras,
        random_seed=config.random_seed,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def cross_validate_pipeline(
    frame: pd.DataFrame,
    config: Optional[Config] = None,
    model_name: Optional[str] = None,
    label_column: str = LABEL_COLUMN,
    scoring: str = "accuracy",
) -> List[float]:
    """Stratified cross-validation, with the fold count clamped to class sizes.

    Returns an empty list when the rarest class has fewer than two members,
    which makes stratified folds impossible.
    """
    config = config or Config()
    model_name = model_name or config.model_name

    labels = frame[label_column]
    smallest = int(labels.value_counts().min())
    folds = min(config.cv_folds, smallest)
    if folds < 2:
        return []

    pipeline = build_pipeline(
        model_name=model_name,
        extra_columns=feature_columns(frame, label_column=label_column),
        n_anchors=config.n_anchors,
        random_seed=config.random_seed,
    )
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=config.random_seed
    )
    scores = cross_val_score(
        pipeline,
        frame.drop(columns=[label_column]),
        labels,
        cv=splitter,
        scoring=scoring,
    )
    return [float(score) for score in scores]


def compare_models(
    frame: pd.DataFrame,
    config: Optional[Config] = None,
    model_names: Optional[Sequence[str]] = None,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Train every candidate model and return a leaderboard sorted by accuracy."""
    config = config or Config()
    rows = []
    for name in model_names or available_models():
        result = train_model(
            frame, config=config, model_name=name, label_column=label_column, run_cv=False
        )
        rows.append(
            {
                "model": name,
                "accuracy": result.accuracy,
                "balanced_accuracy": result.metrics["balanced_accuracy"],
                "macro_f1": result.metrics["macro_f1"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )


def predict_proba_or_none(pipeline: Pipeline, X) -> Optional[np.ndarray]:
    """Class probabilities when the estimator supports them, otherwise ``None``."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        return np.asarray(pipeline.predict_proba(X), dtype=float)
    except (AttributeError, NotImplementedError):
        return None


def points_frame(pipeline: Pipeline, lat, lon) -> pd.DataFrame:
    """Build a model-ready frame from bare coordinates.

    Auxiliary columns the pipeline was trained on (elevation, temperature, …)
    are filled with the training median rather than zero, so a coordinate-only
    prediction is not dragged off by an out-of-distribution feature value.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    if lat.shape != lon.shape:
        raise ValueError(f"lat/lon length mismatch: {lat.shape} vs {lon.shape}")

    frame = pd.DataFrame({LATITUDE_COLUMN: lat, LONGITUDE_COLUMN: lon})
    return fill_missing_features(pipeline, frame)


def fill_missing_features(pipeline: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Add any auxiliary column the pipeline expects, using training medians."""
    builder = pipeline.named_steps.get("features")
    expected = list(getattr(builder, "extra_columns_", []) or [])
    medians = getattr(builder, "extra_medians_", {}) or {}

    filled = frame.copy()
    for column in expected:
        if column not in filled.columns:
            filled[column] = medians.get(column, 0.0)
    return filled


def predict_locations(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    top_k: int = 1,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Predict labels for new points.

    Args:
        pipeline: a fitted pipeline.
        frame: points to classify; a label column, if present, is ignored.
        top_k: number of ranked alternatives to include per row.

    Returns:
        The input frame plus ``predicted_location``, ``confidence`` and, when
        ``top_k > 1``, ``rank_{i}_location`` / ``rank_{i}_confidence`` columns.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    features = validate_dataset(frame, label_column=None)
    if label_column in features.columns:
        features = features.drop(columns=[label_column])

    output = features.copy()
    output["predicted_location"] = pipeline.predict(features)

    proba = predict_proba_or_none(pipeline, features)
    if proba is None:
        output["confidence"] = np.nan
        return output

    classes = np.asarray(pipeline.classes_)
    output["confidence"] = proba.max(axis=1)

    k = min(top_k, len(classes))
    order = np.argsort(-proba, axis=1)[:, :k]
    for rank in range(k):
        column = order[:, rank]
        output[f"rank_{rank + 1}_location"] = classes[column]
        output[f"rank_{rank + 1}_confidence"] = proba[np.arange(len(proba)), column]
    return output


def feature_importances(pipeline: Pipeline, limit: int = 15) -> pd.DataFrame:
    """Importance per engineered feature, when the estimator exposes one.

    Tree ensembles use ``feature_importances_``; linear models fall back to the
    mean absolute coefficient. Returns an empty frame for models with neither.
    """
    builder = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    names = list(builder.get_feature_names_out())

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_, dtype=float)).mean(axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if len(values) != len(names):
        return pd.DataFrame(columns=["feature", "importance"])
    frame = pd.DataFrame({"feature": names, "importance": values})
    return frame.sort_values("importance", ascending=False).head(limit).reset_index(drop=True)


def save_model(result: TrainingResult, path: Path | str) -> Path:
    """Persist the fitted pipeline plus metadata; writes a sibling JSON file."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": result.pipeline, "metadata": result.metadata()}, path)
    path.with_suffix(".json").write_text(
        json.dumps(result.metadata(), indent=2, default=str), encoding="utf-8"
    )
    return path


def load_model(path: Path | str) -> tuple[Pipeline, Dict[str, Any]]:
    """Load a saved bundle.

    Raises:
        FileNotFoundError: if the model file is missing.
        ValueError: if the file is not a bundle written by :func:`save_model`.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"model not found: {path}. Train one with "
            f"`python -m location_classifier train`."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"{path} is not a location-classifier model bundle")
    return bundle["pipeline"], bundle.get("metadata", {})
