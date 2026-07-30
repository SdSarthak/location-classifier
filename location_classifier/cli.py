"""Command-line interface.

Run ``python -m location_classifier --help`` (or ``location-classifier --help``
once installed) for the full list of commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN, Config
from .data import (
    DatasetError,
    SITE_COLLECTIONS,
    describe_dataset,
    get_sites,
    load_dataset,
    make_synthetic_dataset,
    save_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="location-classifier",
        description="Train, evaluate and apply geospatial location classifiers.",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed override")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="directory for figures and reports"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_data = subparsers.add_parser(
        "make-data", help="generate a reproducible synthetic dataset"
    )
    make_data.add_argument(
        "--sites",
        default="metros",
        choices=sorted(SITE_COLLECTIONS),
        help="built-in site collection to sample around",
    )
    make_data.add_argument("--samples-per-class", type=int, default=None)
    make_data.add_argument("--spread-km", type=float, default=None)
    make_data.add_argument(
        "--noise-scale", type=float, default=1.0, help="multiplier on auxiliary noise"
    )
    make_data.add_argument("--output", type=Path, default=None, help="CSV to write")
    make_data.set_defaults(func=_cmd_make_data)

    info = subparsers.add_parser("info", help="summarise a dataset")
    info.add_argument("--data", type=Path, default=None)
    info.set_defaults(func=_cmd_info)

    train = subparsers.add_parser("train", help="train and persist a classifier")
    train.add_argument("--data", type=Path, default=None)
    train.add_argument("--model", default=None, help="estimator name")
    train.add_argument("--model-path", type=Path, default=None, help="where to save")
    train.add_argument("--n-anchors", type=int, default=None)
    train.add_argument("--test-size", type=float, default=None)
    train.add_argument("--cv-folds", type=int, default=None)
    train.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    train.add_argument("--plots", action="store_true", help="write figures to the output dir")
    train.set_defaults(func=_cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="score a saved model on a dataset")
    evaluate.add_argument("--data", type=Path, default=None)
    evaluate.add_argument("--model-path", type=Path, default=None)
    evaluate.add_argument("--plots", action="store_true")
    evaluate.add_argument("--json", action="store_true", help="print metrics as JSON")
    evaluate.set_defaults(func=_cmd_evaluate)

    predict = subparsers.add_parser("predict", help="classify new points")
    predict.add_argument("--model-path", type=Path, default=None)
    predict.add_argument("--data", type=Path, default=None, help="CSV of points to classify")
    predict.add_argument("--lat", type=float, default=None, help="single-point latitude")
    predict.add_argument("--lon", type=float, default=None, help="single-point longitude")
    predict.add_argument("--top-k", type=int, default=3)
    predict.add_argument("--output", type=Path, default=None, help="CSV to write")
    predict.set_defaults(func=_cmd_predict)

    cluster = subparsers.add_parser("cluster", help="group points without labels")
    cluster.add_argument("--data", type=Path, default=None)
    cluster.add_argument("--method", default="kmeans", choices=["kmeans", "dbscan"])
    cluster.add_argument("--n-clusters", type=int, default=None)
    cluster.add_argument("--eps-km", type=float, default=3.0)
    cluster.add_argument("--min-samples", type=int, default=5)
    cluster.add_argument("--plot", action="store_true")
    cluster.set_defaults(func=_cmd_cluster)

    compare = subparsers.add_parser("compare", help="benchmark every registered model")
    compare.add_argument("--data", type=Path, default=None)
    compare.add_argument("--plot", action="store_true")
    compare.set_defaults(func=_cmd_compare)

    return parser


def _config_from(args: argparse.Namespace) -> Config:
    """Build the config from the environment plus whatever the user passed."""
    return Config.from_env(
        random_seed=getattr(args, "seed", None),
        output_dir=getattr(args, "output_dir", None),
        data_path=getattr(args, "data", None),
        model_path=getattr(args, "model_path", None),
        model_name=getattr(args, "model", None),
        n_anchors=getattr(args, "n_anchors", None),
        test_size=getattr(args, "test_size", None),
        cv_folds=getattr(args, "cv_folds", None),
        samples_per_class=getattr(args, "samples_per_class", None),
        spread_km=getattr(args, "spread_km", None),
    )


# -- commands ----------------------------------------------------------------


def _cmd_make_data(args: argparse.Namespace, config: Config) -> int:
    frame = make_synthetic_dataset(
        samples_per_class=config.samples_per_class,
        spread_km=config.spread_km,
        sites=get_sites(args.sites),
        random_seed=config.random_seed,
        noise_scale=args.noise_scale,
    )
    path = save_dataset(frame, args.output or config.data_path)
    print(f"wrote {len(frame)} rows to {path}")
    print(describe_dataset(frame))
    return 0


def _cmd_info(args: argparse.Namespace, config: Config) -> int:
    print(describe_dataset(load_dataset(config.data_path)))
    return 0


def _cmd_train(args: argparse.Namespace, config: Config) -> int:
    from .evaluate import format_metrics
    from .model import feature_importances, save_model, train_model

    frame = load_dataset(config.data_path)
    result = train_model(frame, config=config, run_cv=not args.no_cv)

    print(f"model: {result.model_name}")
    print(f"train/test: {result.n_train}/{result.n_test}")
    if result.cv_scores:
        scores = ", ".join(f"{score:.4f}" for score in result.cv_scores)
        print(f"cv accuracy: {result.cv_mean:.4f} ({scores})")
    print()
    print(format_metrics(result.metrics, title="Held-out evaluation"))

    path = save_model(result, config.model_path)
    print(f"\nsaved model to {path}")

    if args.plots:
        _write_training_plots(config, frame, result, feature_importances(result.pipeline))
    return 0


def _write_training_plots(config: Config, frame, result, importances) -> None:
    from . import visualize

    config.output_dir.mkdir(parents=True, exist_ok=True)
    written = [
        visualize.plot_locations(
            frame, path=config.output_dir / "dataset.png", title="Training data"
        ),
        visualize.plot_confusion_matrix(
            result.metrics, path=config.output_dir / "confusion_matrix.png"
        ),
    ]
    if not importances.empty:
        written.append(
            visualize.plot_feature_importances(
                importances, path=config.output_dir / "feature_importance.png"
            )
        )
    for path in written:
        print(f"wrote {path}")


def _cmd_evaluate(args: argparse.Namespace, config: Config) -> int:
    from .evaluate import evaluate_predictions, format_metrics
    from .model import load_model, predict_proba_or_none

    frame = load_dataset(config.data_path)
    pipeline, metadata = load_model(config.model_path)

    features = frame.drop(columns=[LABEL_COLUMN])
    metrics = evaluate_predictions(
        frame[LABEL_COLUMN],
        pipeline.predict(features),
        y_proba=predict_proba_or_none(pipeline, features),
        classes=list(pipeline.classes_),
    )

    if args.json:
        print(json.dumps(metrics, indent=2, default=str))
    else:
        print(f"model: {metadata.get('model_name', 'unknown')}")
        print(f"trained at: {metadata.get('trained_at', 'unknown')}")
        print()
        print(format_metrics(metrics, title=f"Evaluation on {config.data_path.name}"))

    if args.plots:
        from . import visualize

        config.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"wrote "
            f"{visualize.plot_confusion_matrix(metrics, path=config.output_dir / 'evaluation_confusion.png')}"
        )
    return 0


def _cmd_predict(args: argparse.Namespace, config: Config) -> int:
    from .model import load_model, points_frame, predict_locations

    pipeline, _ = load_model(config.model_path)

    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            raise ValueError("--lat and --lon must be given together")
        points = points_frame(pipeline, args.lat, args.lon)
    else:
        points = load_dataset(config.data_path, label_column=None)

    output = predict_locations(pipeline, points, top_k=args.top_k)

    if args.output:
        path = save_dataset(output, args.output)
        print(f"wrote {len(output)} predictions to {path}")
    else:
        columns = [LATITUDE_COLUMN, LONGITUDE_COLUMN, "predicted_location", "confidence"]
        columns += [c for c in output.columns if c.startswith("rank_")]
        with pd.option_context("display.width", 160, "display.max_columns", 30):
            print(output[columns].head(50).to_string(index=False))
        if len(output) > 50:
            print(f"... {len(output) - 50} more rows (use --output to write them all)")
    return 0


def _cmd_cluster(args: argparse.Namespace, config: Config) -> int:
    from .cluster import cluster_locations

    frame = load_dataset(config.data_path, label_column=None)
    label_column = LABEL_COLUMN if LABEL_COLUMN in frame.columns else None
    result = cluster_locations(
        frame,
        method=args.method,
        n_clusters=args.n_clusters,
        eps_km=args.eps_km,
        min_samples=args.min_samples,
        random_seed=config.random_seed,
        label_column=label_column,
    )
    print(result.describe())

    if args.plot:
        from . import visualize

        config.output_dir.mkdir(parents=True, exist_ok=True)
        path = visualize.plot_clusters(
            frame,
            result.labels,
            centroids=result.centroids,
            path=config.output_dir / "clusters.png",
            title=f"{args.method} clusters",
        )
        print(f"wrote {path}")
    return 0


def _cmd_compare(args: argparse.Namespace, config: Config) -> int:
    from .model import compare_models

    frame = load_dataset(config.data_path)
    board = compare_models(frame, config=config)
    print(board.to_string(index=False))

    if args.plot:
        from . import visualize

        config.output_dir.mkdir(parents=True, exist_ok=True)
        path = visualize.plot_model_comparison(
            board, path=config.output_dir / "model_comparison.png"
        )
        print(f"wrote {path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code instead of raising."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = _config_from(args)
        return int(args.func(args, config))
    except (DatasetError, FileNotFoundError, ValueError, KeyError) as exc:
        message = exc.args[0] if exc.args else str(exc)
        print(f"error: {message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
