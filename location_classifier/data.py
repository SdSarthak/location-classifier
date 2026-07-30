"""Dataset creation, loading and validation.

The project ships a deterministic synthetic generator so the whole pipeline is
runnable — and testable — without downloading anything. Real CSV exports load
through the same validation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import LABEL_COLUMN, LATITUDE_COLUMN, LONGITUDE_COLUMN
from .geo import LAT_BOUNDS, LON_BOUNDS, offset_degrees

#: Auxiliary numeric columns the synthetic generator produces. Real datasets may
#: carry any subset of these, or none at all.
AUXILIARY_COLUMNS: Tuple[str, ...] = ("elevation_m", "avg_temp_c", "population_density")

REQUIRED_COLUMNS: Tuple[str, ...] = (LATITUDE_COLUMN, LONGITUDE_COLUMN)


@dataclass(frozen=True)
class Site:
    """A reference place the synthetic generator scatters points around."""

    name: str
    latitude: float
    longitude: float
    elevation_m: float
    avg_temp_c: float
    population_density: float


#: Real coordinates for eight Indian metros, with roughly accurate elevation,
#: annual mean temperature and population density (people per km^2).
DEFAULT_SITES: Tuple[Site, ...] = (
    Site("Mumbai", 19.0760, 72.8777, 14.0, 27.2, 20_694.0),
    Site("Delhi", 28.6139, 77.2090, 216.0, 25.1, 11_320.0),
    Site("Bengaluru", 12.9716, 77.5946, 920.0, 24.0, 11_371.0),
    Site("Kolkata", 22.5726, 88.3639, 9.0, 26.8, 24_306.0),
    Site("Chennai", 13.0827, 80.2707, 6.7, 28.6, 26_553.0),
    Site("Hyderabad", 17.3850, 78.4867, 542.0, 26.6, 18_480.0),
    Site("Pune", 18.5204, 73.8567, 560.0, 24.6, 5_751.0),
    Site("Ahmedabad", 23.0225, 72.5714, 53.0, 27.5, 11_900.0),
)

#: Neighbourhoods inside a single metro. These sit 5-15 km apart, so with a
#: realistic scatter the classes genuinely overlap — unlike the metro set, this
#: one is not linearly separable and produces an interesting confusion matrix.
MUMBAI_ZONES: Tuple[Site, ...] = (
    Site("Colaba", 18.9067, 72.8147, 11.0, 27.6, 22_100.0),
    Site("Dadar", 19.0176, 72.8562, 10.0, 27.4, 30_500.0),
    Site("Bandra", 19.0596, 72.8295, 12.0, 27.3, 24_800.0),
    Site("Andheri", 19.1197, 72.8468, 15.0, 27.1, 27_200.0),
    Site("Powai", 19.1197, 72.9050, 45.0, 26.8, 12_400.0),
    Site("Borivali", 19.2307, 72.8567, 18.0, 26.9, 19_600.0),
    Site("Thane", 19.2183, 72.9781, 7.0, 27.0, 15_300.0),
    Site("Vashi", 19.0770, 72.9986, 6.0, 27.2, 14_100.0),
)

#: Site collections selectable from the command line.
SITE_COLLECTIONS: dict = {"metros": DEFAULT_SITES, "zones": MUMBAI_ZONES}


def get_sites(name: str) -> Tuple[Site, ...]:
    """Look up a built-in site collection by name.

    Raises:
        KeyError: if the collection does not exist.
    """
    try:
        return SITE_COLLECTIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown site collection '{name}'; choose one of "
            f"{sorted(SITE_COLLECTIONS)}"
        ) from exc


class DatasetError(ValueError):
    """Raised when a dataset is missing columns or holds unusable values."""


def make_synthetic_dataset(
    samples_per_class: int = 120,
    spread_km: float = 6.0,
    sites: Sequence[Site] = DEFAULT_SITES,
    random_seed: int = 42,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """Generate a labelled point cloud around each site.

    Points are scattered with a Gaussian in the local north/east plane and then
    converted to degrees, so ``spread_km`` means the same thing at every
    latitude. Auxiliary features are drawn from per-site distributions, which
    gives a classifier something to learn beyond raw coordinates.

    Args:
        samples_per_class: points generated per site.
        spread_km: standard deviation of the scatter, in kilometres.
        sites: reference sites to sample around.
        random_seed: seed making the output reproducible.
        noise_scale: multiplier on the auxiliary-feature noise; raise it to make
            the classification problem harder.

    Returns:
        DataFrame with latitude, longitude, the auxiliary columns and a label.
    """
    if samples_per_class < 2:
        raise ValueError(f"samples_per_class must be >= 2, got {samples_per_class}")
    if spread_km <= 0:
        raise ValueError(f"spread_km must be > 0, got {spread_km}")
    if not sites:
        raise ValueError("at least one site is required")
    if noise_scale < 0:
        raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")

    rng = np.random.default_rng(random_seed)
    frames: List[pd.DataFrame] = []

    for site in sites:
        north_km = rng.normal(0.0, spread_km, samples_per_class)
        east_km = rng.normal(0.0, spread_km, samples_per_class)
        d_lat, d_lon = offset_degrees(site.latitude, north_km, east_km)

        frames.append(
            pd.DataFrame(
                {
                    LATITUDE_COLUMN: np.clip(site.latitude + d_lat, *LAT_BOUNDS),
                    LONGITUDE_COLUMN: (site.longitude + d_lon + 180.0) % 360.0 - 180.0,
                    "elevation_m": site.elevation_m
                    + rng.normal(0.0, 25.0 * noise_scale, samples_per_class),
                    "avg_temp_c": site.avg_temp_c
                    + rng.normal(0.0, 1.5 * noise_scale, samples_per_class),
                    "population_density": np.clip(
                        site.population_density
                        + rng.normal(0.0, 2_500.0 * noise_scale, samples_per_class),
                        0.0,
                        None,
                    ),
                    LABEL_COLUMN: site.name,
                }
            )
        )

    frame = pd.concat(frames, ignore_index=True)
    # Shuffle so downstream splits do not see the data grouped by class.
    return frame.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def validate_dataset(
    frame: pd.DataFrame,
    label_column: Optional[str] = LABEL_COLUMN,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Check schema and coordinate ranges, returning a cleaned copy.

    Args:
        frame: raw data.
        label_column: label column to require, or ``None`` for unlabelled data.
        drop_invalid: drop rows with unusable coordinates instead of raising.

    Raises:
        DatasetError: on missing columns, an empty frame, or — when
            ``drop_invalid`` is false — any invalid coordinate.
    """
    if frame is None or len(frame) == 0:
        raise DatasetError("dataset is empty")

    required = list(REQUIRED_COLUMNS)
    if label_column:
        required.append(label_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"missing required column(s): {missing}; found {list(frame.columns)}"
        )

    clean = frame.copy()
    for column in REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    valid = (
        clean[LATITUDE_COLUMN].between(*LAT_BOUNDS)
        & clean[LONGITUDE_COLUMN].between(*LON_BOUNDS)
        & clean[LATITUDE_COLUMN].notna()
        & clean[LONGITUDE_COLUMN].notna()
    )
    if label_column:
        valid &= clean[label_column].notna()

    if not valid.all():
        if not drop_invalid:
            bad = list(clean.index[~valid][:10])
            raise DatasetError(f"{int((~valid).sum())} invalid row(s), e.g. index {bad}")
        clean = clean.loc[valid]

    if len(clean) == 0:
        raise DatasetError("no valid rows remain after validation")

    if label_column:
        clean[label_column] = clean[label_column].astype(str)
    return clean.reset_index(drop=True)


def load_dataset(
    path: Path | str,
    label_column: Optional[str] = LABEL_COLUMN,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Read a CSV from disk and validate it.

    Raises:
        FileNotFoundError: if the path does not exist.
        DatasetError: if the file cannot be parsed or fails validation.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}. Generate one with "
            f"`python -m location_classifier make-data`."
        )
    try:
        frame = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise DatasetError(f"could not parse {path}: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"{path} is empty") from exc
    return validate_dataset(frame, label_column=label_column, drop_invalid=drop_invalid)


def save_dataset(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write a dataset to CSV, creating parent directories as needed."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def feature_columns(frame: pd.DataFrame, label_column: Optional[str] = LABEL_COLUMN) -> List[str]:
    """Numeric non-coordinate columns usable as extra model inputs."""
    skip = set(REQUIRED_COLUMNS) | ({label_column} if label_column else set())
    return [
        column
        for column in frame.columns
        if column not in skip and pd.api.types.is_numeric_dtype(frame[column])
    ]


def describe_dataset(frame: pd.DataFrame, label_column: Optional[str] = LABEL_COLUMN) -> str:
    """Human-readable summary used by the CLI."""
    lines = [f"rows: {len(frame)}", f"columns: {', '.join(map(str, frame.columns))}"]
    lines.append(
        "bounds: lat [{:.4f}, {:.4f}], lon [{:.4f}, {:.4f}]".format(
            frame[LATITUDE_COLUMN].min(),
            frame[LATITUDE_COLUMN].max(),
            frame[LONGITUDE_COLUMN].min(),
            frame[LONGITUDE_COLUMN].max(),
        )
    )
    if label_column and label_column in frame.columns:
        counts = frame[label_column].value_counts().sort_index()
        lines.append(f"classes: {len(counts)}")
        lines.extend(f"  {name}: {count}" for name, count in counts.items())
    return "\n".join(lines)


def stratified_split(
    frame: pd.DataFrame,
    label_column: str = LABEL_COLUMN,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split that degrades gracefully.

    Falls back to an unstratified split when any class is too small to appear on
    both sides, which is common with hand-collected location data.
    """
    from sklearn.model_selection import train_test_split

    counts = frame[label_column].value_counts()
    stratify: Optional[Iterable] = frame[label_column] if counts.min() >= 2 else None
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
