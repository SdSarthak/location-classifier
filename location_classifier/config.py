"""Runtime configuration.

Every tunable lives here so nothing downstream has to hardcode a path. Values
can be overridden by environment variables (see ``.env.example``) or by keyword
arguments, which is what the CLI does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict

ENV_PREFIX = "LOC_CLF_"

#: Column names the rest of the package agrees on.
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"
LABEL_COLUMN = "location"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """Paths and hyper-parameters for the whole pipeline."""

    data_path: Path = _project_root() / "data" / "locations.csv"
    model_path: Path = _project_root() / "models" / "location_classifier.joblib"
    output_dir: Path = _project_root() / "outputs"

    model_name: str = "random_forest"
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    #: Number of learned reference points used for distance features.
    n_anchors: int = 12
    #: Points generated per class by the synthetic dataset builder.
    samples_per_class: int = 120
    #: Standard deviation, in km, of the synthetic scatter around each site.
    spread_km: float = 6.0

    def __post_init__(self) -> None:
        # ``frozen=True`` blocks plain assignment, so coerce via object.__setattr__.
        for name in ("data_path", "model_path", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        if self.n_anchors < 1:
            raise ValueError(f"n_anchors must be >= 1, got {self.n_anchors}")
        if self.samples_per_class < 2:
            raise ValueError(
                f"samples_per_class must be >= 2, got {self.samples_per_class}"
            )
        if self.spread_km <= 0:
            raise ValueError(f"spread_km must be > 0, got {self.spread_km}")

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """Build a config from ``LOC_CLF_*`` environment variables.

        Explicit ``overrides`` win over the environment, which wins over the
        dataclass defaults. Unset or empty environment variables are ignored.
        """
        values: Dict[str, Any] = {}
        for field in fields(cls):
            raw = os.environ.get(ENV_PREFIX + field.name.upper())
            if raw is None or raw.strip() == "":
                continue
            values[field.name] = _coerce(field.name, field.type, raw.strip())
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with the non-``None`` overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(clean) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown config option(s): {sorted(unknown)}")
        return replace(self, **clean)

    def ensure_directories(self) -> None:
        """Create the parent directories the pipeline writes into."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def _coerce(name: str, annotation: Any, raw: str) -> Any:
    """Cast an environment string to the type declared on the dataclass field."""
    text = str(annotation)
    if "Path" in text:
        return Path(raw).expanduser()
    if "int" in text:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{ENV_PREFIX}{name.upper()} must be an integer") from exc
    if "float" in text:
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"{ENV_PREFIX}{name.upper()} must be a number") from exc
    return raw
