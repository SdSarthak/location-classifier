# Location Classifier

Classify a geographic point into a named place, and discover those places when
no labels exist. Latitude and longitude are poor model inputs on their own, so
the package expands each coordinate pair into geodesy-aware features — unit
sphere cartesian coordinates, cyclical encodings, and great-circle distances to
reference anchors learned from the training set — before handing them to a
scikit-learn estimator.

Everything runs offline: a deterministic synthetic dataset generator ships with
the package, so `pip install` to trained model takes about a minute with no
downloads.

```bash
pip install -r requirements.txt
python -m location_classifier make-data --sites zones
python -m location_classifier train --plots
python -m location_classifier predict --lat 19.076 --lon 72.877
```

## What it does

- **Feature engineering** (`features.py`) — `GeoFeatureBuilder` is a scikit-learn
  transformer that turns `(latitude, longitude)` plus any numeric side channels
  into ~25 features: sphere cartesian x/y/z, sin/cos of both angles, distance
  and bearing to the data centroid, and one great-circle distance per learned
  anchor. Anchors come from k-means run in cartesian space, so they are not
  skewed by longitude convergence near the poles.
- **Training** (`model.py`) — six registered estimators behind one pipeline,
  stratified splits, cross-validation with the fold count clamped to the rarest
  class, joblib persistence with a self-describing JSON sidecar.
- **Evaluation** (`evaluate.py`) — accuracy, balanced accuracy, macro/weighted
  F1, Cohen's kappa, top-k accuracy, per-class breakdown, and a ranked list of
  the most-confused label pairs.
- **Clustering** (`cluster.py`) — k-means and DBSCAN in a metric space measured
  in kilometres, so `eps_km` and cluster radii mean the same thing anywhere on
  the globe. Reports size, centre, radius and label purity per cluster.
- **Plots** (`visualize.py`) — scatter maps, cluster maps, confusion heatmaps,
  feature-importance and model-comparison bars. Picks a headless backend
  automatically, so the CLI works over SSH.
- **Geodesy** (`geo.py`) — haversine distance, initial bearing, destination
  point, spherical centroid, and local km-to-degree offsets, all vectorised.

## Install

```bash
git clone https://github.com/SdSarthak/location-classifier.git
cd location-classifier
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or install the package itself, which adds a `location-classifier` command:

```bash
pip install -e .
```

Python 3.9+.

## Getting data

**Option A — generate it (no download).** Two built-in site collections ship
with the package:

| Collection | Sites | Separation | Difficulty |
| --- | --- | --- | --- |
| `metros` | 8 Indian metros (Mumbai, Delhi, Bengaluru, …) | hundreds of km | trivially separable |
| `zones` | 8 Mumbai neighbourhoods (Colaba, Dadar, Bandra, …) | 5–15 km | genuinely overlapping |

```bash
python -m location_classifier make-data --sites zones --samples-per-class 150 --spread-km 2.5
```

Points are scattered with a Gaussian in the local north/east plane and then
converted to degrees, so `--spread-km` means the same thing at every latitude.
Each site also emits `elevation_m`, `avg_temp_c` and `population_density` drawn
from per-site distributions. Output is fully determined by `--seed`.

**Option B — bring your own CSV.** Any CSV with these columns works:

| Column | Required | Notes |
| --- | --- | --- |
| `latitude` | yes | decimal degrees, −90 to 90 |
| `longitude` | yes | decimal degrees, −180 to 180 |
| `location` | yes for training | the class label |
| anything else numeric | no | auto-detected and used as extra features |

Rows with unparseable or out-of-range coordinates are dropped with a count
reported; pass `drop_invalid=False` to `load_dataset` to make them fatal
instead. Good public sources are OpenStreetMap extracts via
[Geofabrik](https://download.geofabrik.de/), the
[GeoNames](https://www.geonames.org/) place dumps, or any GPS trace exported to
CSV. Datasets are gitignored — keep them in `data/`.

## Usage

### Command line

```bash
python -m location_classifier <command> [options]
```

| Command | What it does |
| --- | --- |
| `make-data` | write a reproducible synthetic dataset |
| `info` | row counts, bounds and class balance for a dataset |
| `train` | fit, cross-validate, report and save a model |
| `evaluate` | score a saved model against a dataset (`--json` for machine output) |
| `predict` | classify a CSV, or a single `--lat/--lon` pair |
| `cluster` | k-means or DBSCAN grouping with a per-cluster summary |
| `compare` | benchmark every registered estimator on one split |

```bash
# train a specific estimator and write figures to outputs/
python -m location_classifier train --model gradient_boosting --n-anchors 10 --plots

# top-3 guesses for one point
python -m location_classifier predict --lat 19.076 --lon 72.877 --top-k 3

# DBSCAN with a 1.2 km neighbourhood radius
python -m location_classifier cluster --method dbscan --eps-km 1.2 --min-samples 8 --plot

# leaderboard across all six models
python -m location_classifier compare
```

Every command accepts `--seed` and `--output-dir`; paths default to
`data/locations.csv`, `models/location_classifier.joblib` and `outputs/`.

### Python

```python
from location_classifier import data, model
from location_classifier.config import Config

frame = data.make_synthetic_dataset(
    samples_per_class=150, spread_km=2.5, sites=data.MUMBAI_ZONES, random_seed=42
)

result = model.train_model(frame, config=Config(n_anchors=10, cv_folds=5))
print(result.accuracy, result.cv_mean)

points = model.points_frame(result.pipeline, [19.076], [72.877])
print(model.predict_locations(result.pipeline, points, top_k=3))
```

`points_frame` fills the auxiliary columns the model was trained on with their
training medians, so a bare coordinate pair stays in distribution.

### Notebook

`location_identifier.ipynb` walks through the whole pipeline — dataset, feature
matrix, training, error analysis, model comparison, clustering, persistence. It
imports the package rather than redefining logic, so nothing in it is stranded.

```bash
pip install -r requirements-dev.txt
jupyter notebook location_identifier.ipynb
```

## Configuration

Settings resolve as **CLI flag → environment variable → default**. Copy
`.env.example` and export what you need:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LOC_CLF_DATA_PATH` | `data/locations.csv` | dataset location |
| `LOC_CLF_MODEL_PATH` | `models/location_classifier.joblib` | saved model |
| `LOC_CLF_OUTPUT_DIR` | `outputs` | figures and reports |
| `LOC_CLF_MODEL_NAME` | `random_forest` | estimator |
| `LOC_CLF_RANDOM_SEED` | `42` | seed |
| `LOC_CLF_TEST_SIZE` | `0.2` | held-out fraction |
| `LOC_CLF_CV_FOLDS` | `5` | cross-validation folds |
| `LOC_CLF_N_ANCHORS` | `12` | learned distance anchors |
| `LOC_CLF_SAMPLES_PER_CLASS` | `120` | synthetic points per class |
| `LOC_CLF_SPREAD_KM` | `6.0` | synthetic scatter |

The CLI does not auto-load `.env`; export the variables yourself, or use
`python-dotenv` if you prefer.

## Results

On the `zones` collection (8 overlapping Mumbai neighbourhoods, 150 points each,
2.5 km spread, seed 42) — held-out accuracy by estimator:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| gradient_boosting | 0.950 | 0.950 |
| logistic_regression | 0.942 | 0.943 |
| random_forest | 0.933 | 0.934 |
| knn | 0.929 | 0.930 |
| svm | 0.929 | 0.929 |
| extra_trees | 0.921 | 0.922 |

Top-2 accuracy reaches 0.992, and the residual errors sit almost entirely
between geographically adjacent zones (Dadar↔Bandra, Andheri↔Bandra) — the
model is wrong exactly where the point clouds overlap. The `metros` collection
is separable and every model scores 1.000, which makes it a useful sanity check
rather than a benchmark.

Reproduce with:

```bash
python -m location_classifier make-data --sites zones --samples-per-class 150 --spread-km 2.5
python -m location_classifier compare
```

## Project layout

```
location_classifier/
  config.py      dataclass config with environment overrides
  data.py        synthetic generator, CSV loading, validation, splits
  geo.py         haversine, bearing, centroid, coordinate conversions
  features.py    GeoFeatureBuilder scikit-learn transformer
  model.py       pipeline registry, training, persistence, inference
  evaluate.py    metrics, per-class reports, confusion analysis
  cluster.py     k-means / DBSCAN in kilometre space
  visualize.py   matplotlib figures
  cli.py         argparse command line
tests/           174 tests, no dataset download required
location_identifier.ipynb
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is deterministic and offline: it covers the geodesy maths against
known distances, dataset validation, feature-matrix shape and invariants,
train/save/load round trips, clustering behaviour including DBSCAN noise
handling, every plot function, and every CLI command including its error paths.

## License

MIT — see [LICENSE](LICENSE).
