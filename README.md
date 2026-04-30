# NY ISO Electricity Price Spike Detection & Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Test Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen.svg)]()

## 1 — Purpose

This package builds an end-to-end pipeline for **detecting and forecasting
intraday electricity price spikes** in US wholesale electricity markets — using
NYISO (New York ISO) as the default freely accessible data source, with optional
support for PJM Interconnection, the largest wholesale electricity market in the
United States, serving 65 million customers across 13 states.

Electricity prices exhibit heavy-tailed distributions: most hours trade
between $20–40/MWh, but demand surges (heat waves, cold snaps, generator
outages) can push prices above $100/MWh within a single hour.  Accurately
predicting these spikes is critical for grid operators, power traders, and
industrial consumers managing procurement risk.

The pipeline:
1. **Downloads** real hourly LMP (Locational Marginal Price) data from NYISO
   (no API key required) or optionally from PJM, plus weather data from Open-Meteo.
2. **Engineers features** — lag values, rolling statistics, calendar/cyclical
   encodings, weather transformations, and a rolling spike label.
3. **Trains models** — an hourly-average baseline and LightGBM classifiers /
   regressors.
4. **Evaluates** via walk-forward cross-validation (the gold standard for
   time-series model assessment, preventing any look-ahead leakage).
5. **Produces results** — metrics, SHAP feature importances, and publication-
   quality plots.

## 2 — Dataset

| Source | Description |
|--------|-------------|
| **NYISO** (via public CSV archive, no API key) | Hourly real-time zonal LBMP ($/MWh) for N.Y.C. zone, 2022–2024 |
| **Open-Meteo Historical Archive** | Hourly temperature, humidity, wind speed, precipitation for Philadelphia, PA (central PJM footprint) | 
| **PJM Interconnection** (via `gridstatus`, optional) | Hourly real-time LMP for PJM RTO node — requires a free API key at apiportal.pjm.com |

The download step pulls **~26,000 hourly observations** per year of LMP data
and matching weather data.  Total dataset size for the default 3-year window
is approximately 78,000 rows × 9 columns.

A `--demo` mode is also available that uses a built-in dataset constructed
from publicly documented PJM price statistics (median ≈ $30/MWh, diurnal
cycle, seasonal pattern, heavy right tail).  This allows the full pipeline
to run without network access for immediate verification.

## 3 — Installation

```bash
# Clone the repository
git clone https://github.com/Alchemist2711/pjm-spike-forecast.git
cd pjm-spike-forecast

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

> **No API key required** for the default download path (NYISO public archive). PJM data is available as an optional source — see §2.

**Requirements:** Python ≥ 3.10. All dependencies are listed in `pyproject.toml` and will be installed automatically.

## 4 — Usage

### Quick demo (no download needed)

```bash
python3 -m pjm_spike_forecast.pipeline --demo --output-dir results
```

This runs the full pipeline on the built-in demo dataset and writes all
outputs to `results/`.

### Full pipeline with real PJM data

```bash
# Step 1: Download data (takes ~5 minutes depending on connection)
python3 -m pjm_spike_forecast.pipeline --download-only

# Step 2: Run the pipeline
python3 -m pjm_spike_forecast.pipeline --data-dir data/raw --output-dir results
```

### Makefile shortcuts

```bash
make install       # pip install -e ".[dev]"
make download      # download real PJM + weather data
make run           # run pipeline on real data
make demo          # run pipeline on demo data
make test          # run pytest with coverage
make clean         # remove build artifacts
```

### CLI options

```
python3 -m pjm_spike_forecast.pipeline --help

Options:
  --download-only       Download data and exit
  --demo                Use built-in demo dataset
  --data-dir PATH       Directory with raw CSVs (default: data/raw)
  --output-dir PATH     Results directory (default: results)
  --n-splits N          Walk-forward CV folds (default: 5)
  --train-months N      Training window per fold (default: 6)
  --test-months N       Test window per fold (default: 1)
```

## 5 — Importing and Running Useful Scripts

The package can also be used as a library in your own scripts or notebooks:

```python
from pjm_spike_forecast.data import download_pjm_lmp, download_weather, build_demo_dataset
from pjm_spike_forecast.features import build_feature_matrix, get_feature_columns
from pjm_spike_forecast.models import SpikeClassifier, PriceRegressor
from pjm_spike_forecast.evaluation import run_walk_forward_cv, compute_shap_importance
from pjm_spike_forecast.visualization import plot_lmp_timeseries, plot_feature_importance

# Load data
df = build_demo_dataset(n_days=365)

# Engineer features
feat_df = build_feature_matrix(df)
feature_cols = get_feature_columns(feat_df)

# Train a spike classifier
clf = SpikeClassifier()
clf.fit(feat_df[feature_cols], feat_df["spike"])

# Get SHAP importances
shap_df = compute_shap_importance(clf, feat_df[feature_cols])
```

Four Jupyter notebooks in `notebooks/` walk through each stage interactively:

| Notebook | Content |
|----------|---------|
| `01_data_exploration.ipynb` | EDA — distributions, diurnal/seasonal patterns, correlations |
| `02_feature_engineering.ipynb` | Lag, rolling, cyclical, and spike-label construction |
| `03_modeling.ipynb` | Train baseline + LightGBM, compare metrics, SHAP analysis |
| `04_evaluation_results.ipynb` | Walk-forward CV, final plots, full results |

## 6 — Project Structure

```
pjm-spike-forecast/
├── pyproject.toml                  # Package metadata & dependencies
├── README.md                       # This file
├── Makefile                        # Convenience commands
├── requirements.txt                # Flat dependency list
├── .gitignore
│
├── src/pjm_spike_forecast/         # Installable Python package
│   ├── __init__.py
│   ├── config.py                   # All constants & dataclass configs
│   ├── data.py                     # Download (gridstatus + Open-Meteo) & load
│   ├── features.py                 # Feature engineering pipeline
│   ├── models.py                   # Baseline, SpikeClassifier, PriceRegressor
│   ├── evaluation.py               # Metrics, walk-forward CV, SHAP
│   ├── visualization.py            # All plotting functions
│   └── pipeline.py                 # End-to-end orchestration + CLI
│
├── tests/                          # pytest suite (333 tests, >80% coverage)
│   ├── conftest.py                 # Shared fixtures
│   ├── test_config.py              # Config dataclass tests
│   ├── test_data.py                # Data download & loading tests
│   ├── test_features.py            # Feature engineering tests
│   ├── test_models.py              # Model fit/predict tests
│   ├── test_evaluation.py          # Metrics & walk-forward CV tests
│   ├── test_visualization.py       # Plotting smoke tests
│   └── test_pipeline.py            # End-to-end pipeline tests
│
├── notebooks/                      # Jupyter notebooks (interactive walkthrough)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation_results.ipynb
│
├── data/raw/                       # Downloaded CSVs (git-ignored)
└── results/                        # Pipeline outputs (git-ignored)
    ├── cv_baseline.csv
    ├── cv_lgbm.csv
    ├── shap_importance.csv
    ├── final_metrics.txt
    ├── spike_classifier.joblib
    ├── price_regressor.joblib
    └── plots/
        ├── lmp_timeseries.png
        ├── confusion_matrix.png
        ├── actual_vs_predicted.png
        ├── shap_importance.png
        ├── cv_summary_lgbm.png
        └── calibration_curve.png
```

## 7 — Running Tests

```bash
# Run all tests with coverage report
pytest tests/ -v --cov=pjm_spike_forecast --cov-report=term-missing

# Or via Makefile
make test
```

Current coverage: **>80%** (threshold: 80%).

## 8 — Methodology

### Spike Definition
A price spike is defined as any hour where:

> LMP > rolling_mean(168h) + 2 × rolling_std(168h)

Using a **rolling** threshold (rather than a static cutoff) adapts to
seasonal price-level shifts — a $50 price is a spike in spring but normal
in a summer heat wave.

### Walk-Forward Cross-Validation
The train/test splits are always temporal: each fold trains on *N* months
of history and tests on the next *M* months, with a configurable gap to
prevent lag-feature leakage.  This mirrors how the model would actually
be deployed in a real trading or risk management system.

### Models
- **HourlyBaseline**: average LMP per (day_of_week × hour) from training data.
- **SpikeClassifier**: LightGBM binary classifier with automatic
  `scale_pos_weight` to handle class imbalance.
- **PriceRegressor**: LightGBM regressor for continuous LMP prediction.

### Interpretability
SHAP (SHapley Additive exPlanations) values decompose each prediction into
per-feature contributions, revealing which drivers (e.g., recent price lags,
temperature, hour-of-day) matter most.

### Generated Results
The final dataset used and the results produced post running the pipeline are available in the PR for verification:
https://github.com/Alchemist2711/pjm-spike-forecast/pull/3


### Authors
ORIE-5270 Big Data Technology Project Team
Kanishk Agarwal
Pratvi Shah
