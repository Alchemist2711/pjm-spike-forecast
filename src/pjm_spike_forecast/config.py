"""
Configuration constants for the PJM spike forecasting pipeline.

All tunable parameters are centralised here so that experiments
can be reproduced by pointing at this single file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── PJM download defaults ───────────────────────────────────────
PJM_LOCATION = "PJM RTO"          # aggregate RTO-level price
PJM_MARKET = "REAL_TIME_HOURLY"   # real-time hourly LMPs
DEFAULT_START = "2022-01-01"
DEFAULT_END = "2024-12-31"

# ── Weather station (NYC – central to NYISO footprint) ──────────
WEATHER_STATION_LAT = 40.71       # New York City
WEATHER_STATION_LON = -74.01


@dataclass
class FeatureConfig:
    """Parameters controlling feature engineering."""

    # Lag features (hours)
    lmp_lags: List[int] = field(
        default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 168]
    )
    load_lags: List[int] = field(default_factory=lambda: [1, 24, 168])
    temp_lags: List[int] = field(default_factory=lambda: [1, 24])

    # Rolling window sizes (hours)
    rolling_windows: List[int] = field(
        default_factory=lambda: [6, 12, 24, 48, 168]
    )

    # Spike definition: LMP > rolling_mean + spike_std_multiplier * rolling_std
    spike_std_multiplier: float = 2.0
    spike_rolling_window: int = 168  # 1-week rolling window for threshold


@dataclass
class ModelConfig:
    """Hyper-parameters for the LightGBM models."""

    # ── Classifier (spike detection) ──
    clf_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": 42,
        "verbose": -1,
    })

    # ── Regressor (price level) ──
    reg_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "random_state": 42,
        "verbose": -1,
    })


@dataclass
class BacktestConfig:
    """Walk-forward cross-validation settings."""

    n_splits: int = 5
    train_months: int = 6   # months of training data per fold
    test_months: int = 1    # months of test data per fold
    gap_hours: int = 24     # gap between train and test to prevent leakage


@dataclass
class PipelineConfig:
    """Master configuration aggregating all sub-configs."""

    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
