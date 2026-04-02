"""
Shared pytest fixtures for the PJM Spike Forecast test suite.

Every fixture produces data with the same schema as the real pipeline,
so tests exercise the exact same code paths that run in production.
"""

import numpy as np
import pandas as pd
import pytest

from pjm_spike_forecast.config import (
    BacktestConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
)
from pjm_spike_forecast.data import build_demo_dataset


@pytest.fixture
def demo_df() -> pd.DataFrame:
    """One year of realistic demo data (same as --demo flag)."""
    return build_demo_dataset(n_days=120, seed=42)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Tiny dataset (30 days) for fast unit tests."""
    return build_demo_dataset(n_days=30, seed=99)


@pytest.fixture
def featured_df(demo_df):
    """Demo data with all features already engineered."""
    from pjm_spike_forecast.features import build_feature_matrix
    return build_feature_matrix(demo_df)


@pytest.fixture
def feature_cols(featured_df):
    """List of model-input feature column names."""
    from pjm_spike_forecast.features import get_feature_columns
    return get_feature_columns(featured_df)


@pytest.fixture
def fast_pipeline_cfg(tmp_path) -> PipelineConfig:
    """Pipeline config with small windows for fast tests."""
    return PipelineConfig(
        features=FeatureConfig(
            lmp_lags=[1, 24],
            load_lags=[1],
            temp_lags=[1],
            rolling_windows=[6, 24],
            spike_rolling_window=48,
        ),
        model=ModelConfig(
            clf_params={
                "objective": "binary",
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "verbose": -1,
                "random_state": 42,
            },
            reg_params={
                "objective": "regression",
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "verbose": -1,
                "random_state": 42,
            },
        ),
        backtest=BacktestConfig(
            n_splits=2,
            train_months=1,
            test_months=1,
            gap_hours=6,
        ),
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
    )
