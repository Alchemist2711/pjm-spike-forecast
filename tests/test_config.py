"""
Tests for pjm_spike_forecast.config

Verifies that all dataclass configs instantiate with correct defaults,
accept custom values, and enforce post-init constraints.
"""

from pathlib import Path

import pytest

from pjm_spike_forecast.config import (
    BacktestConfig,
    DATA_DIR,
    DEFAULT_END,
    DEFAULT_START,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    PROJECT_ROOT,
    RESULTS_DIR,
    WEATHER_STATION_LAT,
    WEATHER_STATION_LON,
)


# ─────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────

class TestModuleConstants:
    def test_project_root_is_path(self):
        assert isinstance(PROJECT_ROOT, Path)

    def test_data_dir_under_project_root(self):
        assert str(DATA_DIR).startswith(str(PROJECT_ROOT))

    def test_results_dir_under_project_root(self):
        assert str(RESULTS_DIR).startswith(str(PROJECT_ROOT))

    def test_default_start_format(self):
        # Must be a valid ISO date string
        from pandas import Timestamp
        ts = Timestamp(DEFAULT_START)
        assert ts.year == 2022

    def test_default_end_format(self):
        from pandas import Timestamp
        ts = Timestamp(DEFAULT_END)
        assert ts.year == 2024

    def test_weather_coords_are_nyc(self):
        # Latitude should be ~40°N, Longitude ~74°W
        assert 39.0 < WEATHER_STATION_LAT < 42.0
        assert -76.0 < WEATHER_STATION_LON < -72.0


# ─────────────────────────────────────────────────────────────────
# FeatureConfig
# ─────────────────────────────────────────────────────────────────

class TestFeatureConfig:
    def test_default_lmp_lags(self):
        cfg = FeatureConfig()
        assert 1 in cfg.lmp_lags
        assert 24 in cfg.lmp_lags
        assert 168 in cfg.lmp_lags

    def test_default_rolling_windows(self):
        cfg = FeatureConfig()
        assert len(cfg.rolling_windows) > 0
        assert all(w > 0 for w in cfg.rolling_windows)

    def test_default_spike_multiplier(self):
        cfg = FeatureConfig()
        assert cfg.spike_std_multiplier == 2.0

    def test_default_spike_rolling_window(self):
        cfg = FeatureConfig()
        assert cfg.spike_rolling_window == 168  # one week

    def test_custom_lags(self):
        cfg = FeatureConfig(lmp_lags=[1, 2, 3])
        assert cfg.lmp_lags == [1, 2, 3]

    def test_custom_spike_multiplier(self):
        cfg = FeatureConfig(spike_std_multiplier=3.0)
        assert cfg.spike_std_multiplier == 3.0

    def test_mutable_defaults_are_independent(self):
        # Each instance must get its own list, not a shared reference
        cfg1 = FeatureConfig()
        cfg2 = FeatureConfig()
        cfg1.lmp_lags.append(999)
        assert 999 not in cfg2.lmp_lags


# ─────────────────────────────────────────────────────────────────
# ModelConfig
# ─────────────────────────────────────────────────────────────────

class TestModelConfig:
    def test_clf_params_has_objective(self):
        cfg = ModelConfig()
        assert cfg.clf_params["objective"] == "binary"

    def test_reg_params_has_objective(self):
        cfg = ModelConfig()
        assert cfg.reg_params["objective"] == "regression"

    def test_clf_params_has_n_estimators(self):
        cfg = ModelConfig()
        assert cfg.clf_params["n_estimators"] > 0

    def test_reg_params_has_n_estimators(self):
        cfg = ModelConfig()
        assert cfg.reg_params["n_estimators"] > 0

    def test_clf_and_reg_params_are_independent(self):
        cfg = ModelConfig()
        cfg.clf_params["n_estimators"] = 1
        assert cfg.reg_params["n_estimators"] != 1

    def test_mutable_defaults_are_independent(self):
        cfg1 = ModelConfig()
        cfg2 = ModelConfig()
        cfg1.clf_params["foo"] = "bar"
        assert "foo" not in cfg2.clf_params

    def test_custom_clf_params(self):
        custom = {"objective": "binary", "n_estimators": 10, "verbose": -1}
        cfg = ModelConfig(clf_params=custom)
        assert cfg.clf_params["n_estimators"] == 10


# ─────────────────────────────────────────────────────────────────
# BacktestConfig
# ─────────────────────────────────────────────────────────────────

class TestBacktestConfig:
    def test_default_n_splits(self):
        cfg = BacktestConfig()
        assert cfg.n_splits == 5

    def test_default_train_months(self):
        cfg = BacktestConfig()
        assert cfg.train_months == 6

    def test_default_test_months(self):
        cfg = BacktestConfig()
        assert cfg.test_months == 1

    def test_default_gap_hours(self):
        cfg = BacktestConfig()
        assert cfg.gap_hours == 24

    def test_custom_n_splits(self):
        cfg = BacktestConfig(n_splits=3)
        assert cfg.n_splits == 3

    def test_custom_gap(self):
        cfg = BacktestConfig(gap_hours=0)
        assert cfg.gap_hours == 0


# ─────────────────────────────────────────────────────────────────
# PipelineConfig
# ─────────────────────────────────────────────────────────────────

class TestPipelineConfig:
    def test_default_sub_configs_instantiated(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.features, FeatureConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.backtest, BacktestConfig)

    def test_data_dir_coerced_to_path(self, tmp_path):
        cfg = PipelineConfig(data_dir=str(tmp_path / "data"))
        assert isinstance(cfg.data_dir, Path)

    def test_results_dir_coerced_to_path(self, tmp_path):
        cfg = PipelineConfig(results_dir=str(tmp_path / "results"))
        assert isinstance(cfg.results_dir, Path)

    def test_results_dir_created_on_init(self, tmp_path):
        results = tmp_path / "new_results_dir"
        assert not results.exists()
        cfg = PipelineConfig(results_dir=results)
        assert results.exists()

    def test_custom_feature_config(self):
        feat = FeatureConfig(lmp_lags=[1, 2])
        cfg = PipelineConfig(features=feat)
        assert cfg.features.lmp_lags == [1, 2]

    def test_custom_backtest_config(self):
        bt = BacktestConfig(n_splits=2)
        cfg = PipelineConfig(backtest=bt)
        assert cfg.backtest.n_splits == 2
