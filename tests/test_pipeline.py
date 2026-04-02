"""Tests for pjm_spike_forecast.pipeline and pjm_spike_forecast.config."""

import pytest
from pathlib import Path

from pjm_spike_forecast.config import (
    BacktestConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
)


class TestFeatureConfig:

    def test_default_lmp_lags(self):
        cfg = FeatureConfig()
        assert 1 in cfg.lmp_lags
        assert 24 in cfg.lmp_lags
        assert 168 in cfg.lmp_lags

    def test_default_rolling_windows(self):
        cfg = FeatureConfig()
        assert 24 in cfg.rolling_windows

    def test_custom_values(self):
        cfg = FeatureConfig(lmp_lags=[1, 2], spike_std_multiplier=3.0)
        assert cfg.lmp_lags == [1, 2]
        assert cfg.spike_std_multiplier == 3.0


class TestModelConfig:

    def test_default_clf_params(self):
        cfg = ModelConfig()
        assert cfg.clf_params["objective"] == "binary"
        assert cfg.clf_params["n_estimators"] > 0

    def test_default_reg_params(self):
        cfg = ModelConfig()
        assert cfg.reg_params["objective"] == "regression"


class TestBacktestConfig:

    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.n_splits == 5
        assert cfg.train_months == 6
        assert cfg.gap_hours == 24

    def test_custom(self):
        cfg = BacktestConfig(n_splits=3, train_months=3, test_months=2)
        assert cfg.n_splits == 3
        assert cfg.train_months == 3


class TestPipelineConfig:

    def test_creates_results_dir(self, tmp_path):
        out = tmp_path / "my_results"
        cfg = PipelineConfig(results_dir=out)
        assert out.exists()

    def test_nested_configs(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.features, FeatureConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.backtest, BacktestConfig)

    def test_path_conversion(self, tmp_path):
        cfg = PipelineConfig(data_dir=str(tmp_path), results_dir=str(tmp_path / "res"))
        assert isinstance(cfg.data_dir, Path)
        assert isinstance(cfg.results_dir, Path)


class TestDemoPipeline:
    """Integration test: run the full pipeline in demo mode."""

    def test_demo_pipeline_runs(self, fast_pipeline_cfg):
        from pjm_spike_forecast.pipeline import run_pipeline
        run_pipeline(fast_pipeline_cfg, demo=True)

        # Check that results were produced
        results_dir = fast_pipeline_cfg.results_dir
        assert (results_dir / "final_metrics.txt").exists()
        assert (results_dir / "shap_importance.csv").exists()
        assert (results_dir / "spike_classifier.joblib").exists()
        assert (results_dir / "price_regressor.joblib").exists()

    def test_demo_pipeline_produces_plots(self, fast_pipeline_cfg):
        from pjm_spike_forecast.pipeline import run_pipeline
        run_pipeline(fast_pipeline_cfg, demo=True)

        plots_dir = fast_pipeline_cfg.results_dir / "plots"
        assert plots_dir.exists()
        assert (plots_dir / "lmp_timeseries.png").exists()
        assert (plots_dir / "confusion_matrix.png").exists()
        assert (plots_dir / "actual_vs_predicted.png").exists()
