"""Tests for pjm_spike_forecast.pipeline"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pjm_spike_forecast.config import (
    BacktestConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
)
from pjm_spike_forecast.pipeline import (
    main,
    run_pipeline,
    step_download,
    step_evaluate,
    step_features,
    step_final_model,
    step_load,
    step_save_results,
)


# ─────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def demo_raw_df():
    from pjm_spike_forecast.data import build_demo_dataset
    return build_demo_dataset(n_days=180, seed=42)


@pytest.fixture(scope="module")
def demo_feat_df(demo_raw_df):
    from pjm_spike_forecast.features import build_feature_matrix
    from pjm_spike_forecast.config import FeatureConfig
    cfg = FeatureConfig(lmp_lags=[1, 24], rolling_windows=[24],
                        spike_rolling_window=48)
    return build_feature_matrix(demo_raw_df, cfg)


@pytest.fixture(scope="module")
def demo_feature_cols(demo_feat_df):
    return [c for c in demo_feat_df.columns
            if c not in {"spike", "lmp", "spike_threshold"}]


@pytest.fixture
def fast_pipeline_cfg(tmp_path):
    """Minimal PipelineConfig using a very fast feature + CV config."""
    return PipelineConfig(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        backtest=BacktestConfig(
            n_splits=2,
            train_months=1,
            test_months=1,
            gap_hours=6,
        ),
        features=FeatureConfig(
            lmp_lags=[1, 24],
            rolling_windows=[24],
            spike_rolling_window=48,
        ),
    )


# ─────────────────────────────────────────────────────────────────
# 1.  Config classes
# ─────────────────────────────────────────────────────────────────

class TestFeatureConfig:

    def test_default_lmp_lags(self):
        cfg = FeatureConfig()
        assert 1 in cfg.lmp_lags
        assert 24 in cfg.lmp_lags
        assert 168 in cfg.lmp_lags

    def test_default_rolling_windows(self):
        cfg = FeatureConfig()
        assert 24 in cfg.rolling_windows

    def test_custom_lmp_lags(self):
        cfg = FeatureConfig(lmp_lags=[1, 2])
        assert cfg.lmp_lags == [1, 2]

    def test_custom_spike_std_multiplier(self):
        cfg = FeatureConfig(spike_std_multiplier=3.0)
        assert cfg.spike_std_multiplier == 3.0

    def test_custom_spike_rolling_window(self):
        cfg = FeatureConfig(spike_rolling_window=48)
        assert cfg.spike_rolling_window == 48

    def test_custom_rolling_windows(self):
        cfg = FeatureConfig(rolling_windows=[6, 12])
        assert 6 in cfg.rolling_windows


class TestModelConfig:

    def test_default_clf_params_objective(self):
        cfg = ModelConfig()
        assert cfg.clf_params["objective"] == "binary"

    def test_default_clf_params_n_estimators(self):
        cfg = ModelConfig()
        assert cfg.clf_params["n_estimators"] > 0

    def test_default_reg_params_objective(self):
        cfg = ModelConfig()
        assert cfg.reg_params["objective"] == "regression"

    def test_default_reg_params_n_estimators(self):
        cfg = ModelConfig()
        assert cfg.reg_params["n_estimators"] > 0

    def test_clf_and_reg_are_dicts(self):
        cfg = ModelConfig()
        assert isinstance(cfg.clf_params, dict)
        assert isinstance(cfg.reg_params, dict)


class TestBacktestConfig:

    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.n_splits == 5
        assert cfg.train_months == 6
        assert cfg.gap_hours == 24

    def test_custom_n_splits(self):
        cfg = BacktestConfig(n_splits=3)
        assert cfg.n_splits == 3

    def test_custom_train_months(self):
        cfg = BacktestConfig(train_months=3)
        assert cfg.train_months == 3

    def test_custom_test_months(self):
        cfg = BacktestConfig(test_months=2)
        assert cfg.test_months == 2

    def test_custom_gap_hours(self):
        cfg = BacktestConfig(gap_hours=48)
        assert cfg.gap_hours == 48


class TestPipelineConfig:

    def test_creates_results_dir(self, tmp_path):
        out = tmp_path / "my_results"
        PipelineConfig(results_dir=out)
        assert out.exists()

    def test_nested_configs(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.features, FeatureConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.backtest, BacktestConfig)

    # Branch: path provided as str → converted to Path
    def test_path_conversion_data_dir(self, tmp_path):
        cfg = PipelineConfig(data_dir=str(tmp_path),
                             results_dir=str(tmp_path / "res"))
        assert isinstance(cfg.data_dir, Path)

    def test_path_conversion_results_dir(self, tmp_path):
        cfg = PipelineConfig(results_dir=str(tmp_path / "res2"))
        assert isinstance(cfg.results_dir, Path)

    def test_custom_backtest_embedded(self, tmp_path):
        bt = BacktestConfig(n_splits=3)
        cfg = PipelineConfig(results_dir=tmp_path / "r", backtest=bt)
        assert cfg.backtest.n_splits == 3


# ─────────────────────────────────────────────────────────────────
# 2.  step_download
# ─────────────────────────────────────────────────────────────────

class TestStepDownload:

    @patch("pjm_spike_forecast.pipeline.download_nyiso_lmp")
    @patch("pjm_spike_forecast.pipeline.download_weather")
    def test_calls_both_downloaders(self, mock_weather, mock_lmp, tmp_path):
        cfg = PipelineConfig(data_dir=tmp_path / "data",
                             results_dir=tmp_path / "res")
        step_download(cfg)
        mock_lmp.assert_called_once()
        mock_weather.assert_called_once()

    @patch("pjm_spike_forecast.pipeline.download_nyiso_lmp")
    @patch("pjm_spike_forecast.pipeline.download_weather")
    def test_creates_data_dir(self, mock_weather, mock_lmp, tmp_path):
        data_dir = tmp_path / "nested" / "data"
        cfg = PipelineConfig(data_dir=data_dir, results_dir=tmp_path / "res")
        step_download(cfg)
        assert data_dir.exists()

    @patch("pjm_spike_forecast.pipeline.download_nyiso_lmp")
    @patch("pjm_spike_forecast.pipeline.download_weather")
    def test_passes_correct_paths(self, mock_weather, mock_lmp, tmp_path):
        cfg = PipelineConfig(data_dir=tmp_path / "data",
                             results_dir=tmp_path / "res")
        step_download(cfg)
        lmp_call_kwargs = mock_lmp.call_args[1]
        weather_call_kwargs = mock_weather.call_args[1]
        assert "save_path" in lmp_call_kwargs
        assert "save_path" in weather_call_kwargs


# ─────────────────────────────────────────────────────────────────
# 3.  step_load
# ─────────────────────────────────────────────────────────────────

class TestStepLoad:

    # Branch: demo=True → returns demo dataset
    def test_demo_returns_dataframe(self, fast_pipeline_cfg):
        df = step_load(fast_pipeline_cfg, demo=True)
        assert isinstance(df, pd.DataFrame)
        assert "lmp" in df.columns

    def test_demo_correct_columns(self, fast_pipeline_cfg):
        df = step_load(fast_pipeline_cfg, demo=True)
        assert "temperature_c" in df.columns

    # Branch: nyiso_lmp.csv exists
    def test_loads_nyiso_csv_when_exists(self, tmp_path, demo_raw_df):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        lmp_path = data_dir / "nyiso_lmp.csv"
        demo_raw_df[["lmp"]].reset_index().to_csv(lmp_path, index=False)
        cfg = PipelineConfig(data_dir=data_dir, results_dir=tmp_path / "res")
        df = step_load(cfg, demo=False)
        assert isinstance(df, pd.DataFrame)
        assert "lmp" in df.columns

    # Branch: nyiso_lmp.csv absent, pjm_lmp.csv present
    def test_falls_back_to_pjm_csv(self, tmp_path, demo_raw_df):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pjm_path = data_dir / "pjm_lmp.csv"
        demo_raw_df[["lmp"]].reset_index().to_csv(pjm_path, index=False)
        cfg = PipelineConfig(data_dir=data_dir, results_dir=tmp_path / "res")
        df = step_load(cfg, demo=False)
        assert "lmp" in df.columns

    # Branch: neither csv exists → FileNotFoundError
    def test_raises_when_no_lmp_file(self, tmp_path):
        data_dir = tmp_path / "empty_data"
        data_dir.mkdir()
        cfg = PipelineConfig(data_dir=data_dir, results_dir=tmp_path / "res")
        with pytest.raises(FileNotFoundError, match="LMP data not found"):
            step_load(cfg, demo=False)

    # Branch: weather.csv exists → merged
    def test_merges_weather_when_present(self, tmp_path, demo_raw_df):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        lmp_path = data_dir / "nyiso_lmp.csv"
        weather_path = data_dir / "weather.csv"
        demo_raw_df[["lmp"]].reset_index().to_csv(lmp_path, index=False)
        demo_raw_df[["temperature_c", "relative_humidity",
                     "wind_speed_kmh", "precipitation_mm"]].reset_index(
        ).to_csv(weather_path, index=False)
        cfg = PipelineConfig(data_dir=data_dir, results_dir=tmp_path / "res")
        df = step_load(cfg, demo=False)
        assert "temperature_c" in df.columns

    # Branch: weather.csv absent → warning logged, LMP-only returned
    def test_no_weather_returns_lmp_only(self, tmp_path, demo_raw_df):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        lmp_path = data_dir / "nyiso_lmp.csv"
        demo_raw_df[["lmp"]].reset_index().to_csv(lmp_path, index=False)
        cfg = PipelineConfig(data_dir=data_dir, results_dir=tmp_path / "res")
        df = step_load(cfg, demo=False)
        assert "lmp" in df.columns
        assert "temperature_c" not in df.columns


# ─────────────────────────────────────────────────────────────────
# 4.  step_features
# ─────────────────────────────────────────────────────────────────

class TestStepFeatures:

    def test_returns_dataframe(self, demo_raw_df, fast_pipeline_cfg):
        result = step_features(demo_raw_df, fast_pipeline_cfg)
        assert isinstance(result, pd.DataFrame)

    def test_spike_column_present(self, demo_raw_df, fast_pipeline_cfg):
        result = step_features(demo_raw_df, fast_pipeline_cfg)
        assert "spike" in result.columns

    def test_no_nans(self, demo_raw_df, fast_pipeline_cfg):
        result = step_features(demo_raw_df, fast_pipeline_cfg)
        assert result.isna().sum().sum() == 0

    def test_fewer_rows_than_input(self, demo_raw_df, fast_pipeline_cfg):
        result = step_features(demo_raw_df, fast_pipeline_cfg)
        assert len(result) < len(demo_raw_df)


# ─────────────────────────────────────────────────────────────────
# 5.  step_evaluate
# ─────────────────────────────────────────────────────────────────

class TestStepEvaluate:

    def test_returns_dict_with_both_keys(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        results = step_evaluate(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        assert "baseline" in results
        assert "lgbm" in results

    def test_baseline_results_are_fold_results(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        from pjm_spike_forecast.evaluation import FoldResult
        results = step_evaluate(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        for r in results["baseline"]:
            assert isinstance(r, FoldResult)

    def test_lgbm_results_are_fold_results(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        from pjm_spike_forecast.evaluation import FoldResult
        results = step_evaluate(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        for r in results["lgbm"]:
            assert isinstance(r, FoldResult)


# ─────────────────────────────────────────────────────────────────
# 6.  step_final_model
# ─────────────────────────────────────────────────────────────────

class TestStepFinalModel:

    def test_returns_all_expected_keys(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        result = step_final_model(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        expected = {
            "clf", "reg", "baseline", "test_df", "X_test",
            "spike_pred", "spike_proba", "price_pred", "baseline_price",
            "clf_metrics", "reg_metrics", "bl_reg_metrics", "shap_df",
        }
        assert expected.issubset(result.keys())

    def test_spike_pred_binary(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        result = step_final_model(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        assert set(np.unique(result["spike_pred"])).issubset({0, 1})

    def test_price_pred_finite(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        result = step_final_model(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        assert np.all(np.isfinite(result["price_pred"]))

    def test_shap_df_has_expected_columns(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        result = step_final_model(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        assert "feature" in result["shap_df"].columns
        assert "mean_abs_shap" in result["shap_df"].columns

    def test_clf_metrics_has_f1(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        result = step_final_model(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        assert "f1" in result["clf_metrics"]

    def test_reg_metrics_has_rmse(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        result = step_final_model(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        assert "rmse" in result["reg_metrics"]

    # Branch: SHAP computation fails → fallback zero-importance df
    def test_shap_failure_falls_back_gracefully(
        self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg
    ):
        with patch(
            "pjm_spike_forecast.pipeline.compute_shap_importance",
            side_effect=Exception("shap error")
        ):
            result = step_final_model(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)
        shap_df = result["shap_df"]
        assert isinstance(shap_df, pd.DataFrame)
        assert (shap_df["mean_abs_shap"] == 0).all()


# ─────────────────────────────────────────────────────────────────
# 7.  step_save_results
# ─────────────────────────────────────────────────────────────────

class TestStepSaveResults:

    @pytest.fixture
    def cv_results(self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg):
        return step_evaluate(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)

    @pytest.fixture
    def final(self, demo_feat_df, demo_feature_cols, fast_pipeline_cfg):
        return step_final_model(demo_feat_df, demo_feature_cols, fast_pipeline_cfg)

    def test_creates_output_dir(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert fast_pipeline_cfg.results_dir.exists()

    def test_creates_plots_dir(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert (fast_pipeline_cfg.results_dir / "plots").exists()

    def test_saves_cv_csv(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert (fast_pipeline_cfg.results_dir / "cv_baseline.csv").exists()
        assert (fast_pipeline_cfg.results_dir / "cv_lgbm.csv").exists()

    def test_saves_shap_csv(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert (fast_pipeline_cfg.results_dir / "shap_importance.csv").exists()

    def test_saves_metrics_txt(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert (fast_pipeline_cfg.results_dir / "final_metrics.txt").exists()

    def test_saves_models(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert (fast_pipeline_cfg.results_dir / "spike_classifier.joblib").exists()
        assert (fast_pipeline_cfg.results_dir / "price_regressor.joblib").exists()

    def test_saves_lmp_timeseries_plot(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert (fast_pipeline_cfg.results_dir / "plots" / "lmp_timeseries.png").exists()

    def test_saves_confusion_matrix_plot(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert (fast_pipeline_cfg.results_dir / "plots" / "confusion_matrix.png").exists()

    def test_saves_actual_vs_predicted_plot(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        assert (fast_pipeline_cfg.results_dir / "plots" / "actual_vs_predicted.png").exists()

    # Branch: shap_df sum > 0 → feature importance plot saved
    def test_saves_shap_plot_when_nonzero(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        plots_dir = fast_pipeline_cfg.results_dir / "plots"
        # file is created if shap values are nonzero
        shap_plot = plots_dir / "shap_importance.png"
        if final["shap_df"]["mean_abs_shap"].sum() > 0:
            assert shap_plot.exists()

    # Branch: shap_df sum == 0 → feature importance plot skipped
    def test_skips_shap_plot_when_all_zero(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg, tmp_path
    ):
        cfg2 = PipelineConfig(
            data_dir=tmp_path / "d2",
            results_dir=tmp_path / "r2",
            backtest=fast_pipeline_cfg.backtest,
            features=fast_pipeline_cfg.features,
        )
        zero_final = dict(final)
        zero_final["shap_df"] = pd.DataFrame({
            "feature": ["a"], "mean_abs_shap": [0.0]
        })
        step_save_results(cv_results, zero_final, demo_feat_df, cfg2)
        assert not (cfg2.results_dir / "plots" / "shap_importance.png").exists()

    # Branch: lgbm cv_results non-empty → cv_summary plot saved
    def test_saves_cv_summary_when_lgbm_results_present(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg
    ):
        step_save_results(cv_results, final, demo_feat_df, fast_pipeline_cfg)
        if cv_results.get("lgbm"):
            assert (fast_pipeline_cfg.results_dir / "plots" / "cv_summary_lgbm.png").exists()

    # Branch: lgbm cv_results empty → cv_summary plot skipped
    def test_skips_cv_summary_when_lgbm_empty(
        self, final, demo_feat_df, fast_pipeline_cfg, tmp_path
    ):
        cfg3 = PipelineConfig(
            data_dir=tmp_path / "d3",
            results_dir=tmp_path / "r3",
            backtest=fast_pipeline_cfg.backtest,
            features=fast_pipeline_cfg.features,
        )
        empty_cv = {"baseline": [], "lgbm": []}
        step_save_results(empty_cv, final, demo_feat_df, cfg3)
        assert not (cfg3.results_dir / "plots" / "cv_summary_lgbm.png").exists()

    # Branch: plot_calibration_curve raises → exception silently swallowed
    def test_calibration_curve_exception_suppressed(
        self, cv_results, final, demo_feat_df, fast_pipeline_cfg, tmp_path
    ):
        cfg4 = PipelineConfig(
            data_dir=tmp_path / "d4",
            results_dir=tmp_path / "r4",
            backtest=fast_pipeline_cfg.backtest,
            features=fast_pipeline_cfg.features,
        )
        with patch(
            "pjm_spike_forecast.visualization.plot_calibration_curve",
            side_effect=Exception("calibration error")
        ):
            # Should not raise
            step_save_results(cv_results, final, demo_feat_df, cfg4)


# ─────────────────────────────────────────────────────────────────
# 8.  run_pipeline
# ─────────────────────────────────────────────────────────────────

class TestRunPipeline:

    def test_demo_pipeline_runs_end_to_end(self, fast_pipeline_cfg):
        run_pipeline(fast_pipeline_cfg, demo=True)
        assert (fast_pipeline_cfg.results_dir / "final_metrics.txt").exists()
        assert (fast_pipeline_cfg.results_dir / "spike_classifier.joblib").exists()
        assert (fast_pipeline_cfg.results_dir / "price_regressor.joblib").exists()

    def test_demo_pipeline_produces_plots(self, fast_pipeline_cfg):
        run_pipeline(fast_pipeline_cfg, demo=True)
        plots_dir = fast_pipeline_cfg.results_dir / "plots"
        assert plots_dir.exists()
        assert (plots_dir / "lmp_timeseries.png").exists()
        assert (plots_dir / "confusion_matrix.png").exists()
        assert (plots_dir / "actual_vs_predicted.png").exists()

    def test_demo_pipeline_produces_cv_csvs(self, fast_pipeline_cfg):
        run_pipeline(fast_pipeline_cfg, demo=True)
        assert (fast_pipeline_cfg.results_dir / "cv_baseline.csv").exists()
        assert (fast_pipeline_cfg.results_dir / "cv_lgbm.csv").exists()

    def test_demo_pipeline_produces_shap_csv(self, fast_pipeline_cfg):
        run_pipeline(fast_pipeline_cfg, demo=True)
        assert (fast_pipeline_cfg.results_dir / "shap_importance.csv").exists()

    def test_pipeline_with_real_data(self, tmp_path, demo_raw_df):
        """step_load non-demo path with existing files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        lmp_path = data_dir / "nyiso_lmp.csv"
        demo_raw_df[["lmp"]].reset_index().to_csv(lmp_path, index=False)
        cfg = PipelineConfig(
            data_dir=data_dir,
            results_dir=tmp_path / "results",
            backtest=BacktestConfig(n_splits=1, train_months=1,
                                    test_months=1, gap_hours=6),
            features=FeatureConfig(lmp_lags=[1, 24], rolling_windows=[24],
                                   spike_rolling_window=48),
        )
        run_pipeline(cfg, demo=False)
        assert (cfg.results_dir / "final_metrics.txt").exists()


# ─────────────────────────────────────────────────────────────────
# 9.  main() / CLI
# ─────────────────────────────────────────────────────────────────

class TestMain:

    # Branch: --download-only → step_download called, sys.exit(0)
    @patch("pjm_spike_forecast.pipeline.step_download")
    def test_download_only_flag(self, mock_download, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            with patch(
                "sys.argv",
                ["pipeline",
                 "--download-only",
                 "--data-dir", str(tmp_path),
                 "--output-dir", str(tmp_path / "out")]
            ):
                main()
        assert exc_info.value.code == 0
        mock_download.assert_called_once()

    # Branch: --demo → run_pipeline called with demo=True
    @patch("pjm_spike_forecast.pipeline.run_pipeline")
    def test_demo_flag_calls_run_pipeline(self, mock_run, tmp_path):
        with patch(
            "sys.argv",
            ["pipeline",
             "--demo",
             "--output-dir", str(tmp_path / "out")]
        ):
            main()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("demo") is True or mock_run.call_args[0][1] is True

    # Branch: default (no flags) → run_pipeline called with demo=False
    @patch("pjm_spike_forecast.pipeline.run_pipeline")
    def test_no_flags_calls_run_pipeline_no_demo(self, mock_run, tmp_path):
        with patch(
            "sys.argv",
            ["pipeline",
             "--output-dir", str(tmp_path / "out"),
             "--data-dir", str(tmp_path)]
        ):
            main()
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        demo_arg = (call_args[0][1] if len(call_args[0]) > 1
                    else call_args[1].get("demo", False))
        assert demo_arg is False

    # Branch: --n-splits, --train-months, --test-months forwarded to config
    @patch("pjm_spike_forecast.pipeline.run_pipeline")
    def test_cli_args_forwarded_to_config(self, mock_run, tmp_path):
        with patch(
            "sys.argv",
            ["pipeline",
             "--demo",
             "--output-dir", str(tmp_path / "out"),
             "--n-splits", "3",
             "--train-months", "2",
             "--test-months", "1"]
        ):
            main()
        cfg_arg = mock_run.call_args[0][0]
        assert cfg_arg.backtest.n_splits == 3
        assert cfg_arg.backtest.train_months == 2
        assert cfg_arg.backtest.test_months == 1

    @patch("pjm_spike_forecast.pipeline.run_pipeline")
    def test_custom_data_dir_forwarded(self, mock_run, tmp_path):
        custom_data = tmp_path / "custom_data"
        with patch(
            "sys.argv",
            ["pipeline",
             "--demo",
             "--data-dir", str(custom_data),
             "--output-dir", str(tmp_path / "out")]
        ):
            main()
        cfg_arg = mock_run.call_args[0][0]
        assert cfg_arg.data_dir == custom_data

    @patch("pjm_spike_forecast.pipeline.run_pipeline")
    def test_custom_output_dir_forwarded(self, mock_run, tmp_path):
        custom_out = tmp_path / "custom_out"
        with patch(
            "sys.argv",
            ["pipeline",
             "--demo",
             "--output-dir", str(custom_out)]
        ):
            main()
        cfg_arg = mock_run.call_args[0][0]
        assert cfg_arg.results_dir == custom_out