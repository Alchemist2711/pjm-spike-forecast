"""
Tests for pjm_spike_forecast.visualization

Smoke tests that verify every plotting function:
  - returns a matplotlib Figure without raising
  - closes figures afterwards to avoid memory leaks during the test run

No pixel-level assertions are made — we test the contract (a Figure is
returned) not the aesthetic output.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pjm_spike_forecast.data import build_demo_dataset
from pjm_spike_forecast.evaluation import FoldResult
from pjm_spike_forecast.features import build_feature_matrix
from pjm_spike_forecast.visualization import (
    plot_actual_vs_predicted,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_cv_summary,
    plot_feature_importance,
    plot_lmp_timeseries,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test to prevent leaks."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def featured_df():
    raw = build_demo_dataset(n_days=60, seed=0)
    from pjm_spike_forecast.config import FeatureConfig
    cfg = FeatureConfig(lmp_lags=[1, 24], rolling_windows=[24], spike_rolling_window=48)
    return build_feature_matrix(raw, cfg)


@pytest.fixture(scope="module")
def importance_df():
    return pd.DataFrame({
        "feature": [f"feat_{i}" for i in range(10)],
        "mean_abs_shap": np.linspace(1.0, 0.1, 10),
    })


@pytest.fixture(scope="module")
def binary_arrays(featured_df):
    y_true = featured_df["spike"].values
    # Dummy proba: spike rows get 0.8, others get 0.2
    y_proba = np.where(y_true == 1, 0.8, 0.2).astype(float)
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba


@pytest.fixture(scope="module")
def price_arrays(featured_df):
    y_true = featured_df["lmp"].values
    y_pred = y_true * np.random.default_rng(7).uniform(0.9, 1.1, len(y_true))
    return y_true, y_pred


@pytest.fixture(scope="module")
def fold_results():
    """Minimal FoldResult list for cv summary plot."""
    return [
        FoldResult(
            fold=i,
            train_start="2023-01-01",
            train_end="2023-06-30",
            test_start="2023-07-01",
            test_end="2023-07-31",
            clf_metrics={"f1": 0.5 + i * 0.05, "auc_roc": 0.7 + i * 0.02},
            reg_metrics={"rmse": 10.0 - i, "mae": 7.0, "mape": 15.0, "r2": 0.6},
        )
        for i in range(3)
    ]


# ─────────────────────────────────────────────────────────────────
# plot_lmp_timeseries
# ─────────────────────────────────────────────────────────────────

class TestPlotLmpTimeseries:
    def test_returns_figure(self, featured_df):
        fig = plot_lmp_timeseries(featured_df)
        assert isinstance(fig, plt.Figure)

    def test_works_without_spike_column(self, featured_df):
        df_no_spike = featured_df.drop(columns=["spike", "spike_threshold"], errors="ignore")
        fig = plot_lmp_timeseries(df_no_spike)
        assert isinstance(fig, plt.Figure)

    def test_works_without_threshold_column(self, featured_df):
        df = featured_df.drop(columns=["spike_threshold"], errors="ignore")
        fig = plot_lmp_timeseries(df)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self, featured_df):
        fig = plot_lmp_timeseries(featured_df, title="My Custom Title")
        assert isinstance(fig, plt.Figure)

    def test_figure_has_axes(self, featured_df):
        fig = plot_lmp_timeseries(featured_df)
        assert len(fig.axes) >= 1


# ─────────────────────────────────────────────────────────────────
# plot_feature_importance
# ─────────────────────────────────────────────────────────────────

class TestPlotFeatureImportance:
    def test_returns_figure(self, importance_df):
        fig = plot_feature_importance(importance_df)
        assert isinstance(fig, plt.Figure)

    def test_top_n_respected(self, importance_df):
        # Only 5 of 10 features should appear
        fig = plot_feature_importance(importance_df, top_n=5)
        ax = fig.axes[0]
        assert len(ax.patches) == 5

    def test_custom_title(self, importance_df):
        fig = plot_feature_importance(importance_df, title="My Importance Chart")
        assert isinstance(fig, plt.Figure)

    def test_single_feature(self):
        df = pd.DataFrame({"feature": ["only_one"], "mean_abs_shap": [1.0]})
        fig = plot_feature_importance(df, top_n=1)
        assert isinstance(fig, plt.Figure)


# ─────────────────────────────────────────────────────────────────
# plot_confusion_matrix
# ─────────────────────────────────────────────────────────────────

class TestPlotConfusionMatrix:
    def test_returns_figure(self, binary_arrays):
        y_true, y_pred, _ = binary_arrays
        fig = plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, plt.Figure)

    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        fig = plot_confusion_matrix(y, y)
        assert isinstance(fig, plt.Figure)

    def test_all_zeros(self):
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0])
        fig = plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self, binary_arrays):
        y_true, y_pred, _ = binary_arrays
        fig = plot_confusion_matrix(y_true, y_pred, title="My CM")
        assert isinstance(fig, plt.Figure)


# ─────────────────────────────────────────────────────────────────
# plot_calibration_curve
# ─────────────────────────────────────────────────────────────────

class TestPlotCalibrationCurve:
    def test_returns_figure(self, binary_arrays):
        y_true, _, y_proba = binary_arrays
        fig = plot_calibration_curve(y_true, y_proba)
        assert isinstance(fig, plt.Figure)

    def test_custom_n_bins(self, binary_arrays):
        y_true, _, y_proba = binary_arrays
        fig = plot_calibration_curve(y_true, y_proba, n_bins=5)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self, binary_arrays):
        y_true, _, y_proba = binary_arrays
        fig = plot_calibration_curve(y_true, y_proba, title="Cal Curve")
        assert isinstance(fig, plt.Figure)

    def test_figure_has_two_lines(self, binary_arrays):
        y_true, _, y_proba = binary_arrays
        fig = plot_calibration_curve(y_true, y_proba)
        ax = fig.axes[0]
        # classifier line + perfect-calibration diagonal
        assert len(ax.lines) >= 2


# ─────────────────────────────────────────────────────────────────
# plot_cv_summary
# ─────────────────────────────────────────────────────────────────

class TestPlotCvSummary:
    def test_returns_figure(self, fold_results):
        fig = plot_cv_summary(fold_results)
        assert isinstance(fig, plt.Figure)

    def test_has_three_subplots(self, fold_results):
        fig = plot_cv_summary(fold_results)
        assert len(fig.axes) == 3

    def test_single_fold(self):
        results = [FoldResult(
            fold=0,
            train_start="2023-01-01", train_end="2023-06-30",
            test_start="2023-07-01", test_end="2023-07-31",
            clf_metrics={"f1": 0.6, "auc_roc": 0.8},
            reg_metrics={"rmse": 12.0, "mae": 8.0, "mape": 20.0, "r2": 0.5},
        )]
        fig = plot_cv_summary(results)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self, fold_results):
        fig = plot_cv_summary(fold_results, title="My CV Plot")
        assert isinstance(fig, plt.Figure)


# ─────────────────────────────────────────────────────────────────
# plot_actual_vs_predicted
# ─────────────────────────────────────────────────────────────────

class TestPlotActualVsPredicted:
    def test_returns_figure(self, price_arrays):
        y_true, y_pred = price_arrays
        fig = plot_actual_vs_predicted(y_true, y_pred)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self, price_arrays):
        y_true, y_pred = price_arrays
        fig = plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Pred")
        assert isinstance(fig, plt.Figure)

    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        fig = plot_actual_vs_predicted(y, y)
        assert isinstance(fig, plt.Figure)

    def test_single_point(self):
        # Edge case: array of length 1
        y_true = np.array([25.0])
        y_pred = np.array([30.0])
        fig = plot_actual_vs_predicted(y_true, y_pred)
        assert isinstance(fig, plt.Figure)
