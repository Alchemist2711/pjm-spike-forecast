"""Tests for pjm_spike_forecast.evaluation."""

import numpy as np
import pandas as pd
import pytest

from pjm_spike_forecast.config import BacktestConfig
from pjm_spike_forecast.evaluation import (
    FoldResult,
    classification_metrics,
    generate_temporal_splits,
    regression_metrics,
    run_walk_forward_cv,
    summarise_cv_results,
)


# ──────────────────────────────────────────────────────────────────
# Classification metrics
# ──────────────────────────────────────────────────────────────────

class TestClassificationMetrics:

    def test_perfect_prediction(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        m = classification_metrics(y, y, y.astype(float))
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        m = classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["true_pos"] == 0
        assert m["false_pos"] == 2
        assert m["false_neg"] == 2

    def test_no_positives_in_pred(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        m = classification_metrics(y_true, y_pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_auc_present_with_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.3, 0.7, 0.9])
        m = classification_metrics(y_true, y_true, y_proba)
        assert "auc_roc" in m
        assert 0 <= m["auc_roc"] <= 1

    def test_auc_absent_without_proba(self):
        y = np.array([0, 1, 0, 1])
        m = classification_metrics(y, y)
        assert "auc_roc" not in m

    def test_confusion_counts_sum(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        m = classification_metrics(y_true, y_pred)
        total = m["true_pos"] + m["false_pos"] + m["true_neg"] + m["false_neg"]
        assert total == len(y_true)


# ──────────────────────────────────────────────────────────────────
# Regression metrics
# ──────────────────────────────────────────────────────────────────

class TestRegressionMetrics:

    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        m = regression_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_rmse_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        m = regression_metrics(y_true, y_pred)
        assert m["rmse"] > 0
        assert m["mae"] > 0

    def test_mape_correct(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        m = regression_metrics(y_true, y_pred)
        # MAPE = mean(|10/100|, |20/200|) * 100 = mean(0.1, 0.1)*100 = 10%
        assert m["mape"] == pytest.approx(10.0)

    def test_mape_handles_zero(self):
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([1.0, 10.0])
        m = regression_metrics(y_true, y_pred)
        # Zero-value rows are excluded from MAPE
        assert np.isfinite(m["mape"])

    def test_r2_negative_for_bad_model(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])  # way off
        m = regression_metrics(y_true, y_pred)
        assert m["r2"] < 0


# ──────────────────────────────────────────────────────────────────
# Temporal splits
# ──────────────────────────────────────────────────────────────────

class TestGenerateTemporalSplits:

    def test_returns_list_of_tuples(self, featured_df):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        assert isinstance(splits, list)
        for train, test in splits:
            assert isinstance(train, pd.DataFrame)
            assert isinstance(test, pd.DataFrame)

    def test_no_overlap(self, featured_df):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=24)
        splits = generate_temporal_splits(featured_df, cfg)
        for train, test in splits:
            assert train.index.max() < test.index.min()

    def test_train_before_test(self, featured_df):
        cfg = BacktestConfig(n_splits=3, train_months=1, test_months=1, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        for train, test in splits:
            assert train.index.max() < test.index.min()

    def test_respects_n_splits(self, featured_df):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        assert len(splits) <= 2

    def test_empty_if_not_enough_data(self, small_df):
        """30 days isn't enough for 6-month train windows."""
        from pjm_spike_forecast.features import build_feature_matrix
        feat = build_feature_matrix(small_df, cfg=None)
        cfg = BacktestConfig(n_splits=3, train_months=6, test_months=3)
        splits = generate_temporal_splits(feat, cfg)
        assert len(splits) == 0


# ──────────────────────────────────────────────────────────────────
# Walk-forward CV
# ──────────────────────────────────────────────────────────────────

class TestWalkForwardCV:

    def test_baseline_cv(self, featured_df, feature_cols):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        results = run_walk_forward_cv(
            featured_df, feature_cols, cfg=cfg, model_type="baseline"
        )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, FoldResult)
            assert "f1" in r.clf_metrics
            assert "rmse" in r.reg_metrics

    def test_lgbm_cv(self, featured_df, feature_cols):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        results = run_walk_forward_cv(
            featured_df, feature_cols, cfg=cfg, model_type="lgbm"
        )
        assert len(results) > 0
        for r in results:
            assert r.clf_metrics["f1"] >= 0
            assert r.reg_metrics["rmse"] >= 0


class TestSummariseCvResults:

    def test_produces_dataframe(self):
        results = [
            FoldResult(fold=0, train_start="a", train_end="b",
                       test_start="c", test_end="d",
                       clf_metrics={"f1": 0.5, "accuracy": 0.9},
                       reg_metrics={"rmse": 5.0, "mae": 3.0}),
        ]
        df = summarise_cv_results(results)
        assert isinstance(df, pd.DataFrame)
        assert "clf_f1" in df.columns
        assert "reg_rmse" in df.columns
        assert len(df) == 1
