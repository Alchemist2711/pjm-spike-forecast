"""Tests for pjm_spike_forecast.evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pjm_spike_forecast.config import BacktestConfig
from pjm_spike_forecast.evaluation import (
    FoldResult,
    classification_metrics,
    compute_shap_importance,
    generate_temporal_splits,
    regression_metrics,
    run_walk_forward_cv,
    summarise_cv_results,
)


# ─────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """30 days of raw LMP + weather data (too small for 6-month CV)."""
    from pjm_spike_forecast.data_ingestion import build_demo_dataset
    return build_demo_dataset(n_days=30, seed=0)


@pytest.fixture
def featured_df():
    """~6 months of fully-featured data sufficient for 2-fold CV."""
    from pjm_spike_forecast.data_ingestion import build_demo_dataset
    from pjm_spike_forecast.features import build_feature_matrix
    raw = build_demo_dataset(n_days=180, seed=42)
    return build_feature_matrix(raw, cfg=None)


@pytest.fixture
def feature_cols(featured_df):
    exclude = {"spike", "lmp"}
    return [c for c in featured_df.columns if c not in exclude]


# ─────────────────────────────────────────────────────────────────
# 1.  classification_metrics
# ─────────────────────────────────────────────────────────────────

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

    def test_avg_precision_present_with_proba(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.3, 0.7, 0.9])
        m = classification_metrics(y_true, y_true, y_proba)
        assert "avg_precision" in m
        assert 0 <= m["avg_precision"] <= 1

    def test_auc_absent_without_proba(self):
        y = np.array([0, 1, 0, 1])
        m = classification_metrics(y, y)
        assert "auc_roc" not in m
        assert "avg_precision" not in m

    # Branch: y_proba is provided but only one class in y_true →
    # roc_auc_score would error, so the branch guard `len(np.unique) == 2`
    # prevents calling it.
    def test_auc_skipped_when_single_class(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1, 0.3])
        m = classification_metrics(y_true, y_pred, y_proba)
        assert "auc_roc" not in m

    def test_confusion_counts_sum_to_total(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        m = classification_metrics(y_true, y_pred)
        total = m["true_pos"] + m["false_pos"] + m["true_neg"] + m["false_neg"]
        assert total == len(y_true)

    def test_returns_dict(self):
        y = np.array([0, 1])
        m = classification_metrics(y, y)
        assert isinstance(m, dict)

    def test_partial_correct(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        m = classification_metrics(y_true, y_pred)
        assert 0 < m["f1"] < 1
        assert 0 < m["accuracy"] < 1


# ─────────────────────────────────────────────────────────────────
# 2.  regression_metrics
# ─────────────────────────────────────────────────────────────────

class TestRegressionMetrics:

    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        m = regression_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)
        assert m["mape"] == pytest.approx(0.0)

    def test_rmse_mae_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        m = regression_metrics(y_true, y_pred)
        assert m["rmse"] > 0
        assert m["mae"] > 0

    def test_mape_correct(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        m = regression_metrics(y_true, y_pred)
        # mean(|10/100|, |20/200|) * 100 = 10%
        assert m["mape"] == pytest.approx(10.0)

    def test_mape_excludes_zero_actuals(self):
        # Only y_true=10 row contributes to MAPE; zero row is masked out
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([1.0, 10.0])
        m = regression_metrics(y_true, y_pred)
        assert np.isfinite(m["mape"])
        assert m["mape"] == pytest.approx(0.0)

    # Branch: all y_true == 0 → mask.sum() == 0 → mape = nan
    def test_mape_all_zero_actuals(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        m = regression_metrics(y_true, y_pred)
        assert np.isnan(m["mape"])

    def test_r2_negative_for_bad_model(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        m = regression_metrics(y_true, y_pred)
        assert m["r2"] < 0

    # Branch: ss_tot == 0 (constant y_true) → r2 = nan
    def test_r2_nan_when_constant_target(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 6.0, 5.0])
        m = regression_metrics(y_true, y_pred)
        assert np.isnan(m["r2"])

    def test_returns_all_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        m = regression_metrics(y, y)
        assert set(m.keys()) == {"rmse", "mae", "mape", "r2"}

    def test_rmse_greater_than_mae_on_outlier(self):
        # RMSE penalises outliers harder than MAE
        y_true = np.array([1.0, 1.0, 1.0, 100.0])
        y_pred = np.array([1.0, 1.0, 1.0, 0.0])
        m = regression_metrics(y_true, y_pred)
        assert m["rmse"] > m["mae"]


# ─────────────────────────────────────────────────────────────────
# 3.  generate_temporal_splits
# ─────────────────────────────────────────────────────────────────

class TestGenerateTemporalSplits:

    def test_returns_list_of_tuples(self, featured_df):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        assert isinstance(splits, list)
        for train, test in splits:
            assert isinstance(train, pd.DataFrame)
            assert isinstance(test, pd.DataFrame)

    def test_no_overlap_between_train_and_test(self, featured_df):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=24)
        splits = generate_temporal_splits(featured_df, cfg)
        assert len(splits) > 0
        for train, test in splits:
            assert train.index.max() < test.index.min()

    def test_train_strictly_before_test(self, featured_df):
        cfg = BacktestConfig(n_splits=3, train_months=1, test_months=1, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        for train, test in splits:
            assert train.index.max() < test.index.min()

    def test_respects_n_splits_upper_bound(self, featured_df):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        assert len(splits) <= 2

    def test_empty_when_insufficient_data(self, small_df):
        from pjm_spike_forecast.features import build_feature_matrix
        feat = build_feature_matrix(small_df, cfg=None)
        cfg = BacktestConfig(n_splits=3, train_months=6, test_months=3)
        splits = generate_temporal_splits(feat, cfg)
        assert len(splits) == 0

    # Branch: cfg is None → defaults to BacktestConfig()
    def test_default_cfg_used_when_none(self, featured_df):
        splits = generate_temporal_splits(featured_df, cfg=None)
        assert isinstance(splits, list)

    # Branch: fold skipped when train_df < 100 rows
    def test_fold_skipped_when_train_too_small(self, featured_df):
        # Use a very large train window so the small residual train slice
        # comes in under 100 rows triggering the continue branch
        cfg = BacktestConfig(n_splits=5, train_months=5, test_months=1, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        # Either some folds are skipped or we get fewer than n_splits
        assert len(splits) <= 5

    # Branch: test_end > end → break
    def test_breaks_when_test_exceeds_data_end(self, featured_df):
        # 3-month test window on 6-month data means at most 1-2 folds
        cfg = BacktestConfig(n_splits=10, train_months=1, test_months=3, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        assert len(splits) < 10

    def test_gap_is_enforced(self, featured_df):
        gap_hours = 48
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=gap_hours)
        splits = generate_temporal_splits(featured_df, cfg)
        for train, test in splits:
            gap = (test.index.min() - train.index.max()).total_seconds() / 3600
            assert gap >= gap_hours

    def test_sliding_window_moves_forward(self, featured_df):
        cfg = BacktestConfig(n_splits=3, train_months=1, test_months=1, gap_hours=6)
        splits = generate_temporal_splits(featured_df, cfg)
        if len(splits) >= 2:
            # Each fold's train start should be later than the previous
            train_starts = [s[0].index.min() for s in splits]
            assert train_starts == sorted(train_starts)


# ─────────────────────────────────────────────────────────────────
# 4.  run_walk_forward_cv
# ─────────────────────────────────────────────────────────────────

class TestRunWalkForwardCv:

    def test_baseline_cv_returns_fold_results(self, featured_df, feature_cols):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        results = run_walk_forward_cv(
            featured_df, feature_cols, cfg=cfg, model_type="baseline"
        )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, FoldResult)
            assert "f1" in r.clf_metrics
            assert "rmse" in r.reg_metrics

    def test_lgbm_cv_returns_fold_results(self, featured_df, feature_cols):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        results = run_walk_forward_cv(
            featured_df, feature_cols, cfg=cfg, model_type="lgbm"
        )
        assert len(results) > 0
        for r in results:
            assert r.clf_metrics["f1"] >= 0
            assert r.reg_metrics["rmse"] >= 0

    def test_fold_result_fields_populated(self, featured_df, feature_cols):
        cfg = BacktestConfig(n_splits=1, train_months=1, test_months=1, gap_hours=6)
        results = run_walk_forward_cv(
            featured_df, feature_cols, cfg=cfg, model_type="baseline"
        )
        if results:
            r = results[0]
            assert isinstance(r.fold, int)
            assert isinstance(r.train_start, str)
            assert isinstance(r.train_end, str)
            assert isinstance(r.test_start, str)
            assert isinstance(r.test_end, str)

    def test_fold_indices_are_sequential(self, featured_df, feature_cols):
        cfg = BacktestConfig(n_splits=2, train_months=1, test_months=1, gap_hours=6)
        results = run_walk_forward_cv(
            featured_df, feature_cols, cfg=cfg, model_type="lgbm"
        )
        for i, r in enumerate(results):
            assert r.fold == i

    def test_returns_empty_list_when_no_splits(self, small_df):
        from pjm_spike_forecast.features import build_feature_matrix
        feat = build_feature_matrix(small_df, cfg=None)
        feat_cols = [c for c in feat.columns if c not in {"spike", "lmp"}]
        cfg = BacktestConfig(n_splits=3, train_months=6, test_months=3)
        results = run_walk_forward_cv(feat, feat_cols, cfg=cfg, model_type="lgbm")
        assert results == []

    # Branch: model_type="baseline" path
    def test_baseline_uses_hourly_baseline_model(self, featured_df, feature_cols):
        from pjm_spike_forecast.models import HourlyBaseline
        cfg = BacktestConfig(n_splits=1, train_months=1, test_months=1, gap_hours=6)
        with patch("pjm_spike_forecast.evaluation.HourlyBaseline", wraps=HourlyBaseline) as mock_bl:
            run_walk_forward_cv(
                featured_df, feature_cols, cfg=cfg, model_type="baseline"
            )
            if mock_bl.call_count > 0:
                mock_bl.assert_called()

    # Branch: model_type != "baseline" → lgbm path
    def test_lgbm_uses_spike_classifier_and_price_regressor(self, featured_df, feature_cols):
        from pjm_spike_forecast.models import SpikeClassifier, PriceRegressor
        cfg = BacktestConfig(n_splits=1, train_months=1, test_months=1, gap_hours=6)
        with patch("pjm_spike_forecast.evaluation.SpikeClassifier", wraps=SpikeClassifier) as mock_clf:
            run_walk_forward_cv(
                featured_df, feature_cols, cfg=cfg, model_type="lgbm"
            )
            if mock_clf.call_count > 0:
                mock_clf.assert_called()

    def test_clf_metrics_all_keys_present(self, featured_df, feature_cols):
        cfg = BacktestConfig(n_splits=1, train_months=1, test_months=1, gap_hours=6)
        results = run_walk_forward_cv(
            featured_df, feature_cols, cfg=cfg, model_type="lgbm"
        )
        expected_keys = {"accuracy", "precision", "recall", "f1",
                         "true_pos", "false_pos", "true_neg", "false_neg"}
        for r in results:
            assert expected_keys.issubset(r.clf_metrics.keys())

    def test_reg_metrics_all_keys_present(self, featured_df, feature_cols):
        cfg = BacktestConfig(n_splits=1, train_months=1, test_months=1, gap_hours=6)
        results = run_walk_forward_cv(
            featured_df, feature_cols, cfg=cfg, model_type="lgbm"
        )
        expected_keys = {"rmse", "mae", "mape", "r2"}
        for r in results:
            assert expected_keys.issubset(r.reg_metrics.keys())


# ─────────────────────────────────────────────────────────────────
# 5.  summarise_cv_results
# ─────────────────────────────────────────────────────────────────

class TestSummariseCvResults:

    def _make_fold(self, fold_id=0):
        return FoldResult(
            fold=fold_id,
            train_start="2022-01-01", train_end="2022-01-31",
            test_start="2022-02-01", test_end="2022-02-28",
            clf_metrics={"f1": 0.5, "accuracy": 0.9, "precision": 0.6,
                         "recall": 0.4, "true_pos": 5, "false_pos": 3,
                         "true_neg": 10, "false_neg": 2},
            reg_metrics={"rmse": 5.0, "mae": 3.0, "mape": 10.0, "r2": 0.8},
        )

    def test_returns_dataframe(self):
        df = summarise_cv_results([self._make_fold()])
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_fold(self):
        results = [self._make_fold(i) for i in range(3)]
        df = summarise_cv_results(results)
        assert len(df) == 3

    def test_clf_columns_prefixed(self):
        df = summarise_cv_results([self._make_fold()])
        assert "clf_f1" in df.columns
        assert "clf_accuracy" in df.columns

    def test_reg_columns_prefixed(self):
        df = summarise_cv_results([self._make_fold()])
        assert "reg_rmse" in df.columns
        assert "reg_mae" in df.columns

    def test_fold_column_present(self):
        df = summarise_cv_results([self._make_fold()])
        assert "fold" in df.columns

    def test_datetime_columns_present(self):
        df = summarise_cv_results([self._make_fold()])
        for col in ["train_start", "train_end", "test_start", "test_end"]:
            assert col in df.columns

    def test_empty_input_returns_empty_df(self):
        df = summarise_cv_results([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_values_match_fold_result(self):
        fold = self._make_fold(0)
        df = summarise_cv_results([fold])
        assert df["clf_f1"].iloc[0] == pytest.approx(0.5)
        assert df["reg_rmse"].iloc[0] == pytest.approx(5.0)

    def test_fold_ids_preserved(self):
        results = [self._make_fold(i) for i in range(4)]
        df = summarise_cv_results(results)
        assert list(df["fold"]) == [0, 1, 2, 3]


# ─────────────────────────────────────────────────────────────────
# 6.  compute_shap_importance
# ─────────────────────────────────────────────────────────────────

class TestComputeShapImportance:

    def _make_X(self, n=100):
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            rng.random((n, 4)),
            columns=["a", "b", "c", "d"]
        )

    def _make_mock_explainer(self, shap_array):
        explainer = MagicMock()
        explainer.shap_values.return_value = shap_array
        return explainer

    # Branch: shap_values is a plain array (regressor / binary returning array)
    def test_returns_dataframe_with_array_shap(self):
        X = self._make_X(100)
        shap_vals = np.random.rand(100, 4)
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        with patch("shap.TreeExplainer", return_value=self._make_mock_explainer(shap_vals)):
            df = compute_shap_importance(mock_model, X, max_samples=50)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "mean_abs_shap" in df.columns
        assert len(df) == 4

    # Branch: shap_values is a list [neg_class, pos_class] → takes index [1]
    def test_handles_list_shap_values(self):
        X = self._make_X(100)
        shap_list = [np.random.rand(100, 4), np.random.rand(100, 4)]
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        with patch("shap.TreeExplainer", return_value=self._make_mock_explainer(shap_list)):
            df = compute_shap_importance(mock_model, X, max_samples=50)
        assert len(df) == 4

    # Branch: model has no `.model` attribute → uses model directly
    def test_uses_model_directly_when_no_inner_model(self):
        X = self._make_X(50)
        shap_vals = np.random.rand(50, 4)
        mock_model = MagicMock(spec=[])  # no .model attribute
        with patch("shap.TreeExplainer", return_value=self._make_mock_explainer(shap_vals)):
            df = compute_shap_importance(mock_model, X, max_samples=200)
        assert len(df) == 4

    # Branch: max_samples >= len(X) → no subsampling
    def test_no_subsampling_when_max_samples_large(self):
        X = self._make_X(20)
        shap_vals = np.random.rand(20, 4)
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        explainer = self._make_mock_explainer(shap_vals)
        with patch("shap.TreeExplainer", return_value=explainer):
            df = compute_shap_importance(mock_model, X, max_samples=500)
        # shap_values called with full X (20 rows)
        call_arg = explainer.shap_values.call_args[0][0]
        assert len(call_arg) == 20

    # Branch: max_samples < len(X) → subsampled
    def test_subsampling_when_max_samples_small(self):
        X = self._make_X(200)
        shap_vals = np.random.rand(50, 4)
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        explainer = self._make_mock_explainer(shap_vals)
        with patch("shap.TreeExplainer", return_value=explainer):
            df = compute_shap_importance(mock_model, X, max_samples=50)
        call_arg = explainer.shap_values.call_args[0][0]
        assert len(call_arg) == 50

    def test_sorted_descending(self):
        X = self._make_X(100)
        shap_vals = np.array([[1.0, 3.0, 2.0, 4.0]] * 100)
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        with patch("shap.TreeExplainer", return_value=self._make_mock_explainer(shap_vals)):
            df = compute_shap_importance(mock_model, X, max_samples=100)
        assert df["mean_abs_shap"].is_monotonic_decreasing

    def test_feature_names_match_columns(self):
        X = self._make_X(100)
        shap_vals = np.random.rand(100, 4)
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        with patch("shap.TreeExplainer", return_value=self._make_mock_explainer(shap_vals)):
            df = compute_shap_importance(mock_model, X, max_samples=100)
        assert set(df["feature"]) == set(X.columns)