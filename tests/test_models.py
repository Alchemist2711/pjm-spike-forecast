"""Tests for pjm_spike_forecast.models."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pjm_spike_forecast.models import (
    HourlyBaseline,
    PriceRegressor,
    SpikeClassifier,
    load_model,
    save_model,
)
from pjm_spike_forecast.config import ModelConfig


# ─────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def demo_df():
    from pjm_spike_forecast.data import build_demo_dataset
    return build_demo_dataset(n_days=180, seed=42)


@pytest.fixture
def featured_df(demo_df):
    from pjm_spike_forecast.features import build_feature_matrix
    return build_feature_matrix(demo_df, cfg=None)


@pytest.fixture
def feature_cols(featured_df):
    return [c for c in featured_df.columns if c not in {"spike", "lmp", "spike_threshold"}]


@pytest.fixture
def small_params_clf():
    return {
        "objective": "binary",
        "n_estimators": 5,
        "max_depth": 3,
        "verbose": -1,
        "random_state": 42,
    }


@pytest.fixture
def small_params_reg():
    return {
        "objective": "regression",
        "n_estimators": 5,
        "max_depth": 3,
        "verbose": -1,
        "random_state": 42,
    }


@pytest.fixture
def fitted_clf(featured_df, feature_cols, small_params_clf):
    X = featured_df[feature_cols]
    y = featured_df["spike"]
    clf = SpikeClassifier(params=small_params_clf)
    clf.fit(X, y)
    return clf


@pytest.fixture
def fitted_reg(featured_df, feature_cols, small_params_reg):
    X = featured_df[feature_cols]
    y = featured_df["lmp"]
    reg = PriceRegressor(params=small_params_reg)
    reg.fit(X, y)
    return reg


@pytest.fixture
def fitted_baseline(featured_df):
    bl = HourlyBaseline()
    bl.fit(featured_df)
    return bl


# ─────────────────────────────────────────────────────────────────
# 1.  HourlyBaseline
# ─────────────────────────────────────────────────────────────────

class TestHourlyBaseline:

    # ── __init__ ──

    def test_init_lookup_is_none(self):
        bl = HourlyBaseline()
        assert bl._lookup is None

    def test_init_global_mean_zero(self):
        bl = HourlyBaseline()
        assert bl._global_mean == 0.0

    def test_init_spike_threshold_zero(self):
        bl = HourlyBaseline()
        assert bl._spike_threshold == 0.0

    # ── fit ──

    def test_fit_returns_self(self, featured_df):
        bl = HourlyBaseline()
        result = bl.fit(featured_df)
        assert result is bl

    def test_fit_sets_lookup(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        assert bl._lookup is not None
        assert isinstance(bl._lookup, pd.Series)

    def test_fit_sets_global_mean(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        assert bl._global_mean == pytest.approx(featured_df["lmp"].mean())

    def test_fit_sets_spike_threshold(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        expected = featured_df["lmp"].mean() + 2 * featured_df["lmp"].std()
        assert bl._spike_threshold == pytest.approx(expected)

    def test_fit_custom_lmp_col(self, featured_df):
        df = featured_df.rename(columns={"lmp": "price"})
        bl = HourlyBaseline()
        bl.fit(df, lmp_col="price")
        assert bl._lookup is not None

    # ── predict_price ──

    def test_predict_price_length(self, featured_df, fitted_baseline):
        preds = fitted_baseline.predict_price(featured_df)
        assert len(preds) == len(featured_df)

    def test_predict_price_finite(self, featured_df, fitted_baseline):
        preds = fitted_baseline.predict_price(featured_df)
        assert np.all(np.isfinite(preds))

    def test_predict_price_varies_by_hour(self, featured_df, fitted_baseline):
        preds = fitted_baseline.predict_price(featured_df)
        assert np.std(preds) > 0

    # Branch: key not found in lookup → fall back to global_mean
    def test_predict_price_fallback_to_global_mean(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        # Create row with a (day_of_week, hour) that doesn't exist in lookup
        unseen = featured_df.iloc[[0]].copy()
        unseen["day_of_week"] = 99  # impossible weekday
        preds = bl.predict_price(unseen)
        assert preds[0] == pytest.approx(bl._global_mean)

    def test_predict_price_unfitted_raises(self, featured_df):
        bl = HourlyBaseline()
        with pytest.raises(RuntimeError, match="not been fitted"):
            bl.predict_price(featured_df)

    # ── predict_spike ──

    def test_predict_spike_binary(self, featured_df, fitted_baseline):
        preds = fitted_baseline.predict_spike(featured_df)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_spike_length(self, featured_df, fitted_baseline):
        preds = fitted_baseline.predict_spike(featured_df)
        assert len(preds) == len(featured_df)

    # Branch: price > threshold → spike = 1
    def test_predict_spike_above_threshold(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        bl._spike_threshold = -999.0  # force all to be spikes
        preds = bl.predict_spike(featured_df)
        assert np.all(preds == 1)

    # Branch: price <= threshold → spike = 0
    def test_predict_spike_below_threshold(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        bl._spike_threshold = 999_999.0  # force none to be spikes
        preds = bl.predict_spike(featured_df)
        assert np.all(preds == 0)

    # ── predict_spike_proba ──

    def test_predict_spike_proba_is_float(self, featured_df, fitted_baseline):
        proba = fitted_baseline.predict_spike_proba(featured_df)
        assert proba.dtype == float

    def test_predict_spike_proba_values(self, featured_df, fitted_baseline):
        proba = fitted_baseline.predict_spike_proba(featured_df)
        assert set(np.unique(proba)).issubset({0.0, 1.0})

    def test_predict_spike_proba_equals_spike(self, featured_df, fitted_baseline):
        proba = fitted_baseline.predict_spike_proba(featured_df)
        spike = fitted_baseline.predict_spike(featured_df)
        np.testing.assert_array_equal(proba, spike.astype(float))


# ─────────────────────────────────────────────────────────────────
# 2.  SpikeClassifier
# ─────────────────────────────────────────────────────────────────

class TestSpikeClassifier:

    # ── __init__ ──

    def test_init_model_is_none(self):
        clf = SpikeClassifier()
        assert clf.model is None

    def test_init_feature_names_empty(self):
        clf = SpikeClassifier()
        assert clf.feature_names == []

    # Branch: params=None → uses ModelConfig defaults
    def test_init_default_params(self):
        clf = SpikeClassifier(params=None)
        assert isinstance(clf.params, dict)
        assert len(clf.params) > 0

    # Branch: params provided → uses those params
    def test_init_custom_params(self, small_params_clf):
        clf = SpikeClassifier(params=small_params_clf)
        assert clf.params["n_estimators"] == 5

    def test_init_params_are_copied(self, small_params_clf):
        clf = SpikeClassifier(params=small_params_clf)
        clf.params["extra"] = "modified"
        assert "extra" not in small_params_clf  # original unchanged

    # ── fit ──

    def test_fit_returns_self(self, featured_df, feature_cols, small_params_clf):
        X, y = featured_df[feature_cols], featured_df["spike"]
        clf = SpikeClassifier(params=small_params_clf)
        assert clf.fit(X, y) is clf

    def test_fit_sets_model(self, featured_df, feature_cols, small_params_clf):
        X, y = featured_df[feature_cols], featured_df["spike"]
        clf = SpikeClassifier(params=small_params_clf)
        clf.fit(X, y)
        assert clf.model is not None

    def test_fit_stores_feature_names(self, featured_df, feature_cols, small_params_clf):
        X, y = featured_df[feature_cols], featured_df["spike"]
        clf = SpikeClassifier(params=small_params_clf)
        clf.fit(X, y)
        assert clf.feature_names == feature_cols

    def test_fit_sets_scale_pos_weight(self, featured_df, feature_cols, small_params_clf):
        X, y = featured_df[feature_cols], featured_df["spike"]
        clf = SpikeClassifier(params=small_params_clf)
        clf.fit(X, y)
        assert "scale_pos_weight" in clf.params

    def test_scale_pos_weight_value(self, featured_df, feature_cols, small_params_clf):
        X, y = featured_df[feature_cols], featured_df["spike"]
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        clf = SpikeClassifier(params=small_params_clf)
        clf.fit(X, y)
        assert clf.params["scale_pos_weight"] == pytest.approx(n_neg / n_pos)

    # Branch: n_pos == 0 → scale_pos_weight not set
    def test_fit_no_positive_labels_no_scale_pos_weight(self, featured_df, feature_cols, small_params_clf):
        X = featured_df[feature_cols]
        y = pd.Series(np.zeros(len(X), dtype=int), index=X.index)
        clf = SpikeClassifier(params=small_params_clf)
        clf.fit(X, y)
        assert "scale_pos_weight" not in clf.params

    # ── predict ──

    def test_predict_length(self, featured_df, feature_cols, fitted_clf):
        preds = fitted_clf.predict(featured_df[feature_cols])
        assert len(preds) == len(featured_df)

    def test_predict_binary_values(self, featured_df, feature_cols, fitted_clf):
        preds = fitted_clf.predict(featured_df[feature_cols])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_unfitted_raises(self, featured_df, feature_cols):
        clf = SpikeClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(featured_df[feature_cols])

    # ── predict_proba ──

    def test_predict_proba_range(self, featured_df, feature_cols, fitted_clf):
        proba = fitted_clf.predict_proba(featured_df[feature_cols])
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_proba_length(self, featured_df, feature_cols, fitted_clf):
        proba = fitted_clf.predict_proba(featured_df[feature_cols])
        assert len(proba) == len(featured_df)

    def test_predict_proba_unfitted_raises(self, featured_df, feature_cols):
        clf = SpikeClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(featured_df[feature_cols])

    # ── feature_importances_ ──

    def test_feature_importances_length(self, feature_cols, fitted_clf):
        imp = fitted_clf.feature_importances_
        assert imp is not None
        assert len(imp) == len(feature_cols)

    def test_feature_importances_non_negative(self, fitted_clf):
        imp = fitted_clf.feature_importances_
        assert np.all(imp >= 0)

    # Branch: model is None → returns None
    def test_feature_importances_unfitted_none(self):
        clf = SpikeClassifier()
        assert clf.feature_importances_ is None


# ─────────────────────────────────────────────────────────────────
# 3.  PriceRegressor
# ─────────────────────────────────────────────────────────────────

class TestPriceRegressor:

    # ── __init__ ──

    def test_init_model_is_none(self):
        reg = PriceRegressor()
        assert reg.model is None

    def test_init_feature_names_empty(self):
        reg = PriceRegressor()
        assert reg.feature_names == []

    # Branch: params=None → uses ModelConfig defaults
    def test_init_default_params(self):
        reg = PriceRegressor(params=None)
        assert isinstance(reg.params, dict)
        assert len(reg.params) > 0

    # Branch: params provided → uses those params
    def test_init_custom_params(self, small_params_reg):
        reg = PriceRegressor(params=small_params_reg)
        assert reg.params["n_estimators"] == 5

    def test_init_params_are_copied(self, small_params_reg):
        reg = PriceRegressor(params=small_params_reg)
        reg.params["extra"] = "modified"
        assert "extra" not in small_params_reg

    # ── fit ──

    def test_fit_returns_self(self, featured_df, feature_cols, small_params_reg):
        X, y = featured_df[feature_cols], featured_df["lmp"]
        reg = PriceRegressor(params=small_params_reg)
        assert reg.fit(X, y) is reg

    def test_fit_sets_model(self, featured_df, feature_cols, small_params_reg):
        X, y = featured_df[feature_cols], featured_df["lmp"]
        reg = PriceRegressor(params=small_params_reg)
        reg.fit(X, y)
        assert reg.model is not None

    def test_fit_stores_feature_names(self, featured_df, feature_cols, small_params_reg):
        X, y = featured_df[feature_cols], featured_df["lmp"]
        reg = PriceRegressor(params=small_params_reg)
        reg.fit(X, y)
        assert reg.feature_names == feature_cols

    # ── predict ──

    def test_predict_length(self, featured_df, feature_cols, fitted_reg):
        preds = fitted_reg.predict(featured_df[feature_cols])
        assert len(preds) == len(featured_df)

    def test_predict_finite(self, featured_df, feature_cols, fitted_reg):
        preds = fitted_reg.predict(featured_df[feature_cols])
        assert np.all(np.isfinite(preds))

    def test_predict_positive_values(self, featured_df, feature_cols, fitted_reg):
        preds = fitted_reg.predict(featured_df[feature_cols])
        # LMP prices are overwhelmingly positive
        assert np.median(preds) > 0

    def test_predictions_correlate_with_target(self, featured_df, feature_cols):
        X, y = featured_df[feature_cols], featured_df["lmp"]
        reg = PriceRegressor(params={
            "objective": "regression",
            "n_estimators": 50,
            "max_depth": 4,
            "verbose": -1,
            "random_state": 42,
        })
        reg.fit(X, y)
        preds = reg.predict(X)
        corr = np.corrcoef(y.values, preds)[0, 1]
        assert corr > 0.5

    def test_predict_unfitted_raises(self, featured_df, feature_cols):
        reg = PriceRegressor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            reg.predict(featured_df[feature_cols])

    # ── feature_importances_ ──

    def test_feature_importances_length(self, feature_cols, fitted_reg):
        imp = fitted_reg.feature_importances_
        assert imp is not None
        assert len(imp) == len(feature_cols)

    def test_feature_importances_non_negative(self, fitted_reg):
        assert np.all(fitted_reg.feature_importances_ >= 0)

    # Branch: model is None → returns None
    def test_feature_importances_unfitted_none(self):
        reg = PriceRegressor()
        assert reg.feature_importances_ is None


# ─────────────────────────────────────────────────────────────────
# 4.  save_model / load_model
# ─────────────────────────────────────────────────────────────────

class TestPersistence:

    def test_save_and_load_classifier_predictions_match(
        self, featured_df, feature_cols, fitted_clf, tmp_path
    ):
        X = featured_df[feature_cols]
        path = tmp_path / "clf.joblib"
        save_model(fitted_clf, path)
        loaded = load_model(path)
        np.testing.assert_array_equal(
            fitted_clf.predict(X), loaded.predict(X)
        )

    def test_save_and_load_regressor_predictions_match(
        self, featured_df, feature_cols, fitted_reg, tmp_path
    ):
        X = featured_df[feature_cols]
        path = tmp_path / "reg.joblib"
        save_model(fitted_reg, path)
        loaded = load_model(path)
        np.testing.assert_array_almost_equal(
            fitted_reg.predict(X), loaded.predict(X)
        )

    def test_save_and_load_baseline(self, featured_df, fitted_baseline, tmp_path):
        path = tmp_path / "baseline.joblib"
        save_model(fitted_baseline, path)
        loaded = load_model(path)
        np.testing.assert_array_almost_equal(
            fitted_baseline.predict_price(featured_df),
            loaded.predict_price(featured_df),
        )

    def test_save_creates_file(self, fitted_reg, tmp_path):
        path = tmp_path / "model.joblib"
        save_model(fitted_reg, path)
        assert path.exists()

    # Branch: parent directory does not exist → mkdir creates it
    def test_save_creates_parent_dirs(self, fitted_reg, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "model.joblib"
        save_model(fitted_reg, deep_path)
        assert deep_path.exists()

    # Branch: path provided as str → converted to Path internally
    def test_save_accepts_string_path(self, fitted_reg, tmp_path):
        path_str = str(tmp_path / "model_str.joblib")
        save_model(fitted_reg, path_str)
        assert Path(path_str).exists()

    def test_load_returns_correct_type_classifier(self, fitted_clf, tmp_path):
        path = tmp_path / "clf.joblib"
        save_model(fitted_clf, path)
        loaded = load_model(path)
        assert isinstance(loaded, SpikeClassifier)

    def test_load_returns_correct_type_regressor(self, fitted_reg, tmp_path):
        path = tmp_path / "reg.joblib"
        save_model(fitted_reg, path)
        loaded = load_model(path)
        assert isinstance(loaded, PriceRegressor)

    def test_loaded_clf_proba_matches(
        self, featured_df, feature_cols, fitted_clf, tmp_path
    ):
        X = featured_df[feature_cols]
        path = tmp_path / "clf_proba.joblib"
        save_model(fitted_clf, path)
        loaded = load_model(path)
        np.testing.assert_array_almost_equal(
            fitted_clf.predict_proba(X), loaded.predict_proba(X)
        )

    def test_loaded_model_feature_importances_match(
        self, fitted_reg, feature_cols, tmp_path
    ):
        path = tmp_path / "reg_imp.joblib"
        save_model(fitted_reg, path)
        loaded = load_model(path)
        np.testing.assert_array_equal(
            fitted_reg.feature_importances_,
            loaded.feature_importances_,
        )

    def test_overwrite_existing_file(self, fitted_reg, fitted_clf, tmp_path):
        path = tmp_path / "model.joblib"
        save_model(fitted_reg, path)
        save_model(fitted_clf, path)  # overwrite
        loaded = load_model(path)
        assert isinstance(loaded, SpikeClassifier)