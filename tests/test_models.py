"""Tests for pjm_spike_forecast.models."""

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


class TestHourlyBaseline:

    def test_fit_and_predict_price(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        preds = bl.predict_price(featured_df)
        assert len(preds) == len(featured_df)
        assert np.all(np.isfinite(preds))

    def test_predict_spike(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        preds = bl.predict_spike(featured_df)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_spike_proba(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        proba = bl.predict_spike_proba(featured_df)
        assert set(np.unique(proba)).issubset({0.0, 1.0})

    def test_unfitted_raises(self, featured_df):
        bl = HourlyBaseline()
        with pytest.raises(RuntimeError, match="not been fitted"):
            bl.predict_price(featured_df)

    def test_predictions_vary_by_hour(self, featured_df):
        bl = HourlyBaseline()
        bl.fit(featured_df)
        preds = bl.predict_price(featured_df)
        # Not all predictions should be identical
        assert np.std(preds) > 0


class TestSpikeClassifier:

    def test_fit_predict(self, featured_df, feature_cols):
        X = featured_df[feature_cols]
        y = featured_df["spike"]
        clf = SpikeClassifier(params={
            "objective": "binary",
            "n_estimators": 5,
            "max_depth": 3,
            "verbose": -1,
            "random_state": 42,
        })
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_range(self, featured_df, feature_cols):
        X = featured_df[feature_cols]
        y = featured_df["spike"]
        clf = SpikeClassifier(params={
            "objective": "binary",
            "n_estimators": 5,
            "max_depth": 3,
            "verbose": -1,
            "random_state": 42,
        })
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_scale_pos_weight_set(self, featured_df, feature_cols):
        X = featured_df[feature_cols]
        y = featured_df["spike"]
        clf = SpikeClassifier(params={
            "objective": "binary",
            "n_estimators": 5,
            "verbose": -1,
        })
        clf.fit(X, y)
        # scale_pos_weight should have been computed
        assert "scale_pos_weight" in clf.params

    def test_feature_importances(self, featured_df, feature_cols):
        X = featured_df[feature_cols]
        y = featured_df["spike"]
        clf = SpikeClassifier(params={
            "objective": "binary",
            "n_estimators": 5,
            "verbose": -1,
        })
        clf.fit(X, y)
        imp = clf.feature_importances_
        assert imp is not None
        assert len(imp) == len(feature_cols)

    def test_unfitted_raises(self, featured_df, feature_cols):
        clf = SpikeClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(featured_df[feature_cols])

    def test_unfitted_proba_raises(self, featured_df, feature_cols):
        clf = SpikeClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(featured_df[feature_cols])

    def test_unfitted_importances_none(self):
        clf = SpikeClassifier()
        assert clf.feature_importances_ is None


class TestPriceRegressor:

    def test_fit_predict(self, featured_df, feature_cols):
        X = featured_df[feature_cols]
        y = featured_df["lmp"]
        reg = PriceRegressor(params={
            "objective": "regression",
            "n_estimators": 5,
            "max_depth": 3,
            "verbose": -1,
            "random_state": 42,
        })
        reg.fit(X, y)
        preds = reg.predict(X)
        assert len(preds) == len(X)
        assert np.all(np.isfinite(preds))

    def test_predictions_correlate_with_target(self, featured_df, feature_cols):
        X = featured_df[feature_cols]
        y = featured_df["lmp"]
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
        assert corr > 0.5  # should learn something

    def test_unfitted_raises(self, featured_df, feature_cols):
        reg = PriceRegressor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            reg.predict(featured_df[feature_cols])

    def test_feature_importances(self, featured_df, feature_cols):
        X = featured_df[feature_cols]
        y = featured_df["lmp"]
        reg = PriceRegressor(params={
            "objective": "regression",
            "n_estimators": 5,
            "verbose": -1,
        })
        reg.fit(X, y)
        assert reg.feature_importances_ is not None
        assert len(reg.feature_importances_) == len(feature_cols)

    def test_unfitted_importances_none(self):
        reg = PriceRegressor()
        assert reg.feature_importances_ is None


class TestPersistence:

    def test_save_and_load_classifier(self, featured_df, feature_cols, tmp_path):
        X = featured_df[feature_cols]
        y = featured_df["spike"]
        clf = SpikeClassifier(params={
            "objective": "binary",
            "n_estimators": 5,
            "verbose": -1,
        })
        clf.fit(X, y)

        path = tmp_path / "clf.joblib"
        save_model(clf, path)
        loaded = load_model(path)

        np.testing.assert_array_equal(
            clf.predict(X), loaded.predict(X)
        )

    def test_save_and_load_regressor(self, featured_df, feature_cols, tmp_path):
        X = featured_df[feature_cols]
        y = featured_df["lmp"]
        reg = PriceRegressor(params={
            "objective": "regression",
            "n_estimators": 5,
            "verbose": -1,
        })
        reg.fit(X, y)

        path = tmp_path / "reg.joblib"
        save_model(reg, path)
        loaded = load_model(path)

        np.testing.assert_array_almost_equal(
            reg.predict(X), loaded.predict(X)
        )

    def test_save_creates_parent_dirs(self, featured_df, feature_cols, tmp_path):
        X = featured_df[feature_cols]
        y = featured_df["lmp"]
        reg = PriceRegressor(params={
            "objective": "regression",
            "n_estimators": 5,
            "verbose": -1,
        })
        reg.fit(X, y)

        deep_path = tmp_path / "a" / "b" / "model.joblib"
        save_model(reg, deep_path)
        assert deep_path.exists()
