"""Tests for pjm_spike_forecast.features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pjm_spike_forecast.config import FeatureConfig
from pjm_spike_forecast.features import (
    add_lag_features,
    add_rolling_features,
    add_temporal_features,
    add_weather_features,
    build_feature_matrix,
    get_feature_columns,
    label_spikes,
)
from pjm_spike_forecast.data import build_demo_dataset


# ─────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """7 days of raw LMP + weather data."""
    return build_demo_dataset(n_days=7, seed=0)


@pytest.fixture
def demo_df():
    """365 days — enough for rolling windows and spike detection."""
    return build_demo_dataset(n_days=365, seed=42)


@pytest.fixture
def featured_df(demo_df):
    return build_feature_matrix(demo_df, cfg=None)


@pytest.fixture
def bare_lmp_df():
    """DataFrame with only an lmp column — no weather columns."""
    idx = pd.date_range("2023-01-01", periods=200, freq="h", tz="UTC")
    return pd.DataFrame({"lmp": np.random.default_rng(0).random(200) * 50 + 20}, index=idx)


# ─────────────────────────────────────────────────────────────────
# 1.  add_temporal_features
# ─────────────────────────────────────────────────────────────────

class TestTemporalFeatures:

    def test_adds_expected_columns(self, small_df):
        result = add_temporal_features(small_df)
        expected = {"hour", "day_of_week", "month", "is_weekend",
                    "hour_sin", "hour_cos", "month_sin", "month_cos"}
        assert expected.issubset(set(result.columns))

    def test_hour_range(self, small_df):
        result = add_temporal_features(small_df)
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23

    def test_day_of_week_range(self, small_df):
        result = add_temporal_features(small_df)
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_month_range(self, small_df):
        result = add_temporal_features(small_df)
        assert result["month"].min() >= 1
        assert result["month"].max() <= 12

    def test_weekend_flag_on_weekends(self, small_df):
        result = add_temporal_features(small_df)
        weekends = result[result["day_of_week"] >= 5]
        assert (weekends["is_weekend"] == 1).all()

    def test_weekend_flag_on_weekdays(self, small_df):
        result = add_temporal_features(small_df)
        weekdays = result[result["day_of_week"] < 5]
        assert (weekdays["is_weekend"] == 0).all()

    def test_cyclical_encoding_bounded(self, small_df):
        result = add_temporal_features(small_df)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            assert result[col].min() >= -1.0 - 1e-9
            assert result[col].max() <= 1.0 + 1e-9

    # hour_sin and hour_cos should satisfy sin²+cos²=1
    def test_cyclical_encoding_unit_circle(self, small_df):
        result = add_temporal_features(small_df)
        identity = result["hour_sin"] ** 2 + result["hour_cos"] ** 2
        np.testing.assert_allclose(identity.values, 1.0, atol=1e-9)

    def test_month_cyclical_unit_circle(self, demo_df):
        result = add_temporal_features(demo_df)
        identity = result["month_sin"] ** 2 + result["month_cos"] ** 2
        np.testing.assert_allclose(identity.values, 1.0, atol=1e-9)

    def test_does_not_modify_input(self, small_df):
        original_cols = set(small_df.columns)
        _ = add_temporal_features(small_df)
        assert set(small_df.columns) == original_cols

    def test_returns_copy_not_same_object(self, small_df):
        result = add_temporal_features(small_df)
        assert result is not small_df

    def test_row_count_preserved(self, small_df):
        result = add_temporal_features(small_df)
        assert len(result) == len(small_df)

    def test_works_on_df_without_weather(self, bare_lmp_df):
        result = add_temporal_features(bare_lmp_df)
        assert "hour" in result.columns


# ─────────────────────────────────────────────────────────────────
# 2.  add_lag_features
# ─────────────────────────────────────────────────────────────────

class TestLagFeatures:

    def test_creates_correct_columns(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[1, 24])
        assert "lmp_lag_1" in result.columns
        assert "lmp_lag_24" in result.columns

    def test_lag_values_correct(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[1])
        assert result["lmp_lag_1"].iloc[5] == pytest.approx(small_df["lmp"].iloc[4])

    def test_lag_2_values_correct(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[2])
        assert result["lmp_lag_2"].iloc[5] == pytest.approx(small_df["lmp"].iloc[3])

    def test_first_rows_are_nan(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[3])
        assert result["lmp_lag_3"].iloc[:3].isna().all()

    # Branch: first row of lag_1 is NaN
    def test_first_row_lag1_is_nan(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[1])
        assert pd.isna(result["lmp_lag_1"].iloc[0])

    # Branch: empty lags list → no new columns
    def test_empty_lags_returns_same_columns(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[])
        assert set(result.columns) == set(small_df.columns)

    def test_multiple_lags_created(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[1, 6, 24, 48])
        for lag in [1, 6, 24, 48]:
            assert f"lmp_lag_{lag}" in result.columns

    def test_does_not_modify_input(self, small_df):
        original_cols = list(small_df.columns)
        _ = add_lag_features(small_df, "lmp", lags=[1])
        assert list(small_df.columns) == original_cols

    def test_works_on_temperature_column(self, small_df):
        result = add_lag_features(small_df, "temperature_c", lags=[24])
        assert "temperature_c_lag_24" in result.columns

    def test_row_count_preserved(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[1, 24])
        assert len(result) == len(small_df)


# ─────────────────────────────────────────────────────────────────
# 3.  add_rolling_features
# ─────────────────────────────────────────────────────────────────

class TestRollingFeatures:

    def test_creates_correct_columns(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[6])
        for suffix in ["rmean_6", "rstd_6", "rmin_6", "rmax_6"]:
            assert f"lmp_{suffix}" in result.columns

    def test_multiple_windows(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[6, 24])
        for w in [6, 24]:
            for stat in ["rmean", "rstd", "rmin", "rmax"]:
                assert f"lmp_{stat}_{w}" in result.columns

    # Branch: empty windows list
    def test_empty_windows_returns_same_columns(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[])
        assert set(result.columns) == set(small_df.columns)

    def test_rolling_mean_within_bounds(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[6])
        valid = result["lmp_rmean_6"].dropna()
        assert valid.min() >= small_df["lmp"].min() - 1
        assert valid.max() <= small_df["lmp"].max() + 1

    def test_rolling_std_non_negative(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[6])
        valid = result["lmp_rstd_6"].dropna()
        assert (valid >= 0).all()

    def test_rolling_min_leq_mean(self, demo_df):
        result = add_rolling_features(demo_df, "lmp", windows=[24])
        valid = result.dropna(subset=["lmp_rmean_24", "lmp_rmin_24"])
        assert (valid["lmp_rmin_24"] <= valid["lmp_rmean_24"] + 1e-9).all()

    def test_rolling_max_geq_mean(self, demo_df):
        result = add_rolling_features(demo_df, "lmp", windows=[24])
        valid = result.dropna(subset=["lmp_rmean_24", "lmp_rmax_24"])
        assert (valid["lmp_rmax_24"] >= valid["lmp_rmean_24"] - 1e-9).all()

    def test_first_rows_are_nan(self, small_df):
        w = 6
        result = add_rolling_features(small_df, "lmp", windows=[w])
        assert result[f"lmp_rmean_{w}"].iloc[:w - 1].isna().all()

    def test_does_not_modify_input(self, small_df):
        original_cols = list(small_df.columns)
        _ = add_rolling_features(small_df, "lmp", windows=[6])
        assert list(small_df.columns) == original_cols

    def test_row_count_preserved(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[6])
        assert len(result) == len(small_df)


# ─────────────────────────────────────────────────────────────────
# 4.  add_weather_features
# ─────────────────────────────────────────────────────────────────

class TestWeatherFeatures:

    def test_adds_temp_squared(self, small_df):
        result = add_weather_features(small_df)
        assert "temp_squared" in result.columns
        np.testing.assert_array_almost_equal(
            result["temp_squared"].values,
            (small_df["temperature_c"] ** 2).values,
        )

    def test_adds_temp_x_humidity(self, small_df):
        result = add_weather_features(small_df)
        assert "temp_x_humidity" in result.columns
        expected = small_df["temperature_c"] * small_df["relative_humidity"] / 100
        np.testing.assert_array_almost_equal(
            result["temp_x_humidity"].values, expected.values
        )

    # Branch: no temperature_c column → temp_squared skipped
    def test_no_temp_column_skips_temp_squared(self):
        df = pd.DataFrame(
            {"lmp": [10.0, 20.0], "relative_humidity": [60.0, 70.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="h", tz="UTC"),
        )
        result = add_weather_features(df)
        assert "temp_squared" not in result.columns

    # Branch: no humidity column → temp_x_humidity skipped
    def test_no_humidity_column_skips_interaction(self):
        df = pd.DataFrame(
            {"lmp": [10.0, 20.0], "temperature_c": [15.0, 20.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="h", tz="UTC"),
        )
        result = add_weather_features(df)
        assert "temp_x_humidity" not in result.columns

    # Branch: neither temperature nor humidity present
    def test_no_weather_columns_returns_unchanged(self):
        df = pd.DataFrame(
            {"lmp": [1.0, 2.0, 3.0]},
            index=pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC"),
        )
        result = add_weather_features(df)
        assert "temp_squared" not in result.columns
        assert "temp_x_humidity" not in result.columns
        assert list(result.columns) == ["lmp"]

    def test_does_not_modify_input(self, small_df):
        original_cols = set(small_df.columns)
        _ = add_weather_features(small_df)
        assert set(small_df.columns) == original_cols

    def test_temp_squared_always_non_negative(self, demo_df):
        result = add_weather_features(demo_df)
        assert (result["temp_squared"] >= 0).all()

    def test_row_count_preserved(self, small_df):
        result = add_weather_features(small_df)
        assert len(result) == len(small_df)


# ─────────────────────────────────────────────────────────────────
# 5.  label_spikes
# ─────────────────────────────────────────────────────────────────

class TestLabelSpikes:

    def test_spike_and_threshold_columns_created(self, small_df):
        result = label_spikes(small_df, window=6, n_std=2.0)
        assert "spike" in result.columns
        assert "spike_threshold" in result.columns

    def test_spike_is_binary(self, demo_df):
        result = label_spikes(demo_df, window=48, n_std=2.0)
        valid = result["spike"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_strict_threshold_fewer_spikes(self, demo_df):
        loose = label_spikes(demo_df, window=48, n_std=1.0)
        strict = label_spikes(demo_df, window=48, n_std=3.0)
        assert loose["spike"].sum() >= strict["spike"].sum()

    def test_first_rows_threshold_nan(self, small_df):
        w = 6
        result = label_spikes(small_df, window=w, n_std=2.0)
        assert result["spike_threshold"].iloc[:w - 1].isna().all()

    # Branch: lmp > threshold → spike == 1
    def test_spike_set_when_lmp_exceeds_threshold(self, demo_df):
        result = label_spikes(demo_df, window=48, n_std=0.01)
        # With tiny std multiplier, most prices will be spikes
        assert result["spike"].sum() > 0

    # Branch: lmp <= threshold → spike == 0
    def test_no_spikes_with_very_high_threshold(self, demo_df):
        result = label_spikes(demo_df, window=48, n_std=100.0)
        assert result["spike"].dropna().sum() == 0

    def test_custom_lmp_col(self, small_df):
        df = small_df.rename(columns={"lmp": "price"})
        result = label_spikes(df, lmp_col="price", window=6, n_std=2.0)
        assert "spike" in result.columns

    def test_does_not_modify_input(self, small_df):
        original_cols = set(small_df.columns)
        _ = label_spikes(small_df, window=6, n_std=2.0)
        assert set(small_df.columns) == original_cols

    def test_row_count_preserved(self, small_df):
        result = label_spikes(small_df, window=6, n_std=2.0)
        assert len(result) == len(small_df)

    def test_spike_threshold_above_rolling_mean(self, demo_df):
        result = label_spikes(demo_df, window=48, n_std=2.0)
        roll_mean = demo_df["lmp"].rolling(48, min_periods=48).mean()
        valid = result["spike_threshold"].dropna()
        valid_mean = roll_mean.dropna()
        assert (valid.values >= valid_mean.values - 1e-9).all()


# ─────────────────────────────────────────────────────────────────
# 6.  build_feature_matrix
# ─────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_no_nans(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert result.isna().sum().sum() == 0

    def test_spike_column_present(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert "spike" in result.columns

    def test_temporal_columns_present(self, demo_df):
        result = build_feature_matrix(demo_df)
        for col in ["hour", "day_of_week", "month", "is_weekend"]:
            assert col in result.columns

    def test_lag_columns_present(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert "lmp_lag_1" in result.columns

    def test_rolling_columns_present(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert "lmp_rmean_24" in result.columns

    def test_weather_feature_columns_present(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert "temp_squared" in result.columns
        assert "temp_x_humidity" in result.columns

    def test_fewer_rows_than_input(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert len(result) < len(demo_df)

    def test_custom_config(self, demo_df):
        cfg = FeatureConfig(lmp_lags=[1, 2], rolling_windows=[6],
                            spike_rolling_window=24)
        result = build_feature_matrix(demo_df, cfg)
        assert "lmp_lag_1" in result.columns
        assert "lmp_lag_2" in result.columns
        assert result.isna().sum().sum() == 0

    # Branch: cfg is None → uses FeatureConfig()
    def test_default_cfg_when_none(self, demo_df):
        result = build_feature_matrix(demo_df, cfg=None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    # Branch: temperature_c present → temperature lags are added
    def test_temperature_lags_added_when_present(self, demo_df):
        cfg = FeatureConfig(temp_lags=[24])
        result = build_feature_matrix(demo_df, cfg)
        assert "temperature_c_lag_24" in result.columns

    # Branch: temperature_c absent → temperature lags skipped
    def test_temperature_lags_skipped_when_absent(self, bare_lmp_df):
        cfg = FeatureConfig(lmp_lags=[1], temp_lags=[24],
                            rolling_windows=[6], spike_rolling_window=10)
        result = build_feature_matrix(bare_lmp_df, cfg)
        assert "temperature_c_lag_24" not in result.columns

    def test_index_is_datetime(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_is_sorted(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert result.index.is_monotonic_increasing

    def test_spike_values_are_binary(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert set(result["spike"].unique()).issubset({0, 1})

    def test_spike_threshold_column_present(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert "spike_threshold" in result.columns

    def test_lmp_column_preserved(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert "lmp" in result.columns
        assert (result["lmp"] > 0).all()


# ─────────────────────────────────────────────────────────────────
# 7.  get_feature_columns
# ─────────────────────────────────────────────────────────────────

class TestGetFeatureColumns:

    def test_excludes_lmp(self, featured_df):
        cols = get_feature_columns(featured_df)
        assert "lmp" not in cols

    def test_excludes_spike(self, featured_df):
        cols = get_feature_columns(featured_df)
        assert "spike" not in cols

    def test_excludes_spike_threshold(self, featured_df):
        cols = get_feature_columns(featured_df)
        assert "spike_threshold" not in cols

    def test_excludes_energy_congestion_loss(self, featured_df):
        cols = get_feature_columns(featured_df)
        for col in ["energy", "congestion", "loss"]:
            assert col not in cols

    def test_includes_hour(self, featured_df):
        cols = get_feature_columns(featured_df)
        assert "hour" in cols

    def test_includes_lag_columns(self, featured_df):
        cols = get_feature_columns(featured_df)
        assert "lmp_lag_1" in cols

    def test_returns_list(self, featured_df):
        cols = get_feature_columns(featured_df)
        assert isinstance(cols, list)

    def test_all_columns_present_in_df(self, featured_df):
        cols = get_feature_columns(featured_df)
        for c in cols:
            assert c in featured_df.columns

    # Branch: df with only excluded columns → empty list
    def test_returns_empty_for_all_excluded_columns(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"lmp": [1.0, 2.0, 3.0], "spike": [0, 0, 1],
             "spike_threshold": [30.0, 31.0, 32.0],
             "energy": [0.9, 0.9, 0.9], "congestion": [0.05, 0.05, 0.05],
             "loss": [0.05, 0.05, 0.05]},
            index=idx,
        )
        cols = get_feature_columns(df)
        assert cols == []

    # Branch: df with no excluded columns → all columns returned
    def test_returns_all_when_no_excluded_columns(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC")
        df = pd.DataFrame({"hour": [0, 1, 2], "temp": [10.0, 11.0, 12.0]}, index=idx)
        cols = get_feature_columns(df)
        assert set(cols) == {"hour", "temp"}