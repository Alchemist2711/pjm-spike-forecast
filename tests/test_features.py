"""Tests for pjm_spike_forecast.features."""

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

    def test_weekend_flag(self, small_df):
        result = add_temporal_features(small_df)
        # Saturday=5, Sunday=6
        weekends = result[result["day_of_week"] >= 5]
        assert (weekends["is_weekend"] == 1).all()
        weekdays = result[result["day_of_week"] < 5]
        assert (weekdays["is_weekend"] == 0).all()

    def test_cyclical_encoding_bounded(self, small_df):
        result = add_temporal_features(small_df)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0

    def test_does_not_modify_input(self, small_df):
        original_cols = set(small_df.columns)
        _ = add_temporal_features(small_df)
        assert set(small_df.columns) == original_cols


class TestLagFeatures:

    def test_creates_correct_columns(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[1, 24])
        assert "lmp_lag_1" in result.columns
        assert "lmp_lag_24" in result.columns

    def test_lag_values_correct(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[1])
        # Row i should have lag = row i-1 value
        assert result["lmp_lag_1"].iloc[5] == small_df["lmp"].iloc[4]

    def test_first_rows_are_nan(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[3])
        assert result["lmp_lag_3"].iloc[:3].isna().all()

    def test_empty_lags_returns_same(self, small_df):
        result = add_lag_features(small_df, "lmp", lags=[])
        assert set(result.columns) == set(small_df.columns)


class TestRollingFeatures:

    def test_creates_correct_columns(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[6])
        for suffix in ["rmean_6", "rstd_6", "rmin_6", "rmax_6"]:
            assert f"lmp_{suffix}" in result.columns

    def test_rolling_mean_reasonable(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[24])
        valid = result["lmp_rmean_24"].dropna()
        # Rolling mean should be between min and max of original
        assert valid.min() >= small_df["lmp"].min() - 1
        assert valid.max() <= small_df["lmp"].max() + 1

    def test_rolling_std_non_negative(self, small_df):
        result = add_rolling_features(small_df, "lmp", windows=[6])
        valid = result["lmp_rstd_6"].dropna()
        assert (valid >= 0).all()


class TestWeatherFeatures:

    def test_adds_temp_squared(self, small_df):
        result = add_weather_features(small_df)
        assert "temp_squared" in result.columns
        np.testing.assert_array_almost_equal(
            result["temp_squared"].values,
            (small_df["temperature_c"] ** 2).values,
        )

    def test_adds_interaction(self, small_df):
        result = add_weather_features(small_df)
        assert "temp_x_humidity" in result.columns

    def test_no_weather_columns_graceful(self):
        """If no weather columns, nothing breaks."""
        df = pd.DataFrame({"lmp": [1, 2, 3]},
                          index=pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC"))
        result = add_weather_features(df)
        assert "temp_squared" not in result.columns  # gracefully skipped


class TestLabelSpikes:

    def test_spike_column_created(self, small_df):
        result = label_spikes(small_df, window=48, n_std=2.0)
        assert "spike" in result.columns
        assert "spike_threshold" in result.columns

    def test_spike_is_binary(self, small_df):
        result = label_spikes(small_df, window=48, n_std=2.0)
        valid = result["spike"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_strict_threshold_fewer_spikes(self, demo_df):
        loose = label_spikes(demo_df, window=48, n_std=1.0)
        strict = label_spikes(demo_df, window=48, n_std=3.0)
        assert loose["spike"].dropna().sum() >= strict["spike"].dropna().sum()

    def test_first_rows_nan(self, small_df):
        result = label_spikes(small_df, window=48, n_std=2.0)
        assert result["spike_threshold"].iloc[:47].isna().all()


class TestBuildFeatureMatrix:

    def test_no_nans(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert result.isna().sum().sum() == 0

    def test_spike_column_present(self, demo_df):
        result = build_feature_matrix(demo_df)
        assert "spike" in result.columns

    def test_fewer_rows_than_input(self, demo_df):
        """Dropping NaN from lags/rolling should reduce row count."""
        result = build_feature_matrix(demo_df)
        assert len(result) < len(demo_df)

    def test_custom_config(self, small_df):
        cfg = FeatureConfig(lmp_lags=[1, 2], rolling_windows=[6],
                            spike_rolling_window=24)
        result = build_feature_matrix(small_df, cfg)
        assert "lmp_lag_1" in result.columns
        assert "lmp_lag_2" in result.columns
        assert result.isna().sum().sum() == 0


class TestGetFeatureColumns:

    def test_excludes_targets(self, featured_df):
        cols = get_feature_columns(featured_df)
        assert "lmp" not in cols
        assert "spike" not in cols
        assert "spike_threshold" not in cols

    def test_includes_features(self, featured_df):
        cols = get_feature_columns(featured_df)
        assert "hour" in cols
        assert "lmp_lag_1" in cols
