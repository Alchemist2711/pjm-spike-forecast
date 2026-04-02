"""Tests for pjm_spike_forecast.data."""

import numpy as np
import pandas as pd
import pytest

from pjm_spike_forecast.data import (
    build_demo_dataset,
    load_lmp,
    load_weather,
    merge_lmp_weather,
)


class TestBuildDemoDataset:
    """Verify the demo data generator produces structurally valid data."""

    def test_returns_dataframe(self):
        df = build_demo_dataset(n_days=10)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = build_demo_dataset(n_days=10)
        assert len(df) == 10 * 24

    def test_required_columns_present(self):
        df = build_demo_dataset(n_days=10)
        required = {"lmp", "energy", "congestion", "loss",
                     "temperature_c", "relative_humidity",
                     "wind_speed_kmh", "precipitation_mm"}
        assert required.issubset(set(df.columns))

    def test_datetime_index_is_utc(self):
        df = build_demo_dataset(n_days=10)
        assert df.index.tz is not None
        assert "UTC" in str(df.index.tz)

    def test_lmp_positive(self):
        df = build_demo_dataset(n_days=30)
        assert (df["lmp"] >= 0).all()

    def test_lmp_has_realistic_range(self):
        df = build_demo_dataset(n_days=365, seed=42)
        assert df["lmp"].median() > 10
        assert df["lmp"].median() < 60
        assert df["lmp"].max() > 50  # should have some spikes

    def test_reproducible_with_seed(self):
        a = build_demo_dataset(n_days=10, seed=42)
        b = build_demo_dataset(n_days=10, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self):
        a = build_demo_dataset(n_days=10, seed=1)
        b = build_demo_dataset(n_days=10, seed=2)
        assert not a["lmp"].equals(b["lmp"])

    def test_temperature_reasonable(self):
        df = build_demo_dataset(n_days=365)
        assert df["temperature_c"].min() > -30
        assert df["temperature_c"].max() < 50

    def test_humidity_bounded(self):
        df = build_demo_dataset(n_days=30)
        assert (df["relative_humidity"] >= 10).all()
        assert (df["relative_humidity"] <= 100).all()


class TestLoadLmp:
    """Test CSV loading for LMP data."""

    def test_roundtrip(self, tmp_path):
        """Save demo data as CSV, reload, verify schema."""
        df = build_demo_dataset(n_days=5)
        csv_path = tmp_path / "lmp.csv"

        # Simulate what download_pjm_lmp saves
        save_df = df[["lmp", "energy", "congestion", "loss"]].reset_index()
        save_df.to_csv(csv_path, index=False)

        loaded = load_lmp(csv_path)
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert "lmp" in loaded.columns
        assert len(loaded) == len(df)

    def test_deduplicates(self, tmp_path):
        """If the CSV has duplicate timestamps, keep first."""
        df = build_demo_dataset(n_days=2)
        doubled = pd.concat([df, df.head(10)]).reset_index()
        csv_path = tmp_path / "dup.csv"
        doubled.to_csv(csv_path, index=False)

        loaded = load_lmp(csv_path)
        assert not loaded.index.duplicated().any()


class TestLoadWeather:
    """Test CSV loading for weather data."""

    def test_roundtrip(self, tmp_path):
        df = build_demo_dataset(n_days=5)
        weather_cols = ["temperature_c", "relative_humidity",
                        "wind_speed_kmh", "precipitation_mm"]
        csv_path = tmp_path / "weather.csv"
        save_df = df[weather_cols].reset_index()
        save_df.to_csv(csv_path, index=False)

        loaded = load_weather(csv_path)
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert "temperature_c" in loaded.columns


class TestMergeLmpWeather:
    """Test inner-join of LMP and weather DataFrames."""

    def test_merge_same_index(self):
        df = build_demo_dataset(n_days=5)
        lmp = df[["lmp", "energy", "congestion", "loss"]]
        weather = df[["temperature_c", "relative_humidity",
                       "wind_speed_kmh", "precipitation_mm"]]
        merged = merge_lmp_weather(lmp, weather)
        assert len(merged) == len(df)
        assert "lmp" in merged.columns
        assert "temperature_c" in merged.columns

    def test_merge_partial_overlap(self):
        df = build_demo_dataset(n_days=10)
        lmp = df[["lmp"]].iloc[:120]        # first 5 days
        weather = df[["temperature_c"]].iloc[48:]  # from day 3
        merged = merge_lmp_weather(lmp, weather)
        assert len(merged) < len(df)
        assert len(merged) == 120 - 48  # overlap
