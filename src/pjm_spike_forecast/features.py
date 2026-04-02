"""
Feature engineering for PJM LMP spike forecasting.

Every function takes a DataFrame and returns an augmented copy,
making the pipeline composable and each transformation independently
testable.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from pjm_spike_forecast.config import FeatureConfig


# ──────────────────────────────────────────────────────────────────
# Temporal / calendar features
# ──────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived features from the datetime index.

    New columns: ``hour``, ``day_of_week``, ``month``, ``is_weekend``,
    ``hour_sin``, ``hour_cos``, ``month_sin``, ``month_cos``.
    """
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["day_of_week"] = idx.weekday
    out["month"] = idx.month
    out["is_weekend"] = (idx.weekday >= 5).astype(int)

    # Cyclical encoding prevents the model from seeing
    # hour 23 and hour 0 as maximally distant.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


# ──────────────────────────────────────────────────────────────────
# Lag features
# ──────────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: List[int],
) -> pd.DataFrame:
    """Create lag features for a given column.

    Parameters
    ----------
    df : DataFrame
        Must be sorted by datetime index (ascending).
    column : str
        Source column name.
    lags : list[int]
        Lag offsets in number of rows (hours, if hourly data).

    Returns
    -------
    DataFrame with new columns ``{column}_lag_{k}`` for each k in *lags*.
    """
    out = df.copy()
    for lag in lags:
        out[f"{column}_lag_{lag}"] = out[column].shift(lag)
    return out


# ──────────────────────────────────────────────────────────────────
# Rolling statistics
# ──────────────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: List[int],
) -> pd.DataFrame:
    """Compute rolling mean, std, min, max for *column*.

    Parameters
    ----------
    column : str
        Source column.
    windows : list[int]
        Window sizes in rows.

    Returns
    -------
    DataFrame with columns ``{column}_rmean_{w}``, ``{column}_rstd_{w}``,
    ``{column}_rmin_{w}``, ``{column}_rmax_{w}`` for each window *w*.
    """
    out = df.copy()
    series = out[column]

    for w in windows:
        roll = series.rolling(window=w, min_periods=w)
        out[f"{column}_rmean_{w}"] = roll.mean()
        out[f"{column}_rstd_{w}"] = roll.std()
        out[f"{column}_rmin_{w}"] = roll.min()
        out[f"{column}_rmax_{w}"] = roll.max()

    return out


# ──────────────────────────────────────────────────────────────────
# Weather-derived features
# ──────────────────────────────────────────────────────────────────

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-linear weather transformations.

    - ``temp_squared``: captures the U-shaped relationship between
      temperature and electricity demand (heating + cooling).
    - ``temp_x_humidity``: heat-index proxy.
    """
    out = df.copy()

    if "temperature_c" in out.columns:
        out["temp_squared"] = out["temperature_c"] ** 2
    if "temperature_c" in out.columns and "relative_humidity" in out.columns:
        out["temp_x_humidity"] = out["temperature_c"] * out["relative_humidity"] / 100

    return out


# ──────────────────────────────────────────────────────────────────
# Spike labelling
# ──────────────────────────────────────────────────────────────────

def label_spikes(
    df: pd.DataFrame,
    lmp_col: str = "lmp",
    window: int = 168,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Create a binary ``spike`` label.

    A spike is defined as an hour where the LMP exceeds a *rolling*
    threshold = rolling_mean + n_std × rolling_std, computed over
    the preceding *window* hours.  Using a rolling threshold (rather
    than a static one) accounts for seasonal level shifts.

    Parameters
    ----------
    lmp_col : str
        Column containing the LMP series.
    window : int
        Rolling window size in hours.
    n_std : float
        Number of standard deviations above the rolling mean.

    Returns
    -------
    DataFrame with new columns ``spike_threshold`` and ``spike``.
    """
    out = df.copy()
    roll = out[lmp_col].rolling(window=window, min_periods=window)
    out["spike_threshold"] = roll.mean() + n_std * roll.std()
    out["spike"] = (out[lmp_col] > out["spike_threshold"]).astype(int)
    return out


# ──────────────────────────────────────────────────────────────────
# Full feature-engineering pipeline
# ──────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Run the complete feature-engineering pipeline.

    1. Temporal features
    2. LMP lag features
    3. Temperature / weather lag features
    4. Rolling statistics on LMP
    5. Weather transformations
    6. Spike labelling
    7. Drop rows with NaN from lagging / rolling

    Parameters
    ----------
    df : DataFrame
        Raw merged LMP + weather data, datetime-indexed.
    cfg : FeatureConfig, optional
        If None, uses default ``FeatureConfig()``.

    Returns
    -------
    DataFrame ready for model training (no NaNs, spike label present).
    """
    if cfg is None:
        cfg = FeatureConfig()

    out = df.copy()

    # 1. Calendar
    out = add_temporal_features(out)

    # 2. LMP lags
    out = add_lag_features(out, "lmp", cfg.lmp_lags)

    # 3. Weather lags
    if "temperature_c" in out.columns:
        out = add_lag_features(out, "temperature_c", cfg.temp_lags)

    # 4. Rolling stats on LMP
    out = add_rolling_features(out, "lmp", cfg.rolling_windows)

    # 5. Weather transformations
    out = add_weather_features(out)

    # 6. Spike label
    out = label_spikes(
        out,
        window=cfg.spike_rolling_window,
        n_std=cfg.spike_std_multiplier,
    )

    # 7. Drop NaN rows created by lags / rolling
    out = out.dropna()

    return out


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of columns that are valid model inputs.

    Excludes target columns (``lmp``, ``spike``, ``spike_threshold``),
    raw component prices, and weather columns that were only used
    to derive features.
    """
    exclude = {
        "lmp", "spike", "spike_threshold",
        "energy", "congestion", "loss",
    }
    return [c for c in df.columns if c not in exclude]
