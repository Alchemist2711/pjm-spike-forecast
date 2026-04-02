"""
Data ingestion for US electricity market data.

Primary source : NYISO public CSV archive (http://mis.nyiso.com/public/)
    – real-time hourly zonal LBMP data for New York ISO.
    – **No API key required.**  Files are freely downloadable.

Backup source  : PJM via gridstatus (requires free API key).
Weather source : Open-Meteo free API (no key required).

The module also exposes a ``build_demo_dataset`` helper that constructs
a realistic demo set from publicly known statistical properties of
wholesale electricity prices so the pipeline can be exercised without
network access.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# 1.  Download real NYISO LMP data (NO API KEY NEEDED)
# ──────────────────────────────────────────────────────────────────

NYISO_BASE = "http://mis.nyiso.com/public/csv/realtime"

def _download_nyiso_month(year: int, month: int, zone: str) -> pd.DataFrame:
    """Download one month of NYISO real-time zonal LBMP data.

    NYISO publishes monthly ZIP archives at a predictable URL.
    Each ZIP contains a single CSV with 5-minute data for all zones.
    We filter to the requested zone and resample to hourly.
    """
    date_str = f"{year}{month:02d}01"
    url = f"{NYISO_BASE}/{date_str}realtime_zone_csv.zip"

    logger.info("  Fetching %s ...", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = sorted([n for n in zf.namelist() if n.endswith(".csv")])
        if not csv_names:
            raise ValueError(f"No CSV found in {url}")
        # Monthly ZIPs contain one CSV per day — read all of them
        day_frames = []
        for csv_name in csv_names:
            with zf.open(csv_name) as f:
                day_frames.append(pd.read_csv(f))
        df = pd.concat(day_frames, ignore_index=True)

    # Standardise columns
    # NYISO columns: Time Stamp, Name, PTID, LBMP ($/MWh),
    #                Marginal Cost Losses ($/MWHr), Marginal Cost Congestion ($/MWHr)
    df.columns = df.columns.str.strip()

    # Filter to requested zone
    if "Name" in df.columns:
        df = df[df["Name"].str.strip() == zone].copy()

    # NYISO uses "$/MWHr" (not "$/MWh") — match flexibly
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if col == "Time Stamp":
            rename[col] = "datetime_utc"
        elif "lbmp" in cl:
            rename[col] = "lmp"
        elif "loss" in cl:
            rename[col] = "loss"
        elif "congestion" in cl:
            rename[col] = "congestion"
    df = df.rename(columns=rename)

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    # NYISO timestamps are Eastern; convert to UTC
    df["datetime_utc"] = df["datetime_utc"].dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="shift_forward")
    df["datetime_utc"] = df["datetime_utc"].dt.tz_convert("UTC")

    # Drop rows where tz conversion produced NaT
    df = df.dropna(subset=["datetime_utc"])

    keep = [c for c in ["datetime_utc", "lmp", "congestion", "loss"] if c in df.columns]
    df = df[keep].copy()

    # Resample 5-min data to hourly (mean)
    df = df.set_index("datetime_utc")
    df = df.resample("h").mean().dropna()

    # Energy component = LMP - congestion - loss
    if "congestion" in df.columns and "loss" in df.columns:
        df["energy"] = df["lmp"] - df["congestion"] - df["loss"]

    return df


def download_nyiso_lmp(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    zone: str = "N.Y.C.",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download real-time hourly LBMP data from NYISO.

    NYISO (New York Independent System Operator) is one of the largest
    US wholesale electricity markets.  Data is freely available from
    their public archive — **no API key required**.

    Parameters
    ----------
    start, end : str
        ISO-format date strings (e.g. "2022-01-01").
    zone : str
        NYISO pricing zone.  Common options:
        "N.Y.C.", "LONGIL", "CAPITL", "WEST", "HUD VL", "CENTRL"
    save_path : Path, optional
        If provided, save the result as CSV.

    Returns
    -------
    pd.DataFrame
        Columns: ``datetime_utc``, ``lmp``, ``energy``,
        ``congestion``, ``loss``.
    """
    logger.info(
        "Downloading NYISO real-time LBMP for zone '%s' (%s → %s) …",
        zone, start, end,
    )

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    chunks = []
    current = start_dt.to_period("M")
    end_period = end_dt.to_period("M")

    while current <= end_period:
        try:
            chunk = _download_nyiso_month(current.year, current.month, zone)
            chunks.append(chunk)
        except Exception as exc:
            logger.warning("  Failed %s: %s", current, exc)
        current += 1

    if not chunks:
        raise RuntimeError(
            "No NYISO data was downloaded. Check your internet connection "
            "and that the zone name is correct (e.g. 'N.Y.C.')."
        )

    df = pd.concat(chunks)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Trim to requested range
    df = df.loc[start:end]

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out = df.reset_index()
        out.to_csv(save_path, index=False)
        logger.info("Saved NYISO LMP → %s  (%d rows)", save_path, len(df))

    return df


# ──────────────────────────────────────────────────────────────────
# 1b.  PJM download (requires API key — optional)
# ──────────────────────────────────────────────────────────────────

def download_pjm_lmp(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download real-time hourly LMP from PJM via *gridstatus*.

    Requires a free PJM API key.  Set the environment variable
    ``PJM_API_KEY`` before calling.  See README for instructions.
    """
    import os
    try:
        import gridstatus
    except ImportError as exc:
        raise ImportError(
            "gridstatus is required for PJM download.  "
            "pip install gridstatus"
        ) from exc

    api_key = os.environ.get("PJM_API_KEY")
    if not api_key:
        raise ValueError(
            "PJM API key required.  Get one free at "
            "https://apiportal.pjm.com/ then run:\n"
            "  export PJM_API_KEY='your-key-here'"
        )

    logger.info("Downloading PJM LMP data from %s to %s …", start, end)
    pjm = gridstatus.PJM(api_key=api_key)

    chunks = []
    current = pd.Timestamp(start, tz="US/Eastern")
    end_ts = pd.Timestamp(end, tz="US/Eastern")

    while current < end_ts:
        chunk_end = min(current + pd.Timedelta(days=90), end_ts)
        logger.info("  chunk %s → %s", current.date(), chunk_end.date())
        try:
            df_chunk = pjm.get_lmp(
                date=current.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                market="REAL_TIME_HOURLY",
                locations=["PJM RTO"],
            )
            chunks.append(df_chunk)
        except Exception as exc:
            logger.warning("Chunk failed (%s → %s): %s",
                           current.date(), chunk_end.date(), exc)
        current = chunk_end

    if not chunks:
        raise RuntimeError("No PJM data downloaded.")

    raw = pd.concat(chunks, ignore_index=True)
    col_map = {
        "Time": "datetime_utc", "Interval Start": "datetime_utc",
        "LMP": "lmp", "Energy": "energy",
        "Congestion": "congestion", "Loss": "loss",
    }
    raw = raw.rename(columns={c: col_map[c] for c in raw.columns if c in col_map})
    keep = [c for c in ["datetime_utc", "lmp", "energy", "congestion", "loss"]
            if c in raw.columns]
    df = raw[keep].copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df = df.drop_duplicates(subset=["datetime_utc"], keep="first")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Saved PJM LMP → %s  (%d rows)", save_path, len(df))
    return df


# ──────────────────────────────────────────────────────────────────
# 2.  Download weather data from Open-Meteo (free, no API key)
# ──────────────────────────────────────────────────────────────────

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

def download_weather(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    latitude: float = 40.71,
    longitude: float = -74.01,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download hourly weather from Open-Meteo historical archive.

    Parameters
    ----------
    start, end : str
        Date range (ISO format).
    latitude, longitude : float
        Station coordinates.  Default = New York City (central NYISO).
    save_path : Path, optional
        Persist result as CSV.

    Returns
    -------
    pd.DataFrame
        Columns: ``datetime_utc``, ``temperature_c``, ``relative_humidity``,
        ``wind_speed_kmh``, ``precipitation_mm``.
    """
    logger.info("Downloading weather data from Open-Meteo …")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "UTC",
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "datetime_utc": pd.to_datetime(hourly["time"], utc=True),
        "temperature_c": hourly["temperature_2m"],
        "relative_humidity": hourly["relative_humidity_2m"],
        "wind_speed_kmh": hourly["wind_speed_10m"],
        "precipitation_mm": hourly["precipitation"],
    })

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Saved weather → %s  (%d rows)", save_path, len(df))

    return df


# ──────────────────────────────────────────────────────────────────
# 3.  Load previously-downloaded data
# ──────────────────────────────────────────────────────────────────

def load_lmp(path: Path) -> pd.DataFrame:
    """Load a CSV of LMP data (from NYISO or PJM download).

    Returns a DataFrame indexed by ``datetime_utc`` (UTC, hourly).
    """
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.set_index("datetime_utc").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_weather(path: Path) -> pd.DataFrame:
    """Load a CSV of weather data produced by ``download_weather``."""
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.set_index("datetime_utc").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def merge_lmp_weather(lmp: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Inner-join LMP and weather on their datetime index."""
    merged = lmp.join(weather, how="inner")
    return merged


# ──────────────────────────────────────────────────────────────────
# 4.  Demo dataset (for --demo flag and unit tests)
# ──────────────────────────────────────────────────────────────────

def build_demo_dataset(
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a realistic demo dataset that mimics wholesale electricity LMP statistics.

    The data is constructed from **publicly known stylised facts** of
    US ISO hourly LMPs:

    * Median price ~ $25–35/MWh
    * Strong diurnal pattern (peak 14:00–18:00 ET)
    * Seasonal load-driven pattern (summer and winter peaks)
    * Heavy right tail — occasional spikes >$100/MWh
    * Temperature and load are positively correlated

    **This is NOT random noise.**  It encodes the structural features
    that the model is designed to capture, allowing end-to-end
    pipeline verification before real data is available.

    Parameters
    ----------
    n_days : int
        Number of days to generate (default 365 = one year).
    seed : int
        Numpy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Datetime-indexed (UTC, hourly) with columns:
        ``lmp``, ``energy``, ``congestion``, ``loss``,
        ``temperature_c``, ``relative_humidity``, ``wind_speed_kmh``,
        ``precipitation_mm``.
    """
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24

    idx = pd.date_range(
        "2023-01-01", periods=n_hours, freq="h", tz="UTC"
    )

    hour = np.array(idx.hour)
    month = np.array(idx.month)

    # ── Temperature (°C) – seasonal + diurnal ──
    seasonal_temp = 10 * np.sin(2 * np.pi * (np.array(idx.dayofyear) - 100) / 365)
    diurnal_temp = 4 * np.sin(2 * np.pi * (hour - 6) / 24)
    temp_noise = rng.normal(0, 2, n_hours)
    temperature = 12 + seasonal_temp + diurnal_temp + temp_noise

    # ── System load (MW) – correlated with temp extremes ──
    base_load = 80_000
    temp_load = 1500 * np.abs(temperature - 18)  # heating/cooling demand
    diurnal_load = 15_000 * np.sin(2 * np.pi * (hour - 6) / 24)
    weekend_mask = np.array(idx.weekday) >= 5
    load = base_load + temp_load + diurnal_load + rng.normal(0, 3000, n_hours)
    load[weekend_mask] *= 0.85
    load = np.clip(load, 40_000, 180_000)

    # ── Base LMP ($/MWh) – function of load ──
    load_norm = (load - load.mean()) / load.std()
    base_lmp = 30 + 8 * load_norm + 3 * load_norm**2

    # ── Spikes – occasional large deviations ──
    spike_prob = np.where(load > np.percentile(load, 90), 0.08, 0.01)
    spikes = rng.binomial(1, spike_prob)
    spike_magnitude = rng.exponential(40, n_hours) * spikes
    lmp = base_lmp + spike_magnitude + rng.normal(0, 3, n_hours)
    lmp = np.clip(lmp, 0, 500)

    # ── Price components ──
    energy = lmp * rng.uniform(0.85, 0.95, n_hours)
    congestion = lmp * rng.uniform(0.02, 0.10, n_hours)
    loss = lmp - energy - congestion

    # ── Weather extras ──
    humidity = 60 + 15 * np.sin(2 * np.pi * (month - 7) / 12) + rng.normal(0, 8, n_hours)
    humidity = np.clip(humidity, 10, 100)
    wind = 10 + 5 * rng.exponential(1, n_hours)
    precip = rng.exponential(0.3, n_hours) * rng.binomial(1, 0.15, n_hours)

    df = pd.DataFrame({
        "lmp": np.round(lmp, 2),
        "energy": np.round(energy, 2),
        "congestion": np.round(congestion, 2),
        "loss": np.round(loss, 2),
        "temperature_c": np.round(temperature, 1),
        "relative_humidity": np.round(humidity, 1),
        "wind_speed_kmh": np.round(wind, 1),
        "precipitation_mm": np.round(precip, 2),
    }, index=idx)

    df.index.name = "datetime_utc"
    return df
