"""Tests for pjm_spike_forecast.data."""

from __future__ import annotations
from pjm_spike_forecast.data import (
    build_demo_dataset,
    load_lmp,
    load_weather,
    merge_lmp_weather,
)

import io
import json
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest
import requests


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_nyiso_csv_bytes(zone: str = "N.Y.C.") -> bytes:
    """Produce minimal NYISO 5-minute CSV bytes for one day."""
    rows = []
    for h in range(24):
        for m in range(0, 60, 5):
            rows.append(
                f"01/01/2023 {h:02d}:{m:02d}:00,{zone},12345,{30 + h:.2f},0.50,1.20"
            )
    header = "Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)"
    return ("\n".join([header] + rows) + "\n").encode()


def _make_nyiso_zip(zone: str = "N.Y.C.", n_files: int = 1) -> bytes:
    """Wrap CSV bytes in a ZIP archive (simulates NYISO monthly ZIP)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(n_files):
            zf.writestr(f"20230101realtime_zone_{i}.csv", _make_nyiso_csv_bytes(zone).decode())
    buf.seek(0)
    return buf.read()


def _mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    return resp


def _open_meteo_payload(n_hours: int = 24) -> dict:
    """Minimal Open-Meteo JSON response."""
    times = pd.date_range("2022-01-01", periods=n_hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [5.0] * n_hours,
            "relative_humidity_2m": [60.0] * n_hours,
            "wind_speed_10m": [12.0] * n_hours,
            "precipitation": [0.0] * n_hours,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.  _download_nyiso_month
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadNyisoMonth:

    @patch("pjm_spike_forecast.data.requests.get")
    def test_returns_dataframe(self, mock_get):
        mock_get.return_value = _mock_response(_make_nyiso_zip())
        df = _download_nyiso_month(2023, 1, "N.Y.C.")
        assert isinstance(df, pd.DataFrame)
        assert "lmp" in df.columns

    @patch("pjm_spike_forecast.data.requests.get")
    def test_correct_url_formed(self, mock_get):
        mock_get.return_value = _mock_response(_make_nyiso_zip())
        _download_nyiso_month(2023, 6, "N.Y.C.")
        called_url = mock_get.call_args[0][0]
        assert "20230601" in called_url
        assert "realtime_zone_csv.zip" in called_url

    @patch("pjm_spike_forecast.data.requests.get")
    def test_filters_to_requested_zone(self, mock_get):
        # CSV has both NYC and LONGIL rows; we only want NYC
        header = "Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)"
        rows = [
            "01/01/2023 01:00:00,N.Y.C.,12345,30.00,0.50,1.20",
            "01/01/2023 01:00:00,LONGIL,12346,32.00,0.60,1.30",
        ]
        csv_bytes = ("\n".join([header] + rows) + "\n").encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("day.csv", csv_bytes.decode())
        buf.seek(0)
        mock_get.return_value = _mock_response(buf.read())
        df = _download_nyiso_month(2023, 1, "N.Y.C.")
        assert len(df) >= 1

    @patch("pjm_spike_forecast.data.requests.get")
    def test_raises_on_http_error(self, mock_get):
        mock_get.return_value = _mock_response(b"", status_code=404)
        with pytest.raises(requests.HTTPError):
            _download_nyiso_month(2023, 1, "N.Y.C.")

    @patch("pjm_spike_forecast.data.requests.get")
    def test_raises_on_empty_zip(self, mock_get):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass  # empty ZIP
        buf.seek(0)
        mock_get.return_value = _mock_response(buf.read())
        with pytest.raises(ValueError, match="No CSV found"):
            _download_nyiso_month(2023, 1, "N.Y.C.")

    @patch("pjm_spike_forecast.data.requests.get")
    def test_multiple_csvs_in_zip(self, mock_get):
        mock_get.return_value = _mock_response(_make_nyiso_zip(n_files=3))
        df = _download_nyiso_month(2023, 1, "N.Y.C.")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @patch("pjm_spike_forecast.data.requests.get")
    def test_resamples_to_hourly(self, mock_get):
        mock_get.return_value = _mock_response(_make_nyiso_zip())
        df = _download_nyiso_month(2023, 1, "N.Y.C.")
        # Index should be at hourly frequency (no sub-hourly entries)
        diffs = df.index.to_series().diff().dropna().dt.total_seconds()
        assert (diffs == 3600).all()

    @patch("pjm_spike_forecast.data.requests.get")
    def test_energy_column_computed(self, mock_get):
        mock_get.return_value = _mock_response(_make_nyiso_zip())
        df = _download_nyiso_month(2023, 1, "N.Y.C.")
        if "energy" in df.columns:
            assert not df["energy"].isna().all()

    @patch("pjm_spike_forecast.data.requests.get")
    def test_timezone_conversion_to_utc(self, mock_get):
        mock_get.return_value = _mock_response(_make_nyiso_zip())
        df = _download_nyiso_month(2023, 1, "N.Y.C.")
        assert df.index.tzinfo is not None
        assert str(df.index.tz) == "UTC"

    @patch("pjm_spike_forecast.data.requests.get")
    def test_index_name_is_datetime_utc(self, mock_get):
        mock_get.return_value = _mock_response(_make_nyiso_zip())
        df = _download_nyiso_month(2023, 1, "N.Y.C.")
        assert df.index.name == "datetime_utc"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  download_nyiso_lmp
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadNyisoLmp:

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_single_month_returned(self, mock_month):
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        mock_month.return_value = pd.DataFrame({"lmp": np.ones(24)}, index=idx)
        df = download_nyiso_lmp(start="2022-01-01", end="2022-01-31")
        assert isinstance(df, pd.DataFrame)
        mock_month.assert_called_once()

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_multiple_months_concatenated(self, mock_month):
        def side_effect(year, month, zone):
            idx = pd.date_range(f"{year}-{month:02d}-01", periods=24, freq="h", tz="UTC")
            return pd.DataFrame({"lmp": np.ones(24)}, index=idx)
        mock_month.side_effect = side_effect
        df = download_nyiso_lmp(start="2022-01-01", end="2022-03-31")
        assert mock_month.call_count == 3
        assert len(df) > 24

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_trims_to_requested_range(self, mock_month):
        # Returns more data than requested; should be trimmed
        idx = pd.date_range("2022-01-01", periods=24 * 60, freq="h", tz="UTC")
        mock_month.return_value = pd.DataFrame({"lmp": np.ones(24 * 60)}, index=idx)
        df = download_nyiso_lmp(start="2022-01-15", end="2022-01-20")
        assert df.index.min() >= pd.Timestamp("2022-01-15", tz="UTC")

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_deduplicates_index(self, mock_month):
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        dup_df = pd.DataFrame({"lmp": np.ones(24)}, index=idx)
        mock_month.return_value = pd.concat([dup_df, dup_df])
        df = download_nyiso_lmp(start="2022-01-01", end="2022-01-31")
        assert df.index.is_unique

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_raises_when_all_months_fail(self, mock_month):
        mock_month.side_effect = Exception("network failure")
        with pytest.raises(RuntimeError, match="No NYISO data"):
            download_nyiso_lmp(start="2022-01-01", end="2022-01-31")

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_save_path_writes_csv(self, mock_month, tmp_path):
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        mock_month.return_value = pd.DataFrame({"lmp": np.ones(24)}, index=idx)
        out = tmp_path / "lmp.csv"
        download_nyiso_lmp(start="2022-01-01", end="2022-01-31", save_path=out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert "lmp" in loaded.columns

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_save_path_creates_parent_dirs(self, mock_month, tmp_path):
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        mock_month.return_value = pd.DataFrame({"lmp": np.ones(24)}, index=idx)
        out = tmp_path / "nested" / "dir" / "lmp.csv"
        download_nyiso_lmp(start="2022-01-01", end="2022-01-31", save_path=out)
        assert out.exists()

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_partial_month_failure_skipped(self, mock_month):
        """One month fails; remaining months are still returned."""
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        good_df = pd.DataFrame({"lmp": np.ones(24)}, index=idx)

        def side_effect(year, month, zone):
            if month == 1:
                raise Exception("fail")
            return good_df
        mock_month.side_effect = side_effect
        df = download_nyiso_lmp(start="2022-01-01", end="2022-03-31")
        assert isinstance(df, pd.DataFrame)

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_sorted_index(self, mock_month):
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        mock_month.return_value = pd.DataFrame({"lmp": np.ones(24)}, index=idx)
        df = download_nyiso_lmp(start="2022-01-01", end="2022-01-31")
        assert df.index.is_monotonic_increasing

    @patch("pjm_spike_forecast.data._download_nyiso_month")
    def test_custom_zone_passed_through(self, mock_month):
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        mock_month.return_value = pd.DataFrame({"lmp": np.ones(24)}, index=idx)
        download_nyiso_lmp(start="2022-01-01", end="2022-01-31", zone="LONGIL")
        _, _, zone_arg = mock_month.call_args[0]
        assert zone_arg == "LONGIL"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  download_pjm_lmp
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadPjmLmp:

    def test_raises_without_gridstatus(self, monkeypatch):
        monkeypatch.setenv("PJM_API_KEY", "dummy_key")
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "gridstatus":
                raise ImportError("no module named gridstatus")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="gridstatus is required"):
            download_pjm_lmp()

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("PJM_API_KEY", raising=False)
        mock_gs = MagicMock()
        with patch.dict("sys.modules", {"gridstatus": mock_gs}):
            with pytest.raises(ValueError, match="PJM API key required"):
                download_pjm_lmp()

    def test_raises_when_no_data(self, monkeypatch):
        monkeypatch.setenv("PJM_API_KEY", "test_key")
        mock_gs = MagicMock()
        mock_pjm_instance = MagicMock()
        mock_pjm_instance.get_lmp.side_effect = Exception("API error")
        mock_gs.PJM.return_value = mock_pjm_instance
        with patch.dict("sys.modules", {"gridstatus": mock_gs}):
            with pytest.raises(RuntimeError, match="No PJM data"):
                download_pjm_lmp(start="2022-01-01", end="2022-04-01")

    def test_returns_dataframe_with_expected_columns(self, monkeypatch):
        monkeypatch.setenv("PJM_API_KEY", "test_key")
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        raw = pd.DataFrame({
            "Interval Start": idx,
            "LMP": np.ones(24) * 35.0,
            "Energy": np.ones(24) * 30.0,
            "Congestion": np.ones(24) * 3.0,
            "Loss": np.ones(24) * 2.0,
        })
        mock_gs = MagicMock()
        mock_pjm_instance = MagicMock()
        mock_pjm_instance.get_lmp.return_value = raw
        mock_gs.PJM.return_value = mock_pjm_instance
        with patch.dict("sys.modules", {"gridstatus": mock_gs}):
            df = download_pjm_lmp(start="2022-01-01", end="2022-04-01")
        assert "lmp" in df.columns
        assert "datetime_utc" in df.columns

    def test_save_path_writes_csv(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PJM_API_KEY", "test_key")
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        raw = pd.DataFrame({"Interval Start": idx, "LMP": np.ones(24)})
        mock_gs = MagicMock()
        mock_pjm_instance = MagicMock()
        mock_pjm_instance.get_lmp.return_value = raw
        mock_gs.PJM.return_value = mock_pjm_instance
        out = tmp_path / "pjm.csv"
        with patch.dict("sys.modules", {"gridstatus": mock_gs}):
            download_pjm_lmp(start="2022-01-01", end="2022-04-01", save_path=out)
        assert out.exists()

    def test_deduplicates_datetime(self, monkeypatch):
        monkeypatch.setenv("PJM_API_KEY", "test_key")
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        raw = pd.DataFrame({"Interval Start": list(idx) + list(idx), "LMP": np.ones(48)})
        mock_gs = MagicMock()
        mock_pjm_instance = MagicMock()
        mock_pjm_instance.get_lmp.return_value = raw
        mock_gs.PJM.return_value = mock_pjm_instance
        with patch.dict("sys.modules", {"gridstatus": mock_gs}):
            df = download_pjm_lmp(start="2022-01-01", end="2022-04-01")
        assert df["datetime_utc"].is_unique


# # ─────────────────────────────────────────────────────────────────────────────
# # 4.  download_weather
# # ─────────────────────────────────────────────────────────────────────────────

class TestDownloadWeather:

    @patch("requests.get")
    def test_returns_dataframe_with_expected_columns(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _open_meteo_payload(48),
            raise_for_status=MagicMock(),
        )
        df = download_weather(start="2022-01-01", end="2022-01-02")
        assert set(["datetime_utc", "temperature_c", "relative_humidity",
                    "wind_speed_kmh", "precipitation_mm"]).issubset(df.columns)

    @patch("requests.get")
    def test_correct_number_of_rows(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _open_meteo_payload(72),
            raise_for_status=MagicMock(),
        )
        df = download_weather(start="2022-01-01", end="2022-01-03")
        assert len(df) == 72

    @patch("requests.get")
    def test_correct_api_params_sent(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _open_meteo_payload(24),
            raise_for_status=MagicMock(),
        )
        download_weather(start="2022-06-01", end="2022-06-01", latitude=41.0, longitude=-75.0)
        call_params = mock_get.call_args[1]["params"]
        assert call_params["latitude"] == 41.0
        assert call_params["start_date"] == "2022-06-01"

    @patch("requests.get")
    def test_raises_on_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("503")
        mock_get.return_value = mock_resp
        with pytest.raises(requests.HTTPError):
            download_weather()

    @patch("requests.get")
    def test_save_path_writes_csv(self, mock_get, tmp_path):
        mock_get.return_value = MagicMock(
            json=lambda: _open_meteo_payload(24),
            raise_for_status=MagicMock(),
        )
        out = tmp_path / "weather.csv"
        download_weather(start="2022-01-01", end="2022-01-01", save_path=out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert "temperature_c" in loaded.columns

    @patch("requests.get")
    def test_datetime_utc_aware(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: _open_meteo_payload(24),
            raise_for_status=MagicMock(),
        )
        df = download_weather(start="2022-01-01", end="2022-01-01")
        assert df["datetime_utc"].dt.tz is not None


# ─────────────────────────────────────────────────────────────────────────────
# 5.  load_lmp
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadLmp:

    def test_loads_csv_correctly(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"lmp": np.random.randn(24)}, index=idx)
        df.index.name = "datetime_utc"
        p = tmp_path / "lmp.csv"
        df.reset_index().to_csv(p, index=False)
        result = load_lmp(p)
        assert result.index.name == "datetime_utc"
        assert "lmp" in result.columns

    def test_index_is_utc(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({"lmp": np.ones(10)}, index=idx)
        df.index.name = "datetime_utc"
        p = tmp_path / "lmp.csv"
        df.reset_index().to_csv(p, index=False)
        result = load_lmp(p)
        assert str(result.index.tz) == "UTC"

    def test_deduplicates_index(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({"lmp": np.ones(10)}, index=idx)
        df.index.name = "datetime_utc"
        dup = pd.concat([df, df])
        p = tmp_path / "lmp.csv"
        dup.reset_index().to_csv(p, index=False)
        result = load_lmp(p)
        assert result.index.is_unique

    def test_sorted_index(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame({"lmp": np.ones(10)}, index=idx[::-1])
        df.index.name = "datetime_utc"
        p = tmp_path / "lmp.csv"
        df.reset_index().to_csv(p, index=False)
        result = load_lmp(p)
        assert result.index.is_monotonic_increasing

    def test_preserves_numeric_columns(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({
            "lmp": [30.0, 31.0, 32.0, 28.0, 25.0],
            "energy": [25.0, 26.0, 27.0, 23.0, 20.0],
        }, index=idx)
        df.index.name = "datetime_utc"
        p = tmp_path / "lmp.csv"
        df.reset_index().to_csv(p, index=False)
        result = load_lmp(p)
        assert list(result["lmp"].values) == pytest.approx([30.0, 31.0, 32.0, 28.0, 25.0])


# ─────────────────────────────────────────────────────────────────────────────
# 6.  load_weather
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadWeather:

    def test_loads_csv_correctly(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({
            "temperature_c": np.ones(24) * 10.0,
            "relative_humidity": np.ones(24) * 60.0,
        }, index=idx)
        df.index.name = "datetime_utc"
        p = tmp_path / "weather.csv"
        df.reset_index().to_csv(p, index=False)
        result = load_weather(p)
        assert "temperature_c" in result.columns
        assert result.index.name == "datetime_utc"

    def test_index_is_utc(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_c": np.ones(5)}, index=idx)
        df.index.name = "datetime_utc"
        p = tmp_path / "weather.csv"
        df.reset_index().to_csv(p, index=False)
        result = load_weather(p)
        assert str(result.index.tz) == "UTC"

    def test_deduplicates_index(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_c": np.ones(5)}, index=idx)
        df.index.name = "datetime_utc"
        dup = pd.concat([df, df])
        p = tmp_path / "weather.csv"
        dup.reset_index().to_csv(p, index=False)
        result = load_weather(p)
        assert result.index.is_unique

    def test_sorted_index(self, tmp_path):
        idx = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_c": np.ones(5)}, index=idx[::-1])
        df.index.name = "datetime_utc"
        p = tmp_path / "weather.csv"
        df.reset_index().to_csv(p, index=False)
        result = load_weather(p)
        assert result.index.is_monotonic_increasing


# ─────────────────────────────────────────────────────────────────────────────
# 7.  merge_lmp_weather
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeLmpWeather:

    def _make_lmp(self, start="2022-01-01", n=48):
        idx = pd.date_range(start, periods=n, freq="h", tz="UTC", name="datetime_utc")
        return pd.DataFrame({"lmp": np.ones(n) * 30.0}, index=idx)

    def _make_weather(self, start="2022-01-01", n=48):
        idx = pd.date_range(start, periods=n, freq="h", tz="UTC", name="datetime_utc")
        return pd.DataFrame({"temperature_c": np.ones(n) * 10.0}, index=idx)

    def test_inner_join_aligns_on_index(self):
        lmp = self._make_lmp(n=48)
        weather = self._make_weather(n=48)
        merged = merge_lmp_weather(lmp, weather)
        assert "lmp" in merged.columns
        assert "temperature_c" in merged.columns
        assert len(merged) == 48

    def test_inner_join_drops_non_overlapping(self):
        lmp = self._make_lmp(start="2022-01-01", n=24)
        weather = self._make_weather(start="2022-01-02", n=24)  # 0 overlap
        merged = merge_lmp_weather(lmp, weather)
        assert len(merged) == 0

    def test_partial_overlap(self):
        lmp = self._make_lmp(start="2022-01-01", n=48)
        weather = self._make_weather(start="2022-01-02", n=48)  # 24-hour overlap
        merged = merge_lmp_weather(lmp, weather)
        assert len(merged) == 24

    def test_preserves_datetime_index(self):
        lmp = self._make_lmp()
        weather = self._make_weather()
        merged = merge_lmp_weather(lmp, weather)
        assert merged.index.name == "datetime_utc"
        assert str(merged.index.tz) == "UTC"


# ─────────────────────────────────────────────────────────────────────────────
# 8.  build_demo_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDemoDataset:

    def test_returns_dataframe(self):
        df = build_demo_dataset(n_days=7)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        df = build_demo_dataset(n_days=30)
        assert len(df) == 30 * 24

    def test_expected_columns_present(self):
        df = build_demo_dataset(n_days=1)
        expected = {"lmp", "energy", "congestion", "loss",
                    "temperature_c", "relative_humidity",
                    "wind_speed_kmh", "precipitation_mm"}
        assert expected.issubset(set(df.columns))

    def test_index_name_is_datetime_utc(self):
        df = build_demo_dataset(n_days=1)
        assert df.index.name == "datetime_utc"

    def test_index_is_utc_aware(self):
        df = build_demo_dataset(n_days=1)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"

    def test_index_is_hourly(self):
        df = build_demo_dataset(n_days=7)
        diffs = df.index.to_series().diff().dropna().dt.total_seconds()
        assert (diffs == 3600).all()

    def test_reproducible_with_same_seed(self):
        df1 = build_demo_dataset(n_days=10, seed=99)
        df2 = build_demo_dataset(n_days=10, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_results(self):
        df1 = build_demo_dataset(n_days=10, seed=1)
        df2 = build_demo_dataset(n_days=10, seed=2)
        assert not df1["lmp"].equals(df2["lmp"])

    def test_lmp_within_clipped_bounds(self):
        df = build_demo_dataset(n_days=365)
        assert (df["lmp"] >= 0).all()
        assert (df["lmp"] <= 500).all()

    def test_temperature_realistic_range(self):
        df = build_demo_dataset(n_days=365)
        assert df["temperature_c"].min() > -30
        assert df["temperature_c"].max() < 50

    def test_humidity_clipped_to_valid_range(self):
        df = build_demo_dataset(n_days=365)
        assert (df["relative_humidity"] >= 10).all()
        assert (df["relative_humidity"] <= 100).all()

    def test_precipitation_non_negative(self):
        df = build_demo_dataset(n_days=365)
        assert (df["precipitation_mm"] >= 0).all()

    def test_lmp_positive_median(self):
        df = build_demo_dataset(n_days=365)
        assert df["lmp"].median() > 10

    def test_energy_less_than_lmp(self):
        df = build_demo_dataset(n_days=365)
        assert (df["energy"] <= df["lmp"]).all()

    def test_n_days_one(self):
        df = build_demo_dataset(n_days=1)
        assert len(df) == 24

    def test_no_nan_values(self):
        df = build_demo_dataset(n_days=30)
        assert not df.isnull().any().any()

    def test_spike_distribution_right_skewed(self):
        df = build_demo_dataset(n_days=365, seed=0)
        skewness = df["lmp"].skew()
        assert skewness > 1.0, f"Expected right-skewed LMP; got skew={skewness:.2f}"

    def test_default_n_days_is_365(self):
        df = build_demo_dataset()
        assert len(df) == 365 * 24

    def test_starts_at_2023(self):
        df = build_demo_dataset(n_days=1)
        assert df.index[0].year == 2023
        assert df.index[0].month == 1
        assert df.index[0].day == 1

    def test_weekend_load_effect_visible_in_lmp(self):
        df = build_demo_dataset(n_days=365)
        df["weekday"] = df.index.weekday
        wd_mean = df[df["weekday"] < 5]["lmp"].mean()
        we_mean = df[df["weekday"] >= 5]["lmp"].mean()
        assert wd_mean > we_mean, "Weekday LMP should exceed weekend LMP"