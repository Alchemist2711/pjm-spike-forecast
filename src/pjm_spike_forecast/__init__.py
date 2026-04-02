"""
PJM Spike Forecast
==================
Intraday electricity price spike detection and forecasting
for the PJM Interconnection market.

Modules
-------
- data       : Download and load PJM LMP + weather data
- features   : Feature engineering (lags, rolling stats, calendar)
- models     : Baseline, LightGBM classifier, LightGBM regressor
- evaluation : Metrics, walk-forward cross-validation
- visualization : Plotting utilities
- pipeline   : End-to-end orchestration
"""

__version__ = "1.0.0"
