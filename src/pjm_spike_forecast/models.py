"""
Forecasting models for PJM spike detection and price prediction.

Three models are provided:

1. **HourlyBaseline** – predicts the historical average LMP for each
   hour-of-week.  This is the "naïve but reasonable" benchmark.
2. **SpikeClassifier** – LightGBM binary classifier that predicts
   whether the next hour will be a spike.
3. **PriceRegressor** – LightGBM regressor that predicts the LMP level.

All models follow a consistent ``fit`` / ``predict`` interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator

from pjm_spike_forecast.config import ModelConfig


# ──────────────────────────────────────────────────────────────────
# 1.  Hourly-average baseline
# ──────────────────────────────────────────────────────────────────

class HourlyBaseline:
    """Predict LMP as the training-set average for the same (day_of_week, hour).

    Also provides a deterministic spike prediction: spike = 1 whenever
    the predicted mean exceeds a given threshold.
    """

    def __init__(self) -> None:
        self._lookup: Optional[pd.Series] = None
        self._global_mean: float = 0.0
        self._spike_threshold: float = 0.0

    def fit(
        self,
        df: pd.DataFrame,
        lmp_col: str = "lmp",
        spike_col: str = "spike",
    ) -> "HourlyBaseline":
        """Compute per-(day_of_week, hour) average LMP from training data."""
        self._global_mean = df[lmp_col].mean()
        self._spike_threshold = df[lmp_col].mean() + 2 * df[lmp_col].std()
        grouped = df.groupby(["day_of_week", "hour"])[lmp_col].mean()
        self._lookup = grouped
        return self

    def predict_price(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted LMP for each row."""
        if self._lookup is None:
            raise RuntimeError("Model has not been fitted yet.")

        keys = list(zip(df["day_of_week"], df["hour"]))
        preds = np.array([
            self._lookup.get(k, self._global_mean) for k in keys
        ])
        return preds

    def predict_spike(self, df: pd.DataFrame) -> np.ndarray:
        """Binary spike prediction: 1 if predicted price > threshold."""
        prices = self.predict_price(df)
        return (prices > self._spike_threshold).astype(int)

    def predict_spike_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Pseudo-probability for spike (hard 0/1)."""
        return self.predict_spike(df).astype(float)


# ──────────────────────────────────────────────────────────────────
# 2.  LightGBM spike classifier
# ──────────────────────────────────────────────────────────────────

class SpikeClassifier:
    """LightGBM binary classifier for spike detection.

    Handles class imbalance via ``scale_pos_weight`` computed
    automatically from the training labels.
    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        if params is None:
            params = ModelConfig().clf_params
        self.params = params.copy()
        self.model: Optional[LGBMClassifier] = None
        self.feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "SpikeClassifier":
        """Train the classifier.

        ``scale_pos_weight`` is set to ``n_neg / n_pos`` to compensate
        for the heavy class imbalance (spikes are rare).
        """
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0:
            self.params["scale_pos_weight"] = n_neg / n_pos

        self.feature_names = list(X.columns)
        self.model = LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary spike prediction."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of spike (class 1)."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        return self.model.feature_importances_


# ──────────────────────────────────────────────────────────────────
# 3.  LightGBM price regressor
# ──────────────────────────────────────────────────────────────────

class PriceRegressor:
    """LightGBM regressor for LMP level prediction."""

    def __init__(self, params: Optional[Dict] = None) -> None:
        if params is None:
            params = ModelConfig().reg_params
        self.params = params.copy()
        self.model: Optional[LGBMRegressor] = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PriceRegressor":
        self.feature_names = list(X.columns)
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        return self.model.feature_importances_


# ──────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────

def save_model(model: object, path: Path) -> None:
    """Persist any model to disk via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> object:
    """Load a model saved with ``save_model``."""
    return joblib.load(path)
