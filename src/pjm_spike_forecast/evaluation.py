"""
Evaluation utilities: metrics, walk-forward cross-validation, and
SHAP-based feature importance.

Walk-forward CV is the gold standard for time-series model evaluation:
the training window slides forward in time and the test window always
comes strictly *after* the training window, preventing any look-ahead
leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pjm_spike_forecast.config import BacktestConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Classification metrics (spike detection)
# ──────────────────────────────────────────────────────────────────

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute a full set of binary classification metrics.

    Parameters
    ----------
    y_true  : ground-truth spike labels (0/1).
    y_pred  : predicted spike labels (0/1).
    y_proba : predicted probability of spike (for AUC metrics).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc_roc,
    avg_precision, true_pos, false_pos, true_neg, false_neg.
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0.0),
        "recall": recall_score(y_true, y_pred, zero_division=0.0),
        "f1": f1_score(y_true, y_pred, zero_division=0.0),
        "true_pos": int(tp),
        "false_pos": int(fp),
        "true_neg": int(tn),
        "false_neg": int(fn),
    }

    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        metrics["avg_precision"] = average_precision_score(y_true, y_proba)

    return metrics


# ──────────────────────────────────────────────────────────────────
# Regression metrics (price forecasting)
# ──────────────────────────────────────────────────────────────────

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics for price-level prediction.

    Returns dict with: rmse, mae, mape, r2.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # MAPE – guard against zero division
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# ──────────────────────────────────────────────────────────────────
# Walk-forward cross-validation
# ──────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Container for a single CV fold's output."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    clf_metrics: Dict[str, float]
    reg_metrics: Dict[str, float]


def generate_temporal_splits(
    df: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate (train, test) DataFrame pairs using walk-forward logic.

    Each fold's training window is ``cfg.train_months`` long and the
    test window is ``cfg.test_months`` long.  A gap of
    ``cfg.gap_hours`` is inserted between train end and test start
    to prevent any information leakage from recent lags.

    Parameters
    ----------
    df : DataFrame
        Feature matrix, datetime-indexed and sorted ascending.
    cfg : BacktestConfig, optional
        Defaults to ``BacktestConfig()``.

    Returns
    -------
    List of (train_df, test_df) tuples.
    """
    if cfg is None:
        cfg = BacktestConfig()

    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    start = df.index.min()
    end = df.index.max()

    # First fold starts at the earliest available date
    fold_start = start

    for i in range(cfg.n_splits):
        train_start = fold_start
        train_end = train_start + pd.DateOffset(months=cfg.train_months)
        test_start = train_end + pd.Timedelta(hours=cfg.gap_hours)
        test_end = test_start + pd.DateOffset(months=cfg.test_months)

        if test_end > end:
            break  # not enough data for this fold

        train_df = df.loc[train_start:train_end]
        test_df = df.loc[test_start:test_end]

        if len(train_df) < 100 or len(test_df) < 24:
            fold_start = fold_start + pd.DateOffset(months=cfg.test_months)
            continue

        splits.append((train_df, test_df))
        logger.info(
            "Fold %d: train %s→%s (%d rows), test %s→%s (%d rows)",
            i, train_start.date(), train_end.date(), len(train_df),
            test_start.date(), test_end.date(), len(test_df),
        )

        # Slide forward
        fold_start = fold_start + pd.DateOffset(months=cfg.test_months)

    return splits


def run_walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_spike: str = "spike",
    target_price: str = "lmp",
    cfg: Optional[BacktestConfig] = None,
    model_type: str = "lgbm",
) -> List[FoldResult]:
    """Execute walk-forward CV and return per-fold metrics.

    Parameters
    ----------
    df : DataFrame
        Full feature matrix (datetime-indexed).
    feature_cols : list[str]
        Columns to use as model inputs.
    target_spike, target_price : str
        Target column names.
    cfg : BacktestConfig
    model_type : str
        ``"lgbm"`` (LightGBM) or ``"baseline"`` (HourlyBaseline).

    Returns
    -------
    list[FoldResult] – one entry per fold.
    """
    from pjm_spike_forecast.models import (
        HourlyBaseline,
        PriceRegressor,
        SpikeClassifier,
    )

    splits = generate_temporal_splits(df, cfg)
    results: List[FoldResult] = []

    for i, (train_df, test_df) in enumerate(splits):
        X_train = train_df[feature_cols]
        y_spike_train = train_df[target_spike]
        y_price_train = train_df[target_price]

        X_test = test_df[feature_cols]
        y_spike_test = test_df[target_spike].values
        y_price_test = test_df[target_price].values

        if model_type == "baseline":
            bl = HourlyBaseline().fit(train_df)
            spike_pred = bl.predict_spike(test_df)
            spike_proba = bl.predict_spike_proba(test_df)
            price_pred = bl.predict_price(test_df)
        else:
            # Spike classifier
            clf = SpikeClassifier()
            clf.fit(X_train, y_spike_train)
            spike_pred = clf.predict(X_test)
            spike_proba = clf.predict_proba(X_test)

            # Price regressor
            reg = PriceRegressor()
            reg.fit(X_train, y_price_train)
            price_pred = reg.predict(X_test)

        clf_m = classification_metrics(y_spike_test, spike_pred, spike_proba)
        reg_m = regression_metrics(y_price_test, price_pred)

        results.append(FoldResult(
            fold=i,
            train_start=str(train_df.index.min()),
            train_end=str(train_df.index.max()),
            test_start=str(test_df.index.min()),
            test_end=str(test_df.index.max()),
            clf_metrics=clf_m,
            reg_metrics=reg_m,
        ))

        logger.info(
            "Fold %d — Spike F1=%.3f  AUC=%.3f | Price RMSE=%.2f  MAE=%.2f",
            i,
            clf_m["f1"],
            clf_m.get("auc_roc", 0),
            reg_m["rmse"],
            reg_m["mae"],
        )

    return results


def summarise_cv_results(results: List[FoldResult]) -> pd.DataFrame:
    """Flatten fold results into a tidy summary DataFrame."""
    rows = []
    for r in results:
        row = {"fold": r.fold, "train_start": r.train_start,
               "train_end": r.train_end, "test_start": r.test_start,
               "test_end": r.test_end}
        for k, v in r.clf_metrics.items():
            row[f"clf_{k}"] = v
        for k, v in r.reg_metrics.items():
            row[f"reg_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# SHAP feature importance
# ──────────────────────────────────────────────────────────────────

def compute_shap_importance(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Compute mean |SHAP value| for each feature.

    Parameters
    ----------
    model : a fitted SpikeClassifier or PriceRegressor.
    X : feature DataFrame.
    max_samples : subsample size for SHAP (speed).

    Returns
    -------
    DataFrame with columns ``feature`` and ``mean_abs_shap``, sorted
    descending.
    """
    import shap

    inner = model.model if hasattr(model, "model") else model
    if max_samples < len(X):
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(inner)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifier shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = np.mean(np.abs(shap_values), axis=0)
    df_imp = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return df_imp
