"""
Plotting utilities for the PJM spike forecasting project.

All functions return a ``matplotlib.figure.Figure`` so the caller
can either ``fig.savefig(...)`` or ``plt.show()`` interactively.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pjm_spike_forecast.evaluation import FoldResult


def plot_lmp_timeseries(
    df: pd.DataFrame,
    lmp_col: str = "lmp",
    spike_col: str = "spike",
    title: str = "PJM Hourly LMP with Detected Spikes",
) -> plt.Figure:
    """Time-series plot of LMP with spike hours highlighted in red."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df[lmp_col], linewidth=0.5, color="steelblue", label="LMP ($/MWh)")

    if spike_col in df.columns:
        spikes = df[df[spike_col] == 1]
        ax.scatter(spikes.index, spikes[lmp_col], color="red", s=8,
                   zorder=5, label="Spike")

    if "spike_threshold" in df.columns:
        ax.plot(df.index, df["spike_threshold"], linewidth=0.5,
                color="orange", alpha=0.6, label="Threshold")

    ax.set_xlabel("Date")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Feature Importances (mean |SHAP|)",
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(top["feature"], top["mean_abs_shap"], color="teal")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Spike Detection – Confusion Matrix",
) -> plt.Figure:
    """Confusion matrix heatmap for spike classification."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Spike"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Spike Probability Calibration",
) -> plt.Figure:
    """Calibration (reliability) curve for the spike classifier."""
    fig, ax = plt.subplots(figsize=(6, 5))

    fraction_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    ax.plot(mean_pred, fraction_pos, "s-", color="teal", label="Classifier")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cv_summary(
    results: List[FoldResult],
    title: str = "Walk-Forward CV – Per-Fold Metrics",
) -> plt.Figure:
    """Bar chart comparing key metrics across CV folds."""
    folds = [r.fold for r in results]
    f1s = [r.clf_metrics["f1"] for r in results]
    aucs = [r.clf_metrics.get("auc_roc", 0) for r in results]
    rmses = [r.reg_metrics["rmse"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(folds, f1s, color="teal")
    axes[0].set_title("Spike F1")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylim(0, 1)

    axes[1].bar(folds, aucs, color="coral")
    axes[1].set_title("Spike AUC-ROC")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylim(0, 1)

    axes[2].bar(folds, rmses, color="steelblue")
    axes[2].set_title("Price RMSE ($/MWh)")
    axes[2].set_xlabel("Fold")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted LMP",
) -> plt.Figure:
    """Scatter plot of actual vs predicted LMP."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, color="steelblue")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(np.percentile(y_true, 99), np.percentile(y_pred, 99)),
    ]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Actual LMP ($/MWh)")
    ax.set_ylabel("Predicted LMP ($/MWh)")
    ax.set_title(title)
    fig.tight_layout()
    return fig
