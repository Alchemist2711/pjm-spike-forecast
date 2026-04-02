"""
End-to-end pipeline for PJM spike forecasting.

Usage
-----
    # Download real PJM data (requires internet)
    python -m pjm_spike_forecast.pipeline --download-only

    # Run full pipeline on downloaded data
    python -m pjm_spike_forecast.pipeline --data-dir data/raw --output-dir results

    # Quick demo with built-in dataset (no internet needed)
    python -m pjm_spike_forecast.pipeline --demo --output-dir results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pjm_spike_forecast.config import (
    DATA_DIR,
    DEFAULT_END,
    DEFAULT_START,
    RESULTS_DIR,
    BacktestConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
)
from pjm_spike_forecast.data import (
    build_demo_dataset,
    download_nyiso_lmp,
    download_weather,
    load_lmp,
    load_weather,
    merge_lmp_weather,
)
from pjm_spike_forecast.evaluation import (
    classification_metrics,
    compute_shap_importance,
    regression_metrics,
    run_walk_forward_cv,
    summarise_cv_results,
)
from pjm_spike_forecast.features import (
    build_feature_matrix,
    get_feature_columns,
)
from pjm_spike_forecast.models import (
    HourlyBaseline,
    PriceRegressor,
    SpikeClassifier,
    save_model,
)
from pjm_spike_forecast.visualization import (
    plot_actual_vs_predicted,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_cv_summary,
    plot_feature_importance,
    plot_lmp_timeseries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────────────────────────

def step_download(cfg: PipelineConfig) -> None:
    """Step 0: Download raw data from NYISO + Open-Meteo."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    lmp_path = cfg.data_dir / "nyiso_lmp.csv"
    weather_path = cfg.data_dir / "weather.csv"

    download_nyiso_lmp(
        start=DEFAULT_START,
        end=DEFAULT_END,
        zone="N.Y.C.",
        save_path=lmp_path,
    )
    download_weather(
        start=DEFAULT_START,
        end=DEFAULT_END,
        latitude=40.71,   # NYC
        longitude=-74.01,
        save_path=weather_path,
    )
    logger.info("✓ Data download complete.")


def step_load(cfg: PipelineConfig, demo: bool = False) -> pd.DataFrame:
    """Step 1: Load raw data → merged DataFrame."""
    if demo:
        logger.info("Using demo dataset (1 year, realistic electricity market statistics)")
        df = build_demo_dataset(n_days=365, seed=42)
        return df

    # Look for NYISO first, then PJM
    lmp_path = cfg.data_dir / "nyiso_lmp.csv"
    if not lmp_path.exists():
        lmp_path = cfg.data_dir / "pjm_lmp.csv"
    weather_path = cfg.data_dir / "weather.csv"

    if not lmp_path.exists():
        raise FileNotFoundError(
            f"LMP data not found in {cfg.data_dir}.  "
            "Run with --download-only first, or use --demo."
        )

    lmp = load_lmp(lmp_path)
    logger.info("Loaded LMP data: %d rows, %s → %s",
                len(lmp), lmp.index.min(), lmp.index.max())

    if weather_path.exists():
        weather = load_weather(weather_path)
        logger.info("Loaded weather data: %d rows", len(weather))
        df = merge_lmp_weather(lmp, weather)
        logger.info("Merged dataset: %d rows", len(df))
    else:
        logger.warning("Weather data not found; proceeding with LMP only.")
        df = lmp

    return df


def step_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Step 2: Feature engineering."""
    feat_df = build_feature_matrix(df, cfg.features)
    logger.info(
        "Feature matrix: %d rows × %d columns  (%.1f%% spikes)",
        len(feat_df),
        len(feat_df.columns),
        feat_df["spike"].mean() * 100,
    )
    return feat_df


def step_evaluate(
    feat_df: pd.DataFrame,
    feature_cols: list,
    cfg: PipelineConfig,
) -> dict:
    """Step 3: Walk-forward CV for both baseline and LightGBM."""
    results = {}

    # Baseline
    logger.info("── Baseline walk-forward CV ──")
    bl_results = run_walk_forward_cv(
        feat_df, feature_cols,
        cfg=cfg.backtest,
        model_type="baseline",
    )
    results["baseline"] = bl_results

    # LightGBM
    logger.info("── LightGBM walk-forward CV ──")
    lgbm_results = run_walk_forward_cv(
        feat_df, feature_cols,
        cfg=cfg.backtest,
        model_type="lgbm",
    )
    results["lgbm"] = lgbm_results

    return results


def step_final_model(
    feat_df: pd.DataFrame,
    feature_cols: list,
    cfg: PipelineConfig,
) -> dict:
    """Step 4: Train final models on all data, compute SHAP."""
    # 80/20 temporal split for final evaluation
    split_idx = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_spike_train = train_df["spike"]
    y_spike_test = test_df["spike"]
    y_price_train = train_df["lmp"]
    y_price_test = test_df["lmp"]

    # Train
    clf = SpikeClassifier(cfg.model.clf_params)
    clf.fit(X_train, y_spike_train)

    reg = PriceRegressor(cfg.model.reg_params)
    reg.fit(X_train, y_price_train)

    baseline = HourlyBaseline()
    baseline.fit(train_df)

    # Predict
    spike_pred = clf.predict(X_test)
    spike_proba = clf.predict_proba(X_test)
    price_pred = reg.predict(X_test)
    baseline_price = baseline.predict_price(test_df)

    # Metrics
    clf_metrics = classification_metrics(y_spike_test.values, spike_pred, spike_proba)
    reg_metrics = regression_metrics(y_price_test.values, price_pred)
    bl_reg_metrics = regression_metrics(y_price_test.values, baseline_price)

    logger.info("── Final model (80/20 split) ──")
    logger.info("Spike  — Precision=%.3f  Recall=%.3f  F1=%.3f  AUC=%.3f",
                clf_metrics["precision"], clf_metrics["recall"],
                clf_metrics["f1"], clf_metrics.get("auc_roc", 0))
    logger.info("Price  — RMSE=%.2f  MAE=%.2f  R²=%.3f",
                reg_metrics["rmse"], reg_metrics["mae"], reg_metrics["r2"])
    logger.info("Baseline — RMSE=%.2f  MAE=%.2f",
                bl_reg_metrics["rmse"], bl_reg_metrics["mae"])

    # SHAP importance
    try:
        shap_df = compute_shap_importance(clf, X_test, max_samples=500)
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": [0] * len(feature_cols)})

    return {
        "clf": clf,
        "reg": reg,
        "baseline": baseline,
        "test_df": test_df,
        "X_test": X_test,
        "spike_pred": spike_pred,
        "spike_proba": spike_proba,
        "price_pred": price_pred,
        "baseline_price": baseline_price,
        "clf_metrics": clf_metrics,
        "reg_metrics": reg_metrics,
        "bl_reg_metrics": bl_reg_metrics,
        "shap_df": shap_df,
    }


def step_save_results(
    cv_results: dict,
    final: dict,
    feat_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> None:
    """Step 5: Save metrics, plots, and models."""
    out = cfg.results_dir
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV summaries ──
    for name, res_list in cv_results.items():
        summary = summarise_cv_results(res_list)
        summary.to_csv(out / f"cv_{name}.csv", index=False)
        logger.info("Saved CV summary → %s", out / f"cv_{name}.csv")

    final["shap_df"].to_csv(out / "shap_importance.csv", index=False)

    # ── Metrics JSON-like text ──
    with open(out / "final_metrics.txt", "w") as f:
        f.write("=== Spike Classification (LightGBM) ===\n")
        for k, v in final["clf_metrics"].items():
            f.write(f"  {k:20s}: {v}\n")
        f.write("\n=== Price Regression (LightGBM) ===\n")
        for k, v in final["reg_metrics"].items():
            f.write(f"  {k:20s}: {v}\n")
        f.write("\n=== Price Regression (Baseline) ===\n")
        for k, v in final["bl_reg_metrics"].items():
            f.write(f"  {k:20s}: {v}\n")

    # ── Plots ──
    test_df = final["test_df"]

    fig = plot_lmp_timeseries(feat_df)
    fig.savefig(plots_dir / "lmp_timeseries.png", dpi=150)
    plt.close(fig)

    fig = plot_confusion_matrix(
        test_df["spike"].values, final["spike_pred"]
    )
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    if final["shap_df"]["mean_abs_shap"].sum() > 0:
        fig = plot_feature_importance(final["shap_df"])
        fig.savefig(plots_dir / "shap_importance.png", dpi=150)
        plt.close(fig)

    fig = plot_actual_vs_predicted(
        test_df["lmp"].values, final["price_pred"]
    )
    fig.savefig(plots_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    if "lgbm" in cv_results and cv_results["lgbm"]:
        fig = plot_cv_summary(cv_results["lgbm"])
        fig.savefig(plots_dir / "cv_summary_lgbm.png", dpi=150)
        plt.close(fig)

    try:
        fig = plot_calibration_curve(
            test_df["spike"].values, final["spike_proba"]
        )
        fig.savefig(plots_dir / "calibration_curve.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # ── Models ──
    save_model(final["clf"], out / "spike_classifier.joblib")
    save_model(final["reg"], out / "price_regressor.joblib")

    logger.info("✓ All results saved to %s", out)


# ──────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig, demo: bool = False) -> None:
    """Execute the full pipeline: load → features → evaluate → save."""
    logger.info("═══ PJM Spike Forecast Pipeline ═══")

    # Load
    raw_df = step_load(cfg, demo=demo)

    # Features
    feat_df = step_features(raw_df, cfg)
    feature_cols = get_feature_columns(feat_df)
    logger.info("Using %d features", len(feature_cols))

    # Walk-forward CV
    cv_results = step_evaluate(feat_df, feature_cols, cfg)

    # Final model + SHAP
    final = step_final_model(feat_df, feature_cols, cfg)

    # Save everything
    step_save_results(cv_results, final, feat_df, cfg)

    logger.info("═══ Pipeline complete ═══")


# ──────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PJM Electricity Price Spike Forecasting Pipeline"
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download PJM + weather data and exit."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with built-in demo dataset (no download needed)."
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help="Directory containing raw CSV files."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(RESULTS_DIR),
        help="Directory for results, plots, and models."
    )
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of walk-forward CV folds."
    )
    parser.add_argument(
        "--train-months", type=int, default=6,
        help="Training window per fold (months)."
    )
    parser.add_argument(
        "--test-months", type=int, default=1,
        help="Test window per fold (months)."
    )

    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=Path(args.data_dir),
        results_dir=Path(args.output_dir),
        backtest=BacktestConfig(
            n_splits=args.n_splits,
            train_months=args.train_months,
            test_months=args.test_months,
        ),
    )

    if args.download_only:
        step_download(cfg)
        sys.exit(0)

    run_pipeline(cfg, demo=args.demo)


if __name__ == "__main__":
    main()
