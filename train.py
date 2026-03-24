import mlflow
import mlflow.sklearn
import joblib
import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from models import (
    load_and_split, MODEL_REGISTRY,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES
)
from evaluate import (
    evaluate_model, log_metrics_to_mlflow,
    plot_roc_curve, plot_confusion_matrix,
    plot_feature_importance, compare_models
)
from tune import TUNERS
from drift import detect_drift, generate_drift_data, get_drift_summary

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/insurance_risk.csv"
MLFLOW_EXPERIMENT = "insurance-risk-scoring"
MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_single_model(model_name: str, X_train, y_train, X_val, y_val,
                       tune: bool = False, n_trials: int = 20,
                       run_name: str = None):
    """Train a single model with MLflow tracking."""

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name or model_name):

        # ── Hyperparameter Tuning ─────────────────────────────────────────────
        best_params = {}
        if tune and model_name in TUNERS:
            print(f"\nTuning {model_name} with Optuna ({n_trials} trials)...")
            best_params, best_val_auc = TUNERS[model_name](
                X_train, y_train, X_val, y_val, n_trials=n_trials
            )
            mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("optuna_best_auc", best_val_auc)
        else:
            print(f"\nTraining {model_name} with default params...")

        # ── Train ─────────────────────────────────────────────────────────────
        model_fn = MODEL_REGISTRY[model_name]
        pipeline = model_fn(best_params if best_params else None)
        pipeline.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────────────────
        val_metrics = evaluate_model(pipeline, X_val, y_val)
        print(f"{model_name} — AUC: {val_metrics['roc_auc']:.4f} | "
              f"F1: {val_metrics['f1_score']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f}")

        # Log metrics
        log_metrics_to_mlflow(val_metrics)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tuned", tune)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))

        # ── Log Plots ─────────────────────────────────────────────────────────
        roc_fig = plot_roc_curve(pipeline, X_val, y_val, model_name)
        mlflow.log_figure(roc_fig, "roc_curve.png")

        cm_fig = plot_confusion_matrix(val_metrics, model_name)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        fi_fig = plot_feature_importance(pipeline, NUMERIC_FEATURES, model_name)
        if fi_fig:
            mlflow.log_figure(fi_fig, "feature_importance.png")

        # ── Log Model ─────────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=f"insurance_risk_{model_name}"
        )

        # Save locally too
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")

        return {
            "model": pipeline,
            "metrics": val_metrics,
            "run_id": run_id,
            "model_name": model_name,
            "params": best_params
        }


def train_all_models(tune: bool = False, n_trials: int = 20):
    """Train all 3 models and compare results."""

    print("=" * 60)
    print("MLOps Insurance Risk Scoring Pipeline")
    print("=" * 60)

    # ── Generate data if not exists ───────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("Generating dataset...")
        from data.generate import generate_insurance_dataset
        generate_insurance_dataset(save_path=DATA_PATH)

    # ── Load and split ────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATA_PATH)

    # ── Train all models ──────────────────────────────────────────────────────
    results = {}
    for model_name in ["random_forest", "xgboost", "lightgbm"]:
        results[model_name] = train_single_model(
            model_name, X_train, y_train, X_val, y_val,
            tune=tune, n_trials=n_trials
        )

    # ── Compare models ────────────────────────────────────────────────────────
    comparison = compare_models(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison[["model", "roc_auc", "f1_score",
                       "precision", "recall", "accuracy"]].to_string(index=False))

    # ── Best model ────────────────────────────────────────────────────────────
    best_model_name = comparison.iloc[0]["model"]
    best_result = results[best_model_name]
    print(f"\nBest Model: {best_model_name} (AUC: {comparison.iloc[0]['roc_auc']:.4f})")

    # ── Drift detection ───────────────────────────────────────────────────────
    print("\nRunning drift detection...")
    df = pd.read_csv(DATA_PATH)
    reference = df.sample(frac=0.5, random_state=42)
    drifted = generate_drift_data(reference, drift_magnitude=0.2)

    drift_df = detect_drift(reference, drifted, NUMERIC_FEATURES)
    summary = get_drift_summary(drift_df)
    print(f"Drift summary: {summary['high_drift']} high, "
          f"{summary['medium_drift']} medium, {summary['no_drift']} no drift")

    # Save comparison and drift results
    comparison.to_csv("model_comparison.csv", index=False)
    drift_df.to_csv("drift_report.csv", index=False)

    print("\nPipeline complete!")
    print(f"MLflow UI: run 'mlflow ui' and open http://localhost:5000")
    print(f"Saved models: {MODELS_DIR}/")

    return results, comparison, best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials")
    args = parser.parse_args()

    results, comparison, best = train_all_models(
        tune=args.tune,
        n_trials=args.trials
    )