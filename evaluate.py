import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import mlflow
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io


def evaluate_model(model, X_val, y_val, threshold: float = 0.5) -> dict:
    """Evaluate a trained model and return all metrics."""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_val, y_pred)
    # Handle case where model predicts only one class
    if cm.shape == (1, 1):
        tn, fp, fn, tp = cm[0][0], 0, 0, 0
    elif cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics = {
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_val, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_val, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_val, y_pred_proba), 4),
        "avg_precision": round(average_precision_score(y_val, y_pred_proba), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4)
    }

    return metrics


def log_metrics_to_mlflow(metrics: dict):
    """Log all metrics to the active MLflow run."""
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not (v != v):  # skip NaN
            mlflow.log_metric(k, v)


def plot_roc_curve(model, X_val, y_val, model_name: str):
    """Generate ROC curve figure."""
    y_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    auc = roc_auc_score(y_val, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2E86AB", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(metrics: dict, model_name: str):
    """Generate confusion matrix figure."""
    cm = np.array([
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]]
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Risk", "High Risk"])
    ax.set_yticklabels(["Low Risk", "High Risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15):
    """Plot feature importance for tree-based models."""
    try:
        # Get the actual model from pipeline
        clf = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]

        # Get feature names after preprocessing
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
            ["gender", "marital_status", "policy_type", "exercise_frequency"]
        )
        all_features = list(feature_names) + list(cat_features)

        importances = clf.feature_importances_
        n = min(top_n, len(importances))
        indices = np.argsort(importances)[-n:]

        fig, ax = plt.subplots(figsize=(8, 6))
        feat_labels = [all_features[i] if i < len(all_features) else f"feat_{i}"
                       for i in indices]
        ax.barh(range(n), importances[indices], color="#2E86AB")
        ax.set_yticks(range(n))
        ax.set_yticklabels(feat_labels)
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {model_name}")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        return fig
    except Exception:
        return None


def compare_models(results: dict) -> pd.DataFrame:
    """Build a comparison DataFrame of all models."""
    rows = []
    for model_name, result in results.items():
        row = {"model": model_name}
        row.update(result.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)