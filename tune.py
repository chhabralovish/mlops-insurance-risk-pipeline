import optuna
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score
from models import get_random_forest, get_xgboost, get_lightgbm
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_random_forest(X_train, y_train, X_val, y_val, n_trials: int = 20):
    """Tune Random Forest hyperparameters with Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
        model = get_random_forest(params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"RF Best AUC: {study.best_value:.4f}")
    print(f"RF Best params: {study.best_params}")
    return study.best_params, study.best_value


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 20):
    """Tune XGBoost hyperparameters with Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
        model = get_xgboost(params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"XGB Best AUC: {study.best_value:.4f}")
    print(f"XGB Best params: {study.best_params}")
    return study.best_params, study.best_value


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials: int = 20):
    """Tune LightGBM hyperparameters with Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "verbose": -1
        }
        model = get_lightgbm(params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"LGB Best AUC: {study.best_value:.4f}")
    print(f"LGB Best params: {study.best_params}")
    return study.best_params, study.best_value


TUNERS = {
    "random_forest": tune_random_forest,
    "xgboost": tune_xgboost,
    "lightgbm": tune_lightgbm
}