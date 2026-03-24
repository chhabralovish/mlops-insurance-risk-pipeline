from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np


# ── Feature Config ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "annual_income", "credit_score", "num_dependents",
    "debt_to_income", "years_with_insurer", "previous_claims",
    "claim_amount_history", "coverage_amount", "policy_duration_years",
    "num_policies", "bmi", "smoker", "chronic_conditions"
]

CATEGORICAL_FEATURES = [
    "gender", "marital_status", "policy_type", "exercise_frequency"
]

TARGET = "high_risk"


def get_preprocessor():
    """Build sklearn ColumnTransformer for preprocessing."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])
    return preprocessor


def load_and_split(data_path: str, test_size: float = 0.2, val_size: float = 0.1):
    """Load data and split into train/val/test."""
    df = pd.read_csv(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_random_forest(params: dict = None) -> Pipeline:
    """Random Forest pipeline with default or custom params."""
    default_params = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)

    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", RandomForestClassifier(**default_params))
    ])


def get_xgboost(params: dict = None) -> Pipeline:
    """XGBoost pipeline with default or custom params."""
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)

    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", xgb.XGBClassifier(**default_params))
    ])


def get_lightgbm(params: dict = None) -> Pipeline:
    """LightGBM pipeline with default or custom params."""
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    if params:
        default_params.update(params)

    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("model", lgb.LGBMClassifier(**default_params))
    ])


MODEL_REGISTRY = {
    "random_forest": get_random_forest,
    "xgboost": get_xgboost,
    "lightgbm": get_lightgbm
}