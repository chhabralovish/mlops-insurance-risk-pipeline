import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

np.random.seed(42)


def generate_insurance_dataset(n_samples=5000, save_path="data/insurance_risk.csv"):
    """Generate a realistic insurance risk scoring dataset."""

    n = n_samples

    # Demographics
    age = np.random.normal(45, 15, n).clip(18, 85).astype(int)
    gender = np.random.choice(["Male", "Female"], n, p=[0.52, 0.48])
    marital_status = np.random.choice(
        ["Single", "Married", "Divorced", "Widowed"], n,
        p=[0.30, 0.50, 0.15, 0.05]
    )

    # Financial
    annual_income = np.random.lognormal(11, 0.5, n).clip(15000, 500000).astype(int)
    credit_score = np.random.normal(680, 80, n).clip(300, 850).astype(int)
    num_dependents = np.random.choice([0, 1, 2, 3, 4], n, p=[0.3, 0.25, 0.25, 0.15, 0.05])
    debt_to_income = np.random.beta(2, 5, n).clip(0, 1).round(3)

    # Insurance history
    years_with_insurer = np.random.exponential(5, n).clip(0, 30).astype(int)
    previous_claims = np.random.poisson(0.8, n).clip(0, 10)
    claim_amount_history = (previous_claims * np.random.lognormal(7, 1, n)).clip(0, 500000).astype(int)

    # Policy details
    policy_type = np.random.choice(
        ["Health", "Auto", "Life", "Home", "Travel"], n,
        p=[0.30, 0.25, 0.20, 0.20, 0.05]
    )
    coverage_amount = np.random.lognormal(12, 0.8, n).clip(10000, 2000000).astype(int)
    policy_duration_years = np.random.choice(range(1, 31), n)
    num_policies = np.random.choice([1, 2, 3, 4, 5], n, p=[0.4, 0.3, 0.2, 0.07, 0.03])

    # Health / lifestyle (for health/life policies)
    bmi = np.random.normal(26, 5, n).clip(15, 50).round(1)
    smoker = np.random.choice([0, 1], n, p=[0.75, 0.25])
    exercise_frequency = np.random.choice(
        ["Never", "Rarely", "Sometimes", "Often", "Daily"], n,
        p=[0.15, 0.20, 0.30, 0.25, 0.10]
    )
    chronic_conditions = np.random.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.15, 0.05])

    # Compute risk score (target variable - regression)
    risk_score = (
        0.15 * (age / 85) +
        0.20 * (1 - credit_score / 850) +
        0.15 * (previous_claims / 10) +
        0.10 * debt_to_income +
        0.10 * smoker +
        0.08 * (bmi / 50) +
        0.08 * (chronic_conditions / 3) +
        0.07 * (claim_amount_history / 500000) +
        0.07 * (num_dependents / 4)
    )

    # Add noise
    risk_score = (risk_score + np.random.normal(0, 0.05, n)).clip(0, 1).round(4)

    # Create high risk binary flag — use 0.28 threshold for ~25% high risk rate
    high_risk = (risk_score > 0.28).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "num_dependents": num_dependents,
        "debt_to_income": debt_to_income,
        "years_with_insurer": years_with_insurer,
        "previous_claims": previous_claims,
        "claim_amount_history": claim_amount_history,
        "policy_type": policy_type,
        "coverage_amount": coverage_amount,
        "policy_duration_years": policy_duration_years,
        "num_policies": num_policies,
        "bmi": bmi,
        "smoker": smoker,
        "exercise_frequency": exercise_frequency,
        "chronic_conditions": chronic_conditions,
        "risk_score": risk_score,
        "high_risk": high_risk
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved: {save_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"High risk rate: {high_risk.mean():.1%}")
    print(f"Risk score range: {risk_score.min():.3f} - {risk_score.max():.3f}")
    return df


if __name__ == "__main__":
    df = generate_insurance_dataset()
    print(df.describe())
