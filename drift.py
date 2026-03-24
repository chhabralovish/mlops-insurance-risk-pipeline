import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def calculate_psi(expected: np.ndarray, actual: np.ndarray,
                  buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change — monitor
    PSI > 0.2: Significant change — investigate
    """
    def scale_range(input_arr, min_val, max_val):
        input_arr += -(np.min(input_arr))
        input_arr /= np.max(input_arr) / (max_val - min_val)
        input_arr += min_val
        return input_arr

    breakpoints = np.arange(0, buckets + 1) / buckets * 100

    expected_percents = np.percentile(expected, breakpoints)
    expected_percents = np.unique(expected_percents)

    def get_bucket_counts(data, breaks):
        counts = np.zeros(len(breaks) - 1)
        for i in range(len(breaks) - 1):
            counts[i] = np.sum((data >= breaks[i]) & (data < breaks[i + 1]))
        counts[-1] += np.sum(data == breaks[-1])
        return counts

    expected_counts = get_bucket_counts(expected, expected_percents)
    actual_counts = get_bucket_counts(actual, expected_percents)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test for distribution drift."""
    stat, p_value = stats.ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": p_value < 0.05
    }


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                 numeric_features: list) -> pd.DataFrame:
    """
    Detect data drift between reference (training) and current data.
    Returns DataFrame with PSI and KS test results per feature.
    """
    results = []

    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ref = reference_df[feature].dropna().values
        cur = current_df[feature].dropna().values

        psi = calculate_psi(ref, cur)
        ks = ks_test(ref, cur)

        # PSI interpretation
        if psi < 0.1:
            psi_status = "No Drift"
            severity = "Low"
        elif psi < 0.2:
            psi_status = "Moderate Drift"
            severity = "Medium"
        else:
            psi_status = "Significant Drift"
            severity = "High"

        results.append({
            "feature": feature,
            "psi": psi,
            "psi_status": psi_status,
            "severity": severity,
            "ks_statistic": ks["ks_statistic"],
            "ks_p_value": ks["p_value"],
            "drift_detected": ks["drift_detected"],
            "ref_mean": round(float(np.mean(ref)), 4),
            "cur_mean": round(float(np.mean(cur)), 4),
            "ref_std": round(float(np.std(ref)), 4),
            "cur_std": round(float(np.std(cur)), 4)
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("psi", ascending=False)
    return df


def generate_drift_data(reference_df: pd.DataFrame,
                        drift_magnitude: float = 0.3) -> pd.DataFrame:
    """
    Simulate drifted data for demonstration purposes.
    Adds controlled drift to numeric features.
    """
    drifted = reference_df.copy()
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in ["high_risk", "smoker", "chronic_conditions"]:
            continue
        noise = np.random.normal(
            drift_magnitude * reference_df[col].std(),
            0.1 * reference_df[col].std(),
            len(reference_df)
        )
        drifted[col] = (reference_df[col] + noise).clip(
            reference_df[col].min(),
            reference_df[col].max() * 1.2
        )

    return drifted


def get_drift_summary(drift_df: pd.DataFrame) -> dict:
    """Summarise drift results."""
    if len(drift_df) == 0:
        return {}

    return {
        "total_features": len(drift_df),
        "high_drift": len(drift_df[drift_df["severity"] == "High"]),
        "medium_drift": len(drift_df[drift_df["severity"] == "Medium"]),
        "no_drift": len(drift_df[drift_df["severity"] == "Low"]),
        "max_psi_feature": drift_df.iloc[0]["feature"],
        "max_psi": drift_df.iloc[0]["psi"],
        "ks_drifted": drift_df["drift_detected"].sum()
    }