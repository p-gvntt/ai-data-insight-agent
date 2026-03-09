"""
Anomaly detection using Isolation Forest with summary statistics.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    """
    Detect anomalies using Isolation Forest.
    Returns labels (-1 = anomaly, 1 = normal) aligned to the original df index,
    and a summary.
    """
    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] < 2 or numeric.shape[0] < 10:
        return {"error": "Not enough numeric data for anomaly detection (need ≥2 columns, ≥10 rows)."}

    numeric_imputed = numeric.fillna(numeric.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_imputed)

    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X)
    scores = model.score_samples(X)

    n_anomalies = int((labels == -1).sum())
    n_normal = int((labels == 1).sum())
    anomaly_indices = np.where(labels == -1)[0].tolist()

    return {
        "labels": labels.tolist(),
        "anomaly_count": n_anomalies,
        "normal_count": n_normal,
        "anomaly_pct": round(n_anomalies / len(labels) * 100, 2),
        "anomaly_indices": anomaly_indices,
        "anomaly_scores": {
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "mean": round(float(scores.mean()), 4),
        },
    }