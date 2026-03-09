"""
Anomaly detection using Isolation Forest with summary statistics.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(df, contamination: float = 0.05) -> dict:
    """
    Detect anomalies using Isolation Forest.
    Returns labels (-1 = anomaly, 1 = normal) and a summary.
    """
    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[1] < 2 or numeric.shape[0] < 10:
        return {"error": "Not enough numeric data for anomaly detection (need ≥2 columns, ≥10 rows)."}

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric)

    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X)
    scores = model.score_samples(X)

    n_anomalies = int((labels == -1).sum())
    n_normal = int((labels == 1).sum())

    # Indices of anomalous rows
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