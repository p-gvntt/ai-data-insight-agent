"""
Runs clustering and anomaly detection, returns combined patterns dict.
Accepts optional eda dict for future context-aware decisions.
"""
import pandas as pd
from analysis.clustering import discover_clusters
from analysis.anomalies import detect_anomalies


def pattern_agent(df: pd.DataFrame, eda: dict = None) -> dict:
    clusters  = discover_clusters(df)
    anomalies = detect_anomalies(df)
    return {
        "clusters":  clusters,
        "anomalies": anomalies,
    }