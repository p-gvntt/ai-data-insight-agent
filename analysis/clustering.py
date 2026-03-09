"""
KMeans clustering with automatic optimal k via elbow method.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def discover_clusters(df, max_k: int = 6) -> dict:
    """
    Run KMeans clustering. Automatically selects best k
    using silhouette score. Returns labels and cluster summary.
    """
    numeric = df.select_dtypes(include="number").dropna()
    if numeric.shape[1] < 2 or numeric.shape[0] < 10:
        return {"error": "Not enough numeric data for clustering (need ≥2 columns, ≥10 rows)."}

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric)

    best_k, best_score, best_labels = 2, -1, None
    scores = {}

    for k in range(2, min(max_k + 1, numeric.shape[0])):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = round(float(score), 4)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    # Cluster size summary
    unique, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes = {f"cluster_{int(u)}": int(c) for u, c in zip(unique, counts)}

    return {
        "best_k": best_k,
        "best_silhouette_score": round(best_score, 4),
        "silhouette_scores_by_k": scores,
        "cluster_labels": best_labels.tolist(),
        "cluster_sizes": cluster_sizes,
    }