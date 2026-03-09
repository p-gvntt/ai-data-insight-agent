"""
Tests for analysis/clustering.py
Run with: pytest tests/test_clustering.py -v
"""

import pytest
import pandas as pd
import numpy as np

from analysis.clustering import discover_clusters


# Fixtures
@pytest.fixture
def clusterable_df():
    """Three well-separated clusters, 30 rows."""
    rng = np.random.default_rng(0)
    group_a = pd.DataFrame({"x": rng.normal(0, 0.5, 10),   "y": rng.normal(0, 0.5, 10)})
    group_b = pd.DataFrame({"x": rng.normal(10, 0.5, 10),  "y": rng.normal(10, 0.5, 10)})
    group_c = pd.DataFrame({"x": rng.normal(20, 0.5, 10),  "y": rng.normal(20, 0.5, 10)})
    return pd.concat([group_a, group_b, group_c], ignore_index=True)

@pytest.fixture
def small_df():
    """Only 5 rows — below the minimum threshold."""
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})

@pytest.fixture
def single_col_df():
    """Only one numeric column — below the minimum threshold."""
    return pd.DataFrame({"x": list(range(20))})

@pytest.fixture
def df_with_missing():
    """Numeric df with some NaN rows — should still work after dropna."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(5, 1, 20)})
    df.loc[[0, 1, 2], "x"] = np.nan
    return df


# discover_clusters

class TestDiscoverClusters:
    def test_returns_required_keys(self, clusterable_df):
        r = discover_clusters(clusterable_df)
        for key in ["best_k", "best_silhouette_score", "silhouette_scores_by_k",
                    "cluster_labels", "cluster_sizes"]:
            assert key in r

    def test_best_k_is_integer(self, clusterable_df):
        r = discover_clusters(clusterable_df)
        assert isinstance(r["best_k"], int)

    def test_best_k_within_range(self, clusterable_df):
        r = discover_clusters(clusterable_df, max_k=5)
        assert 2 <= r["best_k"] <= 5

    def test_finds_three_clusters(self, clusterable_df):
        # Well-separated data should resolve to k=3
        r = discover_clusters(clusterable_df, max_k=5)
        assert r["best_k"] == 3

    def test_silhouette_score_between_minus1_and_1(self, clusterable_df):
        r = discover_clusters(clusterable_df)
        assert -1 <= r["best_silhouette_score"] <= 1

    def test_silhouette_scores_by_k_has_entries(self, clusterable_df):
        r = discover_clusters(clusterable_df, max_k=4)
        assert len(r["silhouette_scores_by_k"]) >= 1

    def test_cluster_labels_length_matches_rows(self, clusterable_df):
        r = discover_clusters(clusterable_df)
        assert len(r["cluster_labels"]) == len(clusterable_df)

    def test_cluster_sizes_sum_matches_rows(self, clusterable_df):
        r = discover_clusters(clusterable_df)
        assert sum(r["cluster_sizes"].values()) == len(clusterable_df)

    def test_cluster_sizes_keys_formatted(self, clusterable_df):
        r = discover_clusters(clusterable_df)
        for key in r["cluster_sizes"]:
            assert key.startswith("cluster_")

    def test_too_few_rows_returns_error(self, small_df):
        r = discover_clusters(small_df)
        assert "error" in r

    def test_single_column_returns_error(self, single_col_df):
        r = discover_clusters(single_col_df)
        assert "error" in r

    def test_handles_missing_values(self, df_with_missing):
        r = discover_clusters(df_with_missing)
        assert "error" not in r
        assert "best_k" in r

    def test_label_values_are_integers(self, clusterable_df):
        r = discover_clusters(clusterable_df)
        assert all(isinstance(v, int) for v in r["cluster_labels"])