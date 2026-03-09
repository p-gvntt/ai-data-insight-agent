"""
Tests for analysis/anomalies.py
Run with: pytest tests/test_anomalies.py -v
"""

import pytest
import pandas as pd
import numpy as np

from analysis.anomalies import detect_anomalies


# Fixtures
@pytest.fixture
def clean_df():
    """20 rows of normal data with no obvious anomalies."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 20),
        "y": rng.normal(5, 1, 20),
    })

@pytest.fixture
def df_with_anomalies():
    """Normal data with two injected extreme outliers."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": list(rng.normal(0, 1, 18)) + [100.0, -100.0],
        "y": list(rng.normal(5, 1, 18)) + [100.0, -100.0],
    })
    return df

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
    df.loc[[0, 1], "x"] = np.nan
    return df


# detect_anomalies

class TestDetectAnomalies:
    def test_returns_required_keys(self, clean_df):
        r = detect_anomalies(clean_df)
        for key in ["labels", "anomaly_count", "normal_count",
                    "anomaly_pct", "anomaly_indices", "anomaly_scores"]:
            assert key in r

    def test_labels_are_minus1_or_1(self, clean_df):
        r = detect_anomalies(clean_df)
        assert set(r["labels"]).issubset({-1, 1})

    def test_labels_length_matches_rows(self, clean_df):
        r = detect_anomalies(clean_df)
        assert len(r["labels"]) == len(clean_df)

    def test_anomaly_and_normal_sum_to_total(self, clean_df):
        r = detect_anomalies(clean_df)
        assert r["anomaly_count"] + r["normal_count"] == len(clean_df)

    def test_anomaly_pct_between_0_and_100(self, clean_df):
        r = detect_anomalies(clean_df)
        assert 0 <= r["anomaly_pct"] <= 100

    def test_detects_injected_anomalies(self, df_with_anomalies):
        r = detect_anomalies(df_with_anomalies)
        assert r["anomaly_count"] >= 1

    def test_anomaly_indices_match_labels(self, df_with_anomalies):
        r = detect_anomalies(df_with_anomalies)
        for idx in r["anomaly_indices"]:
            assert r["labels"][idx] == -1

    def test_anomaly_scores_keys(self, clean_df):
        r = detect_anomalies(clean_df)
        assert "min" in r["anomaly_scores"]
        assert "max" in r["anomaly_scores"]
        assert "mean" in r["anomaly_scores"]

    def test_anomaly_scores_min_lte_max(self, clean_df):
        r = detect_anomalies(clean_df)
        assert r["anomaly_scores"]["min"] <= r["anomaly_scores"]["max"]

    def test_custom_contamination(self, clean_df):
        r_low  = detect_anomalies(clean_df, contamination=0.05)
        r_high = detect_anomalies(clean_df, contamination=0.20)
        assert r_high["anomaly_count"] >= r_low["anomaly_count"]

    def test_too_few_rows_returns_error(self, small_df):
        r = detect_anomalies(small_df)
        assert "error" in r

    def test_single_column_returns_error(self, single_col_df):
        r = detect_anomalies(single_col_df)
        assert "error" in r

    def test_handles_missing_values(self, df_with_missing):
        r = detect_anomalies(df_with_missing)
        assert "error" not in r
        assert "anomaly_count" in r