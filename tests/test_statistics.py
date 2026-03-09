"""
Tests for analysis/statistics.py
Run with: pytest tests/test_statistics.py -v
"""

import pytest
import pandas as pd
import numpy as np

from analysis.statistics import (
    run_ttest,
    run_mannwhitney,
    run_anova,
    run_chi_squared,
    run_all_tests,
)


# Fixtures
@pytest.fixture
def identical_groups():
    s = pd.Series([10.0, 11.0, 10.5, 10.2, 10.8])
    return s, s.copy()

@pytest.fixture
def different_groups():
    return pd.Series([1.0, 1.1, 1.2, 1.0, 1.1]), pd.Series([100.0, 101.0, 99.0, 100.5, 100.2])

@pytest.fixture
def group_with_missing():
    return pd.Series([1.0, np.nan, 3.0, 4.0, 5.0]), pd.Series([10.0, 11.0, np.nan, 13.0, 14.0])

@pytest.fixture
def three_groups():
    return (
        pd.Series([1.0, 1.1, 1.2]),
        pd.Series([5.0, 5.1, 5.2]),
        pd.Series([10.0, 10.1, 10.2]),
    )

@pytest.fixture
def cat_df():
    return pd.DataFrame({
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "bought": ["yes", "no", "yes", "yes", "no", "no", "yes", "no"],
    })

@pytest.fixture
def full_df():
    """
    Realistic DataFrame: numeric columns + binary + multi-group categorical.
    t-test and Mann-Whitney should split avg_order_value by churned (binary).
    ANOVA should split avg_order_value by membership (4 groups).
    """
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "customer_id":   range(1, n + 1),          # ID-like — should be skipped
        "age":           rng.integers(18, 70, n).astype(float),
        "avg_order_value": rng.normal(120, 40, n),
        "num_purchases": rng.integers(0, 50, n).astype(float),
        "churned":       rng.choice(["Yes", "No"], n),           # binary
        "membership":    rng.choice(["Bronze", "Silver", "Gold", "Platinum"], n),
        "gender":        rng.choice(["Male", "Female"], n),
    })

@pytest.fixture
def no_cat_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(0, 1, 20), "b": rng.normal(5, 1, 20)})

@pytest.fixture
def single_numeric_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})


# run_ttest
class TestRunTtest:
    def test_required_keys(self, different_groups):
        r = run_ttest(*different_groups)
        for key in ["test", "statistic", "p_value", "significant"]:
            assert key in r

    def test_test_name(self, different_groups):
        assert run_ttest(*different_groups)["test"] == "independent_ttest"

    def test_significant_when_different(self, different_groups):
        assert run_ttest(*different_groups)["significant"] is True

    def test_not_significant_when_identical(self, identical_groups):
        assert run_ttest(*identical_groups)["significant"] is False

    def test_p_value_in_range(self, different_groups):
        r = run_ttest(*different_groups)
        assert 0.0 <= r["p_value"] <= 1.0

    def test_handles_missing(self, group_with_missing):
        r = run_ttest(*group_with_missing)
        assert "p_value" in r


# run_mannwhitney
class TestRunMannWhitney:
    def test_required_keys(self, different_groups):
        r = run_mannwhitney(*different_groups)
        for key in ["test", "statistic", "p_value", "significant"]:
            assert key in r

    def test_test_name(self, different_groups):
        assert run_mannwhitney(*different_groups)["test"] == "mann_whitney_u"

    def test_significant_when_different(self, different_groups):
        assert run_mannwhitney(*different_groups)["significant"] is True

    def test_p_value_in_range(self, different_groups):
        r = run_mannwhitney(*different_groups)
        assert 0.0 <= r["p_value"] <= 1.0

    def test_handles_missing(self, group_with_missing):
        r = run_mannwhitney(*group_with_missing)
        assert "p_value" in r


# run_anova
class TestRunAnova:
    def test_required_keys(self, three_groups):
        r = run_anova(*three_groups)
        for key in ["test", "statistic", "p_value", "significant", "n_groups"]:
            assert key in r

    def test_test_name(self, three_groups):
        assert run_anova(*three_groups)["test"] == "one_way_anova"

    def test_n_groups(self, three_groups):
        assert run_anova(*three_groups)["n_groups"] == 3

    def test_significant_with_distinct_groups(self, three_groups):
        assert run_anova(*three_groups)["significant"] is True

    def test_not_significant_with_same_groups(self):
        s = pd.Series([1.0, 1.1, 1.2, 1.0])
        assert run_anova(s, s.copy(), s.copy())["significant"] is False

    def test_p_value_in_range(self, three_groups):
        r = run_anova(*three_groups)
        assert 0.0 <= r["p_value"] <= 1.0


# run_chi_squared
class TestRunChiSquared:
    def test_required_keys(self, cat_df):
        r = run_chi_squared(cat_df, "gender", "bought")
        for key in ["test", "statistic", "p_value", "degrees_of_freedom", "significant"]:
            assert key in r

    def test_test_name(self, cat_df):
        assert run_chi_squared(cat_df, "gender", "bought")["test"] == "chi_squared"

    def test_dof_positive(self, cat_df):
        assert run_chi_squared(cat_df, "gender", "bought")["degrees_of_freedom"] >= 1

    def test_p_value_in_range(self, cat_df):
        r = run_chi_squared(cat_df, "gender", "bought")
        assert 0.0 <= r["p_value"] <= 1.0

    def test_perfectly_dependent(self):
        df = pd.DataFrame({
            "a": ["x"] * 20 + ["y"] * 20,
            "b": ["p"] * 20 + ["q"] * 20,
        })
        assert run_chi_squared(df, "a", "b")["significant"] is True

    def test_independent_columns(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.choice(["x", "y"], 200),
            "b": rng.choice(["p", "q"], 200),
        })
        r = run_chi_squared(df, "a", "b")
        assert 0.0 <= r["p_value"] <= 1.0


# run_all_tests
class TestRunAllTests:
    def test_ttest_present(self, full_df):
        r = run_all_tests(full_df)
        assert "ttest" in r

    def test_mannwhitney_present(self, full_df):
        r = run_all_tests(full_df)
        assert "mann_whitney" in r

    def test_anova_present(self, full_df):
        # full_df has "membership" with 4 groups so ANOVA should run
        r = run_all_tests(full_df)
        assert "anova" in r

    def test_anova_absent_when_only_binary_categoricals(self):
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            "value": rng.normal(10, 2, n),
            "sex":   rng.choice(["male", "female"], n),  # only binary categorical
        })
        r = run_all_tests(df)
        assert "anova" not in r

    def test_chi_squared_present(self, full_df):
        r = run_all_tests(full_df)
        assert "chi_squared" in r

    def test_ttest_has_group_context(self, full_df):
        r = run_all_tests(full_df)
        for key in ["numeric_column", "grouped_by", "group_a", "group_a_mean", "group_b", "group_b_mean"]:
            assert key in r["ttest"], f"Missing key: {key}"

    def test_mannwhitney_has_group_context(self, full_df):
        r = run_all_tests(full_df)
        for key in ["numeric_column", "grouped_by", "group_a", "group_a_mean", "group_b", "group_b_mean"]:
            assert key in r["mann_whitney"], f"Missing key: {key}"

    def test_anova_has_group_context(self, full_df):
        r = run_all_tests(full_df)
        # full_df has membership (4 groups) so ANOVA must be present
        for key in ["numeric_column", "grouped_by", "group_means"]:
            assert key in r["anova"], f"Missing key: {key}"

    def test_anova_uses_column_with_3_or_more_groups(self, full_df):
        r = run_all_tests(full_df)
        cat_col = r["anova"]["grouped_by"]
        assert full_df[cat_col].nunique() > 2

    def test_ttest_group_means_are_numeric(self, full_df):
        r = run_all_tests(full_df)
        assert isinstance(r["ttest"]["group_a_mean"], float)
        assert isinstance(r["ttest"]["group_b_mean"], float)

    def test_mannwhitney_group_means_are_numeric(self, full_df):
        r = run_all_tests(full_df)
        assert isinstance(r["mann_whitney"]["group_a_mean"], float)
        assert isinstance(r["mann_whitney"]["group_b_mean"], float)

    def test_anova_group_means_structure(self, full_df):
        r = run_all_tests(full_df)
        for entry in r["anova"]["group_means"]:
            assert "group" in entry
            assert "mean" in entry
            assert isinstance(entry["mean"], float)

    def test_ttest_means_direction_matches_statistic(self, full_df):
        r = run_all_tests(full_df)
        t = r["ttest"]
        if t["statistic"] < 0:
            assert t["group_a_mean"] < t["group_b_mean"]
        elif t["statistic"] > 0:
            assert t["group_a_mean"] > t["group_b_mean"]

    def test_ttest_and_mannwhitney_same_column_and_split(self, full_df):
        r = run_all_tests(full_df)
        assert r["ttest"]["numeric_column"] == r["mann_whitney"]["numeric_column"]
        assert r["ttest"]["grouped_by"] == r["mann_whitney"]["grouped_by"]
        assert r["ttest"]["group_a"] == r["mann_whitney"]["group_a"]
        assert r["ttest"]["group_b"] == r["mann_whitney"]["group_b"]

    def test_id_column_not_used(self, full_df):
        r = run_all_tests(full_df)
        assert r["ttest"]["numeric_column"] != "customer_id"
        assert r["mann_whitney"]["numeric_column"] != "customer_id"
        if "anova" in r:
            assert r["anova"]["numeric_column"] != "customer_id"

    def test_anova_group_means_match_categorical_values(self, full_df):
        r = run_all_tests(full_df)
        cat_col = r["anova"]["grouped_by"]
        expected = set(full_df[cat_col].dropna().unique())
        actual = {entry["group"] for entry in r["anova"]["group_means"]}
        assert actual == expected

    def test_no_cat_cols_returns_empty(self, no_cat_df):
        r = run_all_tests(no_cat_df)
        assert r == {}

    def test_empty_df_returns_empty(self):
        assert run_all_tests(pd.DataFrame()) == {}

    def test_chi_squared_columns_compared(self, full_df):
        r = run_all_tests(full_df)
        assert "columns_compared" in r["chi_squared"]
        assert len(r["chi_squared"]["columns_compared"]) == 2
    def test_outliers_excluded_key_present(self, full_df):
        labels = [1] * len(full_df)
        labels[0] = -1  # mark one anomaly
        r = run_all_tests(full_df, anomaly_labels=labels)
        assert r["ttest"]["outliers_excluded"] == 1
        assert r["mann_whitney"]["outliers_excluded"] == 1

    def test_anomaly_rows_excluded_from_means(self, full_df):
        df = full_df.copy()
        df.loc[0, "avg_order_value"] = 999999.0
        labels = [-1] + [1] * (len(df) - 1)
        r_clean = run_all_tests(df, anomaly_labels=labels)
        r_dirty = run_all_tests(df)
        # at least one group mean should differ between clean and dirty runs
        assert (
            r_clean["ttest"]["group_a_mean"] != r_dirty["ttest"]["group_a_mean"]
            or r_clean["ttest"]["group_b_mean"] != r_dirty["ttest"]["group_b_mean"]
        )

    def test_no_anomaly_labels_runs_normally(self, full_df):
        r = run_all_tests(full_df)
        assert "ttest" in r
        assert r["ttest"]["outliers_excluded"] == 0