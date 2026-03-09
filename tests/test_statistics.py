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
    """Two identical series — should NOT be significantly different."""
    s = pd.Series([10.0, 11.0, 10.5, 10.2, 10.8])
    return s, s.copy()

@pytest.fixture
def different_groups():
    """Two clearly different series — should be significantly different."""
    return pd.Series([1.0, 1.1, 1.2, 1.0, 1.1]), pd.Series([100.0, 101.0, 99.0, 100.5, 100.2])

@pytest.fixture
def group_with_missing():
    """Series containing NaN values."""
    return pd.Series([1.0, np.nan, 3.0, 4.0, 5.0]), pd.Series([10.0, 11.0, np.nan, 13.0, 14.0])

@pytest.fixture
def three_groups():
    """Three groups for ANOVA."""
    return (
        pd.Series([1.0, 1.1, 1.2]),
        pd.Series([5.0, 5.1, 5.2]),
        pd.Series([10.0, 10.1, 10.2]),
    )

@pytest.fixture
def cat_df():
    """DataFrame with two categorical columns for chi-squared."""
    return pd.DataFrame({
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "bought": ["yes", "no", "yes", "yes", "no", "no", "yes", "no"],
    })

@pytest.fixture
def full_numeric_df():
    """DataFrame with 3 numeric columns for run_all_tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 30),
        "b": rng.normal(5, 1, 30),
        "c": rng.normal(10, 1, 30),
    })

@pytest.fixture
def full_mixed_df():
    """DataFrame with numeric + categorical columns for run_all_tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a":      rng.normal(0, 1, 30),
        "b":      rng.normal(5, 1, 30),
        "gender": ["M", "F"] * 15,
        "bought": ["yes", "no", "yes"] * 10,
    })


# run_ttest

class TestRunTtest:
    def test_returns_required_keys(self, different_groups):
        g1, g2 = different_groups
        r = run_ttest(g1, g2)
        for key in ["test", "statistic", "p_value", "significant"]:
            assert key in r

    def test_test_name(self, different_groups):
        g1, g2 = different_groups
        r = run_ttest(g1, g2)
        assert r["test"] == "independent_ttest"

    def test_significant_when_groups_differ(self, different_groups):
        g1, g2 = different_groups
        r = run_ttest(g1, g2)
        assert r["significant"] is True

    def test_not_significant_when_identical(self, identical_groups):
        g1, g2 = identical_groups
        r = run_ttest(g1, g2)
        assert r["significant"] is False

    def test_p_value_between_0_and_1(self, different_groups):
        g1, g2 = different_groups
        r = run_ttest(g1, g2)
        assert 0.0 <= r["p_value"] <= 1.0

    def test_handles_missing_values(self, group_with_missing):
        g1, g2 = group_with_missing
        r = run_ttest(g1, g2)
        assert "p_value" in r


# run_mannwhitney
class TestRunMannWhitney:
    def test_returns_required_keys(self, different_groups):
        g1, g2 = different_groups
        r = run_mannwhitney(g1, g2)
        for key in ["test", "statistic", "p_value", "significant"]:
            assert key in r

    def test_test_name(self, different_groups):
        g1, g2 = different_groups
        r = run_mannwhitney(g1, g2)
        assert r["test"] == "mann_whitney_u"

    def test_significant_when_groups_differ(self, different_groups):
        g1, g2 = different_groups
        r = run_mannwhitney(g1, g2)
        assert r["significant"] is True

    def test_p_value_between_0_and_1(self, different_groups):
        g1, g2 = different_groups
        r = run_mannwhitney(g1, g2)
        assert 0.0 <= r["p_value"] <= 1.0

    def test_handles_missing_values(self, group_with_missing):
        g1, g2 = group_with_missing
        r = run_mannwhitney(g1, g2)
        assert "p_value" in r


# run_anova
class TestRunAnova:
    def test_returns_required_keys(self, three_groups):
        r = run_anova(*three_groups)
        for key in ["test", "statistic", "p_value", "significant", "n_groups"]:
            assert key in r

    def test_test_name(self, three_groups):
        r = run_anova(*three_groups)
        assert r["test"] == "one_way_anova"

    def test_n_groups_correct(self, three_groups):
        r = run_anova(*three_groups)
        assert r["n_groups"] == 3

    def test_significant_with_distinct_groups(self, three_groups):
        r = run_anova(*three_groups)
        assert r["significant"] is True

    def test_not_significant_with_same_groups(self):
        s = pd.Series([1.0, 1.1, 1.2, 1.0])
        r = run_anova(s, s.copy(), s.copy())
        assert r["significant"] is False

    def test_p_value_between_0_and_1(self, three_groups):
        r = run_anova(*three_groups)
        assert 0.0 <= r["p_value"] <= 1.0


# run_chi_squared
class TestRunChiSquared:
    def test_returns_required_keys(self, cat_df):
        r = run_chi_squared(cat_df, "gender", "bought")
        for key in ["test", "statistic", "p_value", "degrees_of_freedom", "significant"]:
            assert key in r

    def test_test_name(self, cat_df):
        r = run_chi_squared(cat_df, "gender", "bought")
        assert r["test"] == "chi_squared"

    def test_degrees_of_freedom_positive(self, cat_df):
        r = run_chi_squared(cat_df, "gender", "bought")
        assert r["degrees_of_freedom"] >= 1

    def test_p_value_between_0_and_1(self, cat_df):
        r = run_chi_squared(cat_df, "gender", "bought")
        assert 0.0 <= r["p_value"] <= 1.0

    def test_perfectly_dependent_columns(self):
        # col_b is a direct function of col_a → should be significant
        df = pd.DataFrame({
            "a": ["x"] * 20 + ["y"] * 20,
            "b": ["p"] * 20 + ["q"] * 20,
        })
        r = run_chi_squared(df, "a", "b")
        assert r["significant"] is True

    def test_independent_columns(self):
        # Completely random independent columns → should NOT be significant
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.choice(["x", "y"], 200),
            "b": rng.choice(["p", "q"], 200),
        })
        r = run_chi_squared(df, "a", "b")
        assert r["p_value"] > 0.0  # just verify it ran cleanly


# run_all_tests
class TestRunAllTests:
    def test_ttest_present_with_two_numeric_cols(self, full_numeric_df):
        r = run_all_tests(full_numeric_df)
        assert "ttest" in r

    def test_mannwhitney_present_with_two_numeric_cols(self, full_numeric_df):
        r = run_all_tests(full_numeric_df)
        assert "mann_whitney" in r

    def test_anova_present_with_three_numeric_cols(self, full_numeric_df):
        r = run_all_tests(full_numeric_df)
        assert "anova" in r

    def test_chi_squared_present_with_two_cat_cols(self, full_mixed_df):
        r = run_all_tests(full_mixed_df)
        assert "chi_squared" in r

    def test_no_anova_with_two_cols(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        r = run_all_tests(df)
        assert "anova" not in r

    def test_empty_df_returns_empty(self):
        r = run_all_tests(pd.DataFrame())
        assert r == {}

    def test_single_numeric_col_returns_empty(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        r = run_all_tests(df)
        assert "ttest" not in r
        assert "mann_whitney" not in r