"""
Tests for analysis/eda.py
Run with: pytest tests/test_eda.py -v
"""

import pytest
import pandas as pd
import numpy as np

from analysis.eda import (
    run_basic_eda,
    get_shape_info,
    get_missing_analysis,
    get_descriptive_stats,
    get_distribution_info,
    get_outlier_analysis,
    get_correlation_analysis,
    get_categorical_analysis,
    get_duplicate_analysis,
)


# Fixtures
@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "age":    [25, 30, 35, 40, 45],
        "salary": [30000, 45000, 60000, 80000, 100000],
        "score":  [7.5, 8.0, 6.5, 9.0, 7.0],
    })

@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [None, None, 3.0, 4.0, 5.0],
        "c": [10, 20, 30, 40, 50],
    })

@pytest.fixture
def df_with_categoricals():
    return pd.DataFrame({
        "city":    ["Rome", "Milan", "Rome", "Naples", "Rome"],
        "status":  ["active", "inactive", "active", "active", "inactive"],
        "revenue": [100, 200, 150, 300, 250],
    })

@pytest.fixture
def df_with_outliers():
    normal = list(range(1, 20))
    return pd.DataFrame({"x": normal + [1000], "y": normal + [-500]})

@pytest.fixture
def df_with_duplicates():
    return pd.DataFrame({"a": [1, 1, 2, 3], "b": [4, 4, 5, 6]})

@pytest.fixture
def single_col_df():
    return pd.DataFrame({"x": [1, 2, 3, 4, 5]})

@pytest.fixture
def empty_df():
    return pd.DataFrame()

@pytest.fixture
def zero_variance_df():
    return pd.DataFrame({"x": [5, 5, 5, 5, 5], "y": [1, 2, 3, 4, 5]})

@pytest.fixture
def imbalanced_df():
    # "cat" is 90%, so "a" is imbalanced
    return pd.DataFrame({"cat": ["a"] * 9 + ["b"]})

@pytest.fixture
def duplicate_cols_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [4, 5, 6]})


# Shape Info
class TestGetShapeInfo:
    def test_rows_and_cols(self, simple_df):
        r = get_shape_info(simple_df)
        assert r["rows"] == 5
        assert r["columns"] == 3

    def test_numeric_columns(self, simple_df):
        r = get_shape_info(simple_df)
        assert set(r["numeric_columns"]) == {"age", "salary", "score"}

    def test_categorical_columns(self, df_with_categoricals):
        r = get_shape_info(df_with_categoricals)
        assert "city" in r["categorical_columns"]

    def test_dtypes_keys(self, simple_df):
        r = get_shape_info(simple_df)
        assert "age" in r["dtypes"]

    def test_empty_df(self, empty_df):
        r = get_shape_info(empty_df)
        assert r["rows"] == 0 and r["columns"] == 0


# Missing Analysis
class TestGetMissingAnalysis:
    def test_finds_missing_cols(self, df_with_missing):
        r = get_missing_analysis(df_with_missing)
        assert "a" in r["columns_with_missing"]
        assert "b" in r["columns_with_missing"]

    def test_correct_count(self, df_with_missing):
        r = get_missing_analysis(df_with_missing)
        assert r["columns_with_missing"]["b"]["count"] == 2

    def test_correct_pct(self, df_with_missing):
        r = get_missing_analysis(df_with_missing)
        assert r["columns_with_missing"]["a"]["percent"] == 20.0

    def test_no_missing(self, simple_df):
        r = get_missing_analysis(simple_df)
        assert r["columns_with_missing"] == {}
        assert r["total_missing_cells"] == 0

    def test_complete_rows(self, df_with_missing):
        # row 0: b=NaN, row 1: b=NaN, row 2: a=NaN → only rows 3 and 4 fully complete
        r = get_missing_analysis(df_with_missing)
        assert r["complete_rows"] == 2


# Descriptive Stats
class TestGetDescriptiveStats:
    def test_required_keys(self, simple_df):
        r = get_descriptive_stats(simple_df)
        for key in ["mean", "median", "std", "skewness", "kurtosis", "iqr", "cv"]:
            assert key in r

    def test_median_age(self, simple_df):
        r = get_descriptive_stats(simple_df)
        assert r["median"]["age"] == pytest.approx(35.0, abs=0.01)

    def test_empty_returns_empty(self, empty_df):
        assert get_descriptive_stats(empty_df) == {}

    def test_single_column(self, single_col_df):
        r = get_descriptive_stats(single_col_df)
        assert r["mean"]["x"] == pytest.approx(3.0, abs=0.01)


# Distribution Info
class TestGetDistributionInfo:
    def test_per_column_keys(self, simple_df):
        r = get_distribution_info(simple_df)
        for col in ["age", "salary", "score"]:
            assert col in r

    def test_normality_test_name_small(self, simple_df):
        r = get_distribution_info(simple_df)
        assert r["age"]["normality_test"] == "shapiro-wilk"

    def test_normality_keys(self, simple_df):
        r = get_distribution_info(simple_df)
        assert "normality_stat" in r["age"]
        assert "normality_p" in r["age"]
        assert "is_normal" in r["age"]

    def test_skew_label_valid(self, simple_df):
        r = get_distribution_info(simple_df)
        assert r["age"]["skew_label"] in ("symmetric", "left-skewed", "right-skewed")

    def test_kurtosis_label_valid(self, simple_df):
        r = get_distribution_info(simple_df)
        assert "kurtosis_label" in r["age"]

    def test_histogram_present(self, simple_df):
        r = get_distribution_info(simple_df)
        assert "histogram" in r["age"]
        assert "counts" in r["age"]["histogram"]
        assert "bin_edges" in r["age"]["histogram"]
        assert len(r["age"]["histogram"]["counts"]) == 10

    def test_zero_variance_skipped(self, zero_variance_df):
        r = get_distribution_info(zero_variance_df)
        assert "note" in r["x"]
        assert "normality_stat" not in r["x"]

    def test_too_few_rows_skipped(self):
        df = pd.DataFrame({"x": [1, 2]})
        assert get_distribution_info(df) == {}

    def test_empty_df(self, empty_df):
        assert get_distribution_info(empty_df) == {}

    def test_large_df_uses_dagostino(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 6000)})
        r = get_distribution_info(df)
        assert r["x"]["normality_test"] == "dagostino-pearson"


# Outlier Analysis

class TestGetOutlierAnalysis:
    def test_iqr_detects_outlier(self, df_with_outliers):
        r = get_outlier_analysis(df_with_outliers)
        assert r["x"]["iqr"]["outlier_count"] >= 1

    def test_zscore_detects_outlier(self, df_with_outliers):
        r = get_outlier_analysis(df_with_outliers)
        assert r["x"]["zscore"]["outlier_count"] >= 1

    def test_pct_in_range(self, df_with_outliers):
        r = get_outlier_analysis(df_with_outliers)
        for col in r:
            assert 0 <= r[col]["iqr"]["outlier_pct"] <= 100
            assert 0 <= r[col]["zscore"]["outlier_pct"] <= 100

    def test_fences_logical(self, df_with_outliers):
        r = get_outlier_analysis(df_with_outliers)
        for col in r:
            assert r[col]["iqr"]["lower_fence"] < r[col]["iqr"]["upper_fence"]

    def test_zero_variance_no_crash(self, zero_variance_df):
        r = get_outlier_analysis(zero_variance_df)
        assert r["x"]["zscore"]["outlier_count"] == 0
        assert r["x"]["zscore"]["outlier_pct"] == 0.0


# Correlation Analysis
class TestGetCorrelationAnalysis:
    def test_pearson_and_spearman(self, simple_df):
        r = get_correlation_analysis(simple_df)
        assert "pearson" in r and "spearman" in r

    def test_notable_pairs_no_self(self, simple_df):
        r = get_correlation_analysis(simple_df)
        for pair in r["notable_pairs"]:
            assert pair["col_a"] != pair["col_b"]

    def test_single_col_returns_empty(self, single_col_df):
        assert get_correlation_analysis(single_col_df) == {}

    def test_perfect_correlation(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        r = get_correlation_analysis(df)
        assert len(r["notable_pairs"]) == 1
        assert r["notable_pairs"][0]["pearson_r"] == pytest.approx(1.0, abs=0.001)
        assert r["notable_pairs"][0]["direction"] == "positive"

    def test_custom_threshold(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        # threshold=0.99 - still finds perfect correlation
        r = get_correlation_analysis(df, threshold=0.99)
        assert len(r["notable_pairs"]) == 1
        # threshold=1.01 - nothing qualifies
        r2 = get_correlation_analysis(df, threshold=1.01)
        assert len(r2["notable_pairs"]) == 0


# Categorical Analysis
class TestGetCategoricalAnalysis:
    def test_finds_cat_cols(self, df_with_categoricals):
        r = get_categorical_analysis(df_with_categoricals)
        assert "city" in r and "status" in r

    def test_unique_count(self, df_with_categoricals):
        r = get_categorical_analysis(df_with_categoricals)
        assert r["city"]["unique_count"] == 3

    def test_mode_correct(self, df_with_categoricals):
        r = get_categorical_analysis(df_with_categoricals)
        assert r["city"]["mode"] == "Rome"

    def test_most_common(self, df_with_categoricals):
        r = get_categorical_analysis(df_with_categoricals)
        assert r["city"]["most_common"]["Rome"] == 3

    def test_no_cats_returns_empty(self, simple_df):
        assert get_categorical_analysis(simple_df) == {}

    def test_missing_count(self):
        df = pd.DataFrame({"cat": ["a", None, "b", "a", None]})
        r = get_categorical_analysis(df)
        assert r["cat"]["missing_count"] == 2

    def test_imbalance_detected(self, imbalanced_df):
        r = get_categorical_analysis(imbalanced_df)
        assert r["cat"]["imbalance"]["is_imbalanced"] is True
        assert r["cat"]["imbalance"]["dominant_class_pct"] == pytest.approx(90.0, abs=0.01)

    def test_balanced_not_flagged(self, df_with_categoricals):
        r = get_categorical_analysis(df_with_categoricals)
        assert r["city"]["imbalance"]["is_imbalanced"] is False


# Duplicate Analysis
class TestGetDuplicateAnalysis:
    def test_detects_duplicates(self, df_with_duplicates):
        r = get_duplicate_analysis(df_with_duplicates)
        assert r["duplicate_rows"] == 1

    def test_no_duplicates(self, simple_df):
        r = get_duplicate_analysis(simple_df)
        assert r["duplicate_rows"] == 0

    def test_pct_correct(self, df_with_duplicates):
        r = get_duplicate_analysis(df_with_duplicates)
        assert r["duplicate_pct"] == pytest.approx(25.0, abs=0.01)

    def test_unique_rows(self, df_with_duplicates):
        r = get_duplicate_analysis(df_with_duplicates)
        assert r["unique_rows"] == 3

    def test_duplicate_column_pairs_detected(self, duplicate_cols_df):
        r = get_duplicate_analysis(duplicate_cols_df)
        assert ("a", "b") in r["duplicate_column_pairs"]

    def test_no_duplicate_columns(self, simple_df):
        r = get_duplicate_analysis(simple_df)
        assert r["duplicate_column_pairs"] == []


# run_basic_eda
class TestRunBasicEda:
    def test_all_sections_present(self, simple_df):
        r = run_basic_eda(simple_df)
        for key in ["shape", "missing", "descriptive_stats", "distributions",
                    "outliers", "correlations", "categoricals", "duplicates"]:
            assert key in r, f"Missing section: {key}"

    def test_correct_row_count(self, simple_df):
        r = run_basic_eda(simple_df)
        assert r["shape"]["rows"] == 5

    def test_missing_section(self, df_with_missing):
        r = run_basic_eda(df_with_missing)
        assert r["missing"]["total_missing_cells"] == 3

    def test_mixed_df(self, df_with_categoricals):
        r = run_basic_eda(df_with_categoricals)
        assert "city" in r["categoricals"]
        assert "revenue" in r["shape"]["numeric_columns"]

    def test_large_df(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a":     rng.normal(0, 1, 10000),
            "b":     rng.exponential(1, 10000),
            "c":     rng.integers(0, 100, 10000).astype(float),
            "label": rng.choice(["x", "y", "z"], 10000),
        })
        r = run_basic_eda(df)
        assert r["shape"]["rows"] == 10000