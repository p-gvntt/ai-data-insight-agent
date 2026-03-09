"""
Statistical tests: t-test, Mann-Whitney U, chi-squared, ANOVA.
"""

import pandas as pd
from scipy import stats


def run_ttest(group1, group2) -> dict:
    """Independent samples t-test."""
    stat, p = stats.ttest_ind(group1.dropna(), group2.dropna())
    return {
        "test": "independent_ttest",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "significant": bool(p < 0.05),
    }


def run_mannwhitney(group1, group2) -> dict:
    """Mann-Whitney U test (non-parametric alternative to t-test)."""
    stat, p = stats.mannwhitneyu(group1.dropna(), group2.dropna(), alternative="two-sided")
    return {
        "test": "mann_whitney_u",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "significant": bool(p < 0.05),
    }


def run_anova(*groups) -> dict:
    """One-way ANOVA across multiple groups."""
    cleaned = [g.dropna() for g in groups]
    stat, p = stats.f_oneway(*cleaned)
    return {
        "test": "one_way_anova",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "significant": bool(p < 0.05),
        "n_groups": len(groups),
    }


def run_chi_squared(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Chi-squared test of independence between two categorical columns."""
    contingency = pd.crosstab(df[col_a], df[col_b])
    stat, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "test": "chi_squared",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "degrees_of_freedom": int(dof),
        "significant": bool(p < 0.05),
    }


def run_all_tests(df: pd.DataFrame) -> dict:
    """
    Auto-run appropriate tests on the DataFrame:
    - t-test + Mann-Whitney on first two numeric columns
    - ANOVA if 3+ numeric columns exist
    """
    numeric = df.select_dtypes(include="number")
    cols = numeric.columns.tolist()
    results = {}

    if len(cols) >= 2:
        results["ttest"] = run_ttest(numeric[cols[0]], numeric[cols[1]])
        results["mann_whitney"] = run_mannwhitney(numeric[cols[0]], numeric[cols[1]])

    if len(cols) >= 3:
        results["anova"] = run_anova(*[numeric[c] for c in cols[:3]])

    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if len(cat_cols) >= 2:
        try:
            results["chi_squared"] = run_chi_squared(df, cat_cols[0], cat_cols[1])
        except Exception:
            pass

    return results