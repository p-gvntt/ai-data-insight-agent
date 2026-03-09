"""
Statistical tests: t-test, Mann-Whitney U, chi-squared, ANOVA.

All group-based tests (t-test, Mann-Whitney, ANOVA) split a single
numeric column by a categorical grouping column, so results are
always meaningful and interpretable.
"""

import pandas as pd
from scipy import stats


def run_ttest(group1, group2) -> dict:
    """Independent samples t-test between two groups."""
    stat, p = stats.ttest_ind(group1.dropna(), group2.dropna())
    return {
        "test": "independent_ttest",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "significant": bool(p < 0.05),
    }


def run_mannwhitney(group1, group2) -> dict:
    """Mann-Whitney U test between two groups (non-parametric)."""
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


_PRIORITY_PATTERNS = [
    "revenue", "sales", "order_value", "spend", "purchase",
    "profit", "churn", "score", "salary", "price", "amount",
    "value", "income", "cost", "margin", "conversion", "rate",
]


def _pick_best_numeric(df: pd.DataFrame, meaningful_cols: list) -> str:
    """
    Pick the most analytically interesting numeric column.
    Uses a priority pattern list based on common business column names.
    Falls back to highest coefficient of variation if no pattern matches.
    """
    cols_lower = {col: col.lower() for col in meaningful_cols}

    for pattern in _PRIORITY_PATTERNS:
        for col, col_low in cols_lower.items():
            if pattern in col_low:
                return col

    # Fallback: highest CV
    numeric = df[meaningful_cols]
    cv = numeric.std() / numeric.mean().abs()
    return cv.idxmax()


def _is_id_like(series: pd.Series) -> bool:
    """Return True if the column looks like a row ID."""
    return series.nunique() == len(series)


def _exclude_anomalies(df: pd.DataFrame, anomaly_labels: list) -> pd.DataFrame:
    """
    Return a copy of df with anomaly rows removed so group means
    are not contaminated by outliers already flagged by the pipeline.
    anomaly_labels is the raw labels list from patterns["anomalies"]["labels"]
    where -1 = anomaly and 1 = normal.
    """
    if not anomaly_labels or len(anomaly_labels) != len(df):
        return df
    mask = [label == 1 for label in anomaly_labels]
    return df[mask].reset_index(drop=True)


def run_all_tests(df: pd.DataFrame, anomaly_labels: list = None) -> dict:
    """
    Run meaningful statistical tests:

    - t-test: best numeric column split by a binary categorical column
    - Mann-Whitney: same split as t-test (non-parametric version)
    - ANOVA: best numeric column split across 3+ groups of a categorical column
    - Chi-squared: association between two categorical columns

    anomaly_labels: optional list of Isolation Forest labels (1=normal, -1=anomaly).
    When provided, anomaly rows are excluded before computing group means.
    """
    results = {}

    # Exclude anomaly rows before running any tests
    clean_df = _exclude_anomalies(df, anomaly_labels) if anomaly_labels else df
    outliers_excluded = len(df) - len(clean_df)

    numeric = clean_df.select_dtypes(include="number")
    cat_cols = clean_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # Filter out ID-like numeric columns
    meaningful_numeric = [
        col for col in numeric.columns if not _is_id_like(numeric[col])
    ]

    # Filter out very high-cardinality categoricals (e.g. name columns)
    meaningful_cat = [
        col for col in cat_cols if clean_df[col].nunique() <= 20
    ]

    if not meaningful_numeric or not meaningful_cat:
        return results

    target_num = _pick_best_numeric(clean_df, meaningful_numeric)

    # t-test + Mann-Whitney — binary categorical split
    binary_cat = next(
        (col for col in meaningful_cat if clean_df[col].nunique() == 2), None
    )
    if binary_cat is None:
        binary_cat = meaningful_cat[0]

    group_vals = clean_df[binary_cat].dropna().unique()[:2]
    g1 = clean_df.loc[clean_df[binary_cat] == group_vals[0], target_num].dropna()
    g2 = clean_df.loc[clean_df[binary_cat] == group_vals[1], target_num].dropna()

    if len(g1) > 1 and len(g2) > 1:
        results["ttest"] = run_ttest(g1, g2)
        results["ttest"]["numeric_column"] = target_num
        results["ttest"]["grouped_by"] = binary_cat
        results["ttest"]["group_a"] = str(group_vals[0])
        results["ttest"]["group_a_mean"] = round(float(g1.mean()), 4)
        results["ttest"]["group_b"] = str(group_vals[1])
        results["ttest"]["group_b_mean"] = round(float(g2.mean()), 4)
        results["ttest"]["outliers_excluded"] = outliers_excluded

        results["mann_whitney"] = run_mannwhitney(g1, g2)
        results["mann_whitney"]["numeric_column"] = target_num
        results["mann_whitney"]["grouped_by"] = binary_cat
        results["mann_whitney"]["group_a"] = str(group_vals[0])
        results["mann_whitney"]["group_a_mean"] = round(float(g1.mean()), 4)
        results["mann_whitney"]["group_b"] = str(group_vals[1])
        results["mann_whitney"]["group_b_mean"] = round(float(g2.mean()), 4)
        results["mann_whitney"]["outliers_excluded"] = outliers_excluded

    # ANOVA — only runs when 3+ groups exist to add value beyond t-test
    best_cat_for_anova = max(
        (c for c in meaningful_cat if clean_df[c].nunique() > 2),
        key=lambda c: clean_df[c].nunique(),
        default=None,
    )
    if best_cat_for_anova is not None:
        anova_groups = []
        anova_group_labels = []
        for name, group in clean_df.groupby(best_cat_for_anova):
            g = group[target_num].dropna()
            if len(g) > 1:
                anova_groups.append(g)
                anova_group_labels.append({"group": str(name), "mean": round(float(g.mean()), 4)})

        if len(anova_groups) >= 3:
            results["anova"] = run_anova(*anova_groups)
            results["anova"]["numeric_column"] = target_num
            results["anova"]["grouped_by"] = best_cat_for_anova
            results["anova"]["group_means"] = anova_group_labels
            results["anova"]["outliers_excluded"] = outliers_excluded

    # Chi-squared — not affected by outliers, uses clean_df for consistency
    if len(meaningful_cat) >= 2:
        try:
            results["chi_squared"] = run_chi_squared(clean_df, meaningful_cat[0], meaningful_cat[1])
            results["chi_squared"]["columns_compared"] = [meaningful_cat[0], meaningful_cat[1]]
        except Exception:
            pass

    return results