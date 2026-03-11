"""
Full exploratory data analysis: shape, missing values, descriptive stats,
distributions, outliers, correlations, categoricals, duplicates
"""

import pandas as pd
import numpy as np
from scipy import stats


# Patterns that indicate an ID or surrogate key column — meaningless in correlations
_ID_PATTERNS = [
    "id", "_id", "id_", "customerid", "customer_id", "userid", "user_id",
    "rowid", "row_id", "index", "record_id", "recordid", "uuid", "guid",
    "order_id", "orderid", "transaction_id", "transactionid", "seq", "sequence",
]


def _is_id_like(series: pd.Series) -> bool:
    """
    Returns True if the column looks like an identifier and should be
    excluded from correlation analysis and statistical tests.
    Checks:
      1. Column name matches a known ID pattern (case-insensitive)
      2. All values are unique (nunique == len) — classic surrogate key
    """
    name_lower = series.name.lower().replace(" ", "_")
    if any(name_lower == pat or name_lower.endswith(pat) or name_lower.startswith(pat)
           for pat in _ID_PATTERNS):
        return True
    if series.nunique() == len(series):
        return True
    return False


def get_shape_info(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "string", "category"]).columns.tolist(),
        "datetime_columns": df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist(),
        "boolean_columns": df.select_dtypes(include="bool").columns.tolist(),
    }


def get_missing_analysis(df: pd.DataFrame) -> dict:
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)

    columns_with_missing = {
        col: {"count": int(missing[col]), "percent": float(pct[col])}
        for col in df.columns if missing[col] > 0
    }

    return {
        "total_missing_cells": int(missing.sum()),
        "total_cells": int(total * len(df.columns)),
        "overall_missing_pct": round(missing.sum() / (total * len(df.columns)) * 100, 2) if total > 0 else 0.0,
        "columns_with_missing": columns_with_missing,
        "complete_rows": int(df.dropna().shape[0]),
        "complete_rows_pct": round(df.dropna().shape[0] / total * 100, 2) if total > 0 else 0.0,
    }


def get_descriptive_stats(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {}

    desc = numeric.describe().T
    desc["median"] = numeric.median()
    desc["skewness"] = numeric.skew()
    desc["kurtosis"] = numeric.kurt()
    desc["iqr"] = numeric.quantile(0.75) - numeric.quantile(0.25)
    desc["cv"] = (numeric.std() / numeric.mean()).abs()
    desc["range"] = numeric.max() - numeric.min()

    return desc.round(4).to_dict()


def get_distribution_info(df: pd.DataFrame) -> dict:
    """
    Normality test per numeric column.
    - Shapiro-Wilk for n <= 5000
    - D'Agostino-Pearson (normaltest) for n > 5000
    Skips columns with zero variance.
    """
    numeric = df.select_dtypes(include="number")
    result = {}

    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 3:
            continue

        info = {}

        # Zero-variance guard
        if series.std() == 0:
            info["note"] = "constant column — skipped normality test"
            info["skewness"] = 0.0
            info["skew_label"] = "symmetric"
            info["kurtosis"] = 0.0
            info["kurtosis_label"] = "mesokurtic (normal-like)"
            result[col] = info
            continue

        # Normality test: Shapiro for small, D'Agostino for large
        sample = series if len(series) <= 5000 else series.sample(5000, random_state=42)
        try:
            if len(series) <= 5000:
                stat, p = stats.shapiro(sample)
                info["normality_test"] = "shapiro-wilk"
            else:
                stat, p = stats.normaltest(sample)
                info["normality_test"] = "dagostino-pearson"
            info["normality_stat"] = round(float(stat), 4)
            info["normality_p"] = round(float(p), 4)
            info["is_normal"] = bool(p > 0.05)
        except Exception:
            info["normality_test"] = None
            info["normality_stat"] = None
            info["normality_p"] = None
            info["is_normal"] = None

        # Histogram bins for dashboard use
        counts, bin_edges = np.histogram(series.dropna(), bins=10)
        info["histogram"] = {
            "counts": counts.tolist(),
            "bin_edges": [round(float(e), 4) for e in bin_edges],
        }

        skew = float(series.skew())
        info["skewness"] = round(skew, 4)
        info["skew_label"] = (
            "symmetric" if abs(skew) < 0.5
            else "left-skewed" if skew < 0
            else "right-skewed"
        )

        kurt = float(series.kurt())
        info["kurtosis"] = round(kurt, 4)
        info["kurtosis_label"] = (
            "leptokurtic (heavy tails)" if kurt > 1
            else "platykurtic (light tails)" if kurt < -1
            else "mesokurtic (normal-like)"
        )

        result[col] = info

    return result


def get_outlier_analysis(df: pd.DataFrame) -> dict:
    """
    IQR and Z-score outlier detection.
    Skips columns with zero variance to avoid Z-score RuntimeWarning.
    """
    numeric = df.select_dtypes(include="number")
    result = {}

    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 4:
            continue

        # IQR method
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_outliers = series[(series < lower) | (series > upper)]

        # Z-score method — skip if zero variance
        if series.std() == 0:
            z_outlier_count = 0
            z_outlier_pct = 0.0
        else:
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]
            z_outlier_count = int(len(z_outliers))
            z_outlier_pct = round(len(z_outliers) / len(series) * 100, 2)

        result[col] = {
            "iqr": {
                "lower_fence": round(float(lower), 4),
                "upper_fence": round(float(upper), 4),
                "outlier_count": int(len(iqr_outliers)),
                "outlier_pct": round(len(iqr_outliers) / len(series) * 100, 2),
            },
            "zscore": {
                "outlier_count": z_outlier_count,
                "outlier_pct": z_outlier_pct,
            },
        }

    return result


def get_correlation_analysis(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    Pearson + Spearman correlation matrices.
    Notable pairs are those with |pearson_r| > threshold (default 0.5).
    ID-like columns (name matches known ID patterns, or all values unique)
    are excluded before computing pairs — they produce spurious correlations.
    Each notable pair includes a plain_english description of direction
    so downstream LLMs never need to interpret the sign themselves.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {}

    # Filter out ID-like columns from pair computation only
    non_id_cols = [col for col in numeric.columns if not _is_id_like(numeric[col])]

    pearson  = numeric.corr(method="pearson").round(4).to_dict()
    spearman = numeric.corr(method="spearman").round(4).to_dict()

    if len(non_id_cols) < 2:
        return {"pearson": pearson, "spearman": spearman, "notable_pairs": []}

    corr_matrix = numeric[non_id_cols].corr(method="pearson")
    cols = corr_matrix.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > threshold:
                pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "pearson_r": round(float(val), 4),
                    "strength": "strong" if abs(val) > 0.7 else "moderate",
                    "direction": "positive" if val > 0 else "negative",
                })
    pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    for p in pairs:
        if p["direction"] == "negative":
            p["plain_english"] = (
                f"As {p['col_a']} increases, {p['col_b']} tends to decrease "
                f"(r={p['pearson_r']})"
            )
        else:
            p["plain_english"] = (
                f"As {p['col_a']} increases, {p['col_b']} tends to increase "
                f"(r={p['pearson_r']})"
            )

    return {"pearson": pearson, "spearman": spearman, "notable_pairs": pairs}


def get_categorical_analysis(df: pd.DataFrame, max_categories: int = 30) -> dict:
    """
    Value counts, cardinality, mode, missing count, and class imbalance
    detection for categorical columns.
    """
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    result = {}

    for col in cat_cols:
        series = df[col].dropna()
        vc = series.value_counts()
        unique_count = int(series.nunique())
        total = len(series)

        # Class imbalance: flag if dominant class > 80% of non-null values
        top_freq = int(vc.iloc[0]) if not vc.empty else 0
        imbalance_ratio = round(top_freq / total * 100, 2) if total > 0 else 0.0
        is_imbalanced = imbalance_ratio > 80

        result[col] = {
            "unique_count": unique_count,
            "cardinality": "high" if unique_count > max_categories else "low",
            "most_common": vc.head(10).to_dict(),
            "least_common": vc.tail(5).to_dict() if len(vc) > 5 else {},
            "missing_count": int(df[col].isnull().sum()),
            "mode": str(series.mode().iloc[0]) if not series.empty else None,
            "imbalance": {
                "dominant_class_pct": imbalance_ratio,
                "is_imbalanced": is_imbalanced,
            },
        }

    return result


def get_duplicate_analysis(df: pd.DataFrame) -> dict:
    """
    Full duplicate row analysis including column-level duplicate detection.
    """
    n_dups = int(df.duplicated().sum())

    # Find columns that are entirely duplicated (identical to another column)
    duplicate_column_pairs = []
    cols = df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            try:
                if df[cols[i]].equals(df[cols[j]]):
                    duplicate_column_pairs.append((cols[i], cols[j]))
            except Exception:
                pass

    return {
        "duplicate_rows": n_dups,
        "duplicate_pct": round(n_dups / len(df) * 100, 2) if len(df) > 0 else 0.0,
        "unique_rows": len(df) - n_dups,
        "duplicate_column_pairs": duplicate_column_pairs,
    }


def run_basic_eda(df: pd.DataFrame) -> dict:
    """Master EDA function — returns all analysis sections."""
    return {
        "shape": get_shape_info(df),
        "missing": get_missing_analysis(df),
        "descriptive_stats": get_descriptive_stats(df),
        "distributions": get_distribution_info(df),
        "outliers": get_outlier_analysis(df),
        "correlations": get_correlation_analysis(df),
        "categoricals": get_categorical_analysis(df),
        "duplicates": get_duplicate_analysis(df),
    }