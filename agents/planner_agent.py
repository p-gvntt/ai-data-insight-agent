"""
Planner Agent — inspects the dataset and builds a dynamic analysis plan
before any analysis runs. Downstream agents read the plan and decide
what to execute rather than always running everything.
"""
import pandas as pd


_TARGET_PATTERNS = [
    "survived", "churn", "default", "fraud", "converted", "purchased",
    "clicked", "subscribed", "cancelled", "returned", "outcome", "target",
    "label", "result", "response", "attrition", "dropout",
]

_TIME_PATTERNS = [
    "date", "time", "year", "month", "week", "day", "timestamp",
    "created_at", "updated_at", "order_date", "purchase_date",
]

_SEGMENT_PAIRS = [
    ("income", "spend"), ("income", "score"),
    ("salary", "score"), ("revenue", "score"),
    ("income", "spending"),
]


def _detect_target_variable(df: pd.DataFrame):
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        if any(pat in col_lower for pat in _TARGET_PATTERNS):
            if df[col].nunique() == 2:
                return col
    return None


def _detect_time_column(df: pd.DataFrame):
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        if any(pat in col_lower for pat in _TIME_PATTERNS):
            return col
    return None


def create_plan(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()
    n_rows, n_cols = df.shape

    steps = ["load_dataset", "run_basic_eda"]
    skip  = []
    focus = []
    notes = []

    # Detect target variable
    target = _detect_target_variable(df)
    if target:
        focus.append(
            f"binary target detected: '{target}' — identify which variables are most "
            f"associated with {target} and recommend predictive modelling for this target"
        )

    # Detect time column
    time_col = _detect_time_column(df)
    if time_col:
        focus.append(f"time dimension detected: '{time_col}' — consider trend analysis")

    # Clustering
    if len(numeric_cols) >= 2 and n_rows >= 50:
        steps.append("discover_clusters")
    else:
        skip.append("discover_clusters")
        notes.append(f"clustering skipped: need >=2 numeric cols and >=50 rows (got {len(numeric_cols)} cols, {n_rows} rows)")

    # Anomaly detection
    if len(numeric_cols) >= 1 and n_rows >= 20:
        steps.append("detect_anomalies")
    else:
        skip.append("detect_anomalies")
        notes.append("anomaly detection skipped: need >=1 numeric col and >=20 rows")

    # Statistical tests
    if len(numeric_cols) >= 1 and len(cat_cols) >= 1:
        steps.append("run_statistical_tests")
    else:
        skip.append("run_statistical_tests")
        notes.append("statistical tests skipped: need at least 1 numeric and 1 categorical column")

    steps += ["generate_insights", "generate_report"]

    # Segment profiling hint
    for col_a, col_b in _SEGMENT_PAIRS:
        matches = [c for c in numeric_cols if col_a in c.lower() or col_b in c.lower()]
        if len(matches) >= 2:
            focus.append(
                f"segment profiling recommended: '{matches[0]}' vs '{matches[1]}' "
                f"likely define meaningful customer archetypes — describe what each "
                f"cluster represents in terms of these two variables"
            )
            break

    # Dataset size warnings
    if n_rows < 100:
        focus.append(f"small dataset ({n_rows} rows) — interpret all findings with caution")
    if n_cols > 20:
        focus.append("wide dataset — consider dimensionality reduction before modelling")

    return {
        "steps":  steps,
        "skip":   skip,
        "focus":  focus,
        "notes":  notes,
        "target": target,
    }