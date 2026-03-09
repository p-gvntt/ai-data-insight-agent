"""
Formats the final analysis report with metadata and sections.
"""

from datetime import datetime


def generate_report(insights: str, eda: dict, patterns: dict, stats: dict) -> str:
    """
    Wrap LLM insights into a structured text report with metadata.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    shape = eda.get("shape", {})
    rows = shape.get("rows", "N/A")
    cols = shape.get("columns", "N/A")
    missing_pct = eda.get("missing", {}).get("overall_missing_pct", "N/A")
    duplicates = eda.get("duplicates", {}).get("duplicate_rows", "N/A")

    cluster_info = patterns.get("clusters", {})
    best_k = cluster_info.get("best_k", "N/A")
    anomaly_count = patterns.get("anomalies", {}).get("anomaly_count", "N/A")

    stat_tests = ", ".join(stats.keys()) if stats else "None"

    report = f"""
╔══════════════════════════════════════════════════════════╗
║           AI SENIOR DATA ANALYST — FULL REPORT           ║
╚══════════════════════════════════════════════════════════╝

Generated : {timestamp}

──────────────────────────────────────────────────────────
PIPELINE SUMMARY
──────────────────────────────────────────────────────────
  Dataset size     : {rows} rows × {cols} columns
  Missing data     : {missing_pct}% of all cells
  Duplicate rows   : {duplicates}
  Clusters found   : {best_k}
  Anomalies found  : {anomaly_count}
  Tests performed  : {stat_tests}

──────────────────────────────────────────────────────────
DETAILED INSIGHTS
──────────────────────────────────────────────────────────

{insights}

──────────────────────────────────────────────────────────
STANDARD NEXT STEPS
──────────────────────────────────────────────────────────
  • Investigate flagged anomalies manually
  • Validate cluster labels with domain experts
  • Address columns with high missing-value rates
  • Consider predictive modelling on cleaned data
  • Schedule regular re-runs as new data arrives

══════════════════════════════════════════════════════════
"""
    return report.strip()