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

    next_steps = [
        "  • Investigate flagged anomalies manually",
        "  • Validate cluster labels with domain experts",
    ]

    if isinstance(missing_pct, float) and missing_pct > 0:
        next_steps.append("  • Address columns with high missing-value rates")

    if isinstance(duplicates, int) and duplicates > 0:
        next_steps.append("  • Deduplicate rows before further modelling")

    next_steps += [
        "  • Consider predictive modelling on cleaned data",
        "  • Schedule regular re-runs as new data arrives",
    ]

    next_steps_str = "\n".join(next_steps)

    report = f"""```
╔══════════════════════════════════════════════════════════╗
║           AI SENIOR DATA ANALYST — FULL REPORT           ║
╚══════════════════════════════════════════════════════════╝
  Generated : {timestamp}
  Dataset size     : {rows} rows × {cols} columns
  Missing data     : {missing_pct}% of all cells
  Duplicate rows   : {duplicates}
  Clusters found   : {best_k}
  Anomalies found  : {anomaly_count}
  Tests performed  : {stat_tests}
```
---
{insights}
---
```
  Next steps
{next_steps_str}
```
"""
    return report