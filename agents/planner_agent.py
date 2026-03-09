"""
Defines the analysis pipeline steps.
"""


def create_plan() -> list[str]:
    return [
        "load_dataset",
        "run_basic_eda",
        "discover_clusters",
        "detect_anomalies",
        "run_statistical_tests",
        "generate_insights",
        "generate_report",
    ]