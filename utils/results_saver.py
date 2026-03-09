"""
Saves analysis results and reports to the results/ folder with a timestamp.
"""

import json
import os
from datetime import datetime


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(dataset_name: str, eda: dict, patterns: dict, stats: dict, report: str) -> dict:
    """
    Save full analysis results to results/<timestamp>_<dataset_name>/.

    Creates two files:
      - analysis.json  — structured output from all pipeline stages
      - report.txt     — the final LLM-generated report

    Returns the paths of the saved files.
    """
    _ensure_dir()

    base_name = os.path.splitext(dataset_name)[0]
    folder_name = f"{_timestamp()}_{base_name}"
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save structured JSON
    analysis = {
        "dataset": dataset_name,
        "generated_at": datetime.now().isoformat(),
        "eda": eda,
        "patterns": patterns,
        "statistics": stats,
    }
    json_path = os.path.join(folder_path, "analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Save text report
    report_path = os.path.join(folder_path, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    return {
        "folder": folder_path,
        "analysis_json": json_path,
        "report_txt": report_path,
    }