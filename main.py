"""
main.py — Orchestrates the full AI data analysis pipeline.
"""
import os
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env

from analysis.data_loader import load_dataset
from agents.planner_agent import create_plan
from agents.eda_agent import eda_agent
from agents.pattern_agent import pattern_agent
from agents.stats_agent import stats_agent
from agents.insight_agent import generate_insights
from agents.report_agent import generate_report
from utils.results_saver import save_results
from analysis.visualization import generate_all_charts, save_charts


def run_analysis(file) -> dict:
    """
    Full pipeline:
      1. Load dataset
      2. Plan — inspect data and decide which analyses to run
      3. Run EDA
      4. Detect patterns (clusters + anomalies) — conditionally
      5. Run statistical tests — conditionally
      6. Generate LLM insights — with planner focus hints
      7. Produce final report
      8. Generate visualizations
      9. Save results to results/ folder
    Returns a dict with the report string, charts, and saved file paths.
    """
    dataset_name = getattr(file, "name", "dataset")
    dataset_name = os.path.basename(dataset_name)

    # Load
    df = load_dataset(file)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # Plan
    plan = create_plan(df)
    print(f"Pipeline steps: {plan['steps']}")
    if plan["skip"]:
        print(f"Skipped: {plan['skip']}")
    for note in plan.get("notes", []):
        print(f"  ! {note}")
    for hint in plan.get("focus", []):
        print(f"  -> {hint}")

    # EDA
    eda = eda_agent(df)
    print("EDA complete.")

    # Pattern detection (conditional)
    if "discover_clusters" in plan["steps"] or "detect_anomalies" in plan["steps"]:
        patterns = pattern_agent(df, eda=eda)
    else:
        patterns = {"clusters": None, "anomalies": None, "skipped": plan["skip"]}
    print("Pattern detection complete.")

    # Statistical tests
    anomaly_labels = patterns.get("anomalies", {}).get("labels", None)
    if "run_statistical_tests" in plan["steps"]:
        stats = stats_agent(df, anomaly_labels=anomaly_labels)
        print("Statistical tests complete.")
    else:
        stats = {}
        print("Statistical tests skipped.")

    # Insights
    insights = generate_insights(eda, patterns, stats, focus=plan["focus"])
    print("Insights generated.")

    # Report
    report = generate_report(insights, eda, patterns, stats)
    print("Report ready.")

    # Visualizations
    charts = generate_all_charts(df, eda, patterns)
    print(f"Charts generated: {list(charts.keys())}")

    # Save
    saved = save_results(dataset_name, eda, patterns, stats, report)
    charts_dir = os.path.join(saved["folder"], "charts")
    saved_charts = save_charts(charts, charts_dir)
    print(f"Charts saved to: {charts_dir}")

    return {
        "report":       report,
        "charts":       charts,
        "saved":        saved,
        "saved_charts": saved_charts,
        "plan":         plan,
    }