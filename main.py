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
      2. Run EDA
      3. Detect patterns (clusters + anomalies)
      4. Run statistical tests
      5. Generate LLM insights
      6. Produce final report
      7. Generate visualizations
      8. Save results to results/ folder
    Returns a dict with the report string, charts, and saved file paths.
    """
    plan = create_plan()
    print(f"Pipeline steps: {plan}")

    dataset_name = getattr(file, "name", "dataset")
    dataset_name = os.path.basename(dataset_name)

    df = load_dataset(file)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    eda = eda_agent(df)
    print("EDA complete.")

    patterns = pattern_agent(df)
    print("Pattern detection complete.")

    anomaly_labels = patterns.get("anomalies", {}).get("labels", None)
    stats = stats_agent(df, anomaly_labels=anomaly_labels)
    print("Statistical tests complete.")

    insights = generate_insights(eda, patterns, stats)
    print("Insights generated.")

    report = generate_report(insights, eda, patterns, stats)
    print("Report ready.")

    # Generate all charts
    charts = generate_all_charts(df, eda, patterns)
    print(f"Charts generated: {list(charts.keys())}")

    saved = save_results(dataset_name, eda, patterns, stats, report)

    # Save chart PNGs alongside the report
    charts_dir = os.path.join(saved["folder"], "charts")
    saved_charts = save_charts(charts, charts_dir)
    print(f"Charts saved to: {charts_dir}")

    return {
        "report":  report,
        "charts":  charts,   # figures available for Streamlit st.pyplot()
        "saved":   saved,
        "saved_charts": saved_charts,
    }