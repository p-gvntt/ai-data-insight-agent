"""
Streamlit UI for the AI Data Analyst Agent.
Run with: streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from main import run_analysis

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

st.set_page_config(
    page_title="AI Senior Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Senior Data Analyst Agent")
st.markdown(
    "Upload a dataset or pick one from the `data/` folder. "
    "The pipeline will run EDA, detect patterns & anomalies, run statistical "
    "tests, and generate senior-level insights."
)

# Dataset source selection

source = st.radio("Dataset source", ["Upload a file", "Pick from data/ folder"], horizontal=True)

file = None

if source == "Upload a file":
    file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

else:
    available = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") or f.endswith(".xlsx")
    ] if os.path.exists(DATA_DIR) else []

    if not available:
        st.warning("No datasets found in `data/`. Add .csv or .xlsx files there.")
    else:
        selected = st.selectbox("Choose a dataset", available)
        if selected:
            file = open(os.path.join(DATA_DIR, selected), "rb")
            file.name = selected

# Run analysis

if file:
    if st.button("▶ Run Analysis"):
        with st.spinner("Running full analysis pipeline..."):
            try:
                result = run_analysis(file)
                report = result["report"]
                saved = result["saved"]

                st.success("Analysis complete!")

                # Report display + download
                st.markdown(report)
                st.download_button(
                    label="⬇️ Download Report (.txt)",
                    data=report,
                    file_name=os.path.basename(saved["report_txt"]),
                    mime="text/plain",
                )

                # Show where results were saved
                st.info(
                    f"**Results saved!**\n\n"
                )

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

# Past results browser

st.divider()
st.subheader("📁 Past Results")

if os.path.exists(RESULTS_DIR):
    past_runs = sorted(
        [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))],
        reverse=True,
    )
    if past_runs:
        selected_run = st.selectbox("Select a past run to view", past_runs)
        run_path = os.path.join(RESULTS_DIR, selected_run)
        report_path = os.path.join(run_path, "report.txt")
        json_path = os.path.join(run_path, "analysis.json")

        if os.path.exists(report_path):
            with open(report_path) as f:
                past_report = f.read()
            st.markdown(past_report)
            st.download_button(
                label="⬇️ Download Past Report",
                data=past_report,
                file_name=f"{selected_run}_report.txt",
                mime="text/plain",
            )

        if os.path.exists(json_path):
            with open(json_path) as f:
                import json
                past_json = json.load(f)
            with st.expander("View raw analysis.json"):
                st.json(past_json)
    else:
        st.caption("No past runs yet.")
else:
    st.caption("No past runs yet.")