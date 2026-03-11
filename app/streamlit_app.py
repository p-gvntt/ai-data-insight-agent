"""
Streamlit UI for the AI Data Analyst Agent.
Run with: streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import streamlit as st
from main import run_analysis

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
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


# Suppression messages for charts that were conditionally skipped
_SUPPRESSION_MESSAGES = {
    "correlation_heatmap": "ℹ️ No strong correlations (|r| > 0.5) found — heatmap not shown.",
    "cluster_scatter":     "ℹ️ Cluster quality too weak to show meaningful segments (silhouette < 0.25).",
    "cluster_sizes":       "ℹ️ Cluster quality too weak to show meaningful segments (silhouette < 0.25).",
    "anomaly_scatter":     "ℹ️ No anomalies detected — anomaly plot not shown.",
}

def _format_report(report: str) -> str:
    """
    Convert ALL-CAPS text to Title Case for cleaner Streamlit rendering.
    Handles both section headers (## 1. DATASET OVERVIEW) and bold
    recommendation labels (**TARGETED MARKETING**: ...).
    Leaves code blocks and the metadata header untouched.
    """
    import re
    lines = report.split("\n")
    result = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        if in_code_block:
            result.append(line)
            continue
        line = re.sub(
            r'(#{1,3}\s*\d*\.?\s*)([A-Z][A-Z ]+)',
            lambda m: m.group(1) + m.group(2).title(),
            line
        )
        line = re.sub(
            r'\*\*([A-Z][A-Z ]{2,})\*\*',
            lambda m: "**" + m.group(1).title() + "**",
            line
        )
        if len(line.strip()) > 60 and re.match(r'^\*\*[^*].+[^*]\*\*\.?$', line.strip()):
            line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        result.append(line)
    return "\n".join(result)


def _suppressed_note(name: str, suppressed: list) -> bool:
    if name in suppressed:
        st.info(_SUPPRESSION_MESSAGES.get(name, f"ℹ️ {name} not available for this dataset."))
        return True
    return False


def render_charts(charts: dict):
    """Render all charts in a tabbed layout, with info messages for suppressed ones."""
    suppressed = charts.get("_suppressed", [])

    tab_eda, tab_clusters, tab_anomalies = st.tabs([
        "📈 EDA", "🔵 Clusters", "🔴 Anomalies"
    ])

    # EDA Tab
    with tab_eda:
        if "distributions" in charts:
            st.subheader("Distributions")
            st.caption(
                "Histogram + KDE per numeric column. "
                "Red dashed = mean, green dotted = median. ⚠ flags highly skewed columns."
            )
            st.pyplot(charts["distributions"])

        if "skewness" in charts:
            st.subheader("Skewness")
            st.caption("Green = symmetric, orange = moderate skew, red = highly skewed (|skew| > 1).")
            st.pyplot(charts["skewness"])

        if not _suppressed_note("correlation_heatmap", suppressed):
            if "correlation_heatmap" in charts:
                st.subheader("Correlation Heatmap")
                st.caption(
                    "Pearson correlations for pairs with |r| > 0.5. "
                    "Blue = negative, red = positive. Lower triangle only."
                )
                st.pyplot(charts["correlation_heatmap"])

        if "pair_plot" in charts:
            st.subheader("Pair Plot")
            st.caption("Scatter plots between numeric column pairs. Diagonal = KDE. Capped at 5 columns.")
            st.pyplot(charts["pair_plot"])

    # Clusters Tab
    with tab_clusters:
        if "silhouette_scores" in charts:
            st.subheader("Silhouette Score by k")
            st.caption(
                "Green line = strong threshold (0.5). Orange = weak threshold (0.25). "
                "Red dashed = chosen k."
            )
            st.pyplot(charts["silhouette_scores"])

        col1, col2 = st.columns(2)
        with col1:
            if not _suppressed_note("cluster_scatter", suppressed):
                if "cluster_scatter" in charts:
                    st.subheader("Cluster Projection (PCA)")
                    st.caption("2D PCA projection of all clusters. Red X = anomaly rows.")
                    st.pyplot(charts["cluster_scatter"])
        with col2:
            if not _suppressed_note("cluster_sizes", suppressed):
                if "cluster_sizes" in charts:
                    st.subheader("Cluster Sizes")
                    st.caption("⚠ red bars = singleton / near-singleton clusters (≤ 2 members).")
                    st.pyplot(charts["cluster_sizes"])

    # Anomalies Tab
    with tab_anomalies:
        if not _suppressed_note("anomaly_scatter", suppressed):
            if "anomaly_scatter" in charts:
                st.subheader("Anomaly Scatter")
                st.caption("Auto-selects two highest-variance columns. Red X = flagged anomaly rows.")
                st.pyplot(charts["anomaly_scatter"])


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
            class NamedFile:
                """Wraps a BufferedReader to add a writable name attribute,
                matching the interface of a Streamlit UploadedFile."""
                def __init__(self, path, name):
                    self._f = open(path, "rb")
                    self.name = name
                def read(self, *args): return self._f.read(*args)
                def seek(self, *args): return self._f.seek(*args)
                def tell(self): return self._f.tell()
                def __enter__(self): return self
                def __exit__(self, *args): self._f.close()

            file = NamedFile(os.path.join(DATA_DIR, selected), selected)


# Run analysis
if file:
    if st.button("▶ Run Analysis"):
        with st.spinner("Running full analysis pipeline..."):
            try:
                result = run_analysis(file)
                report = result["report"]
                saved  = result["saved"]
                charts = result.get("charts", {})

                st.success("Analysis complete!")
                st.info(f"**Results saved in the folder!**")

                # Top-level tabs: Report | Visual Analysis
                tab_report, tab_charts = st.tabs(["📋 Report", "📊 Visual Analysis"])

                with tab_report:
                    st.markdown(_format_report(report))
                    st.download_button(
                        label="⬇️ Download Report (.txt)",
                        data=report,
                        file_name=os.path.basename(saved["report_txt"]),
                        mime="text/plain",
                    )

                with tab_charts:
                    if charts:
                        render_charts(charts)
                    else:
                        st.info("No charts available for this run.")

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
        run_path    = os.path.join(RESULTS_DIR, selected_run)
        report_path = os.path.join(run_path, "report.txt")
        json_path   = os.path.join(run_path, "analysis.json")
        charts_dir  = os.path.join(run_path, "charts")

        # Past run tabs: Report | Charts
        past_tab_report, past_tab_charts = st.tabs(["📋 Report", "📊 Charts"])

        with past_tab_report:
            if os.path.exists(report_path):
                with open(report_path) as f:
                    past_report = f.read()
                st.markdown(_format_report(past_report))
                st.download_button(
                    label="⬇️ Download Past Report",
                    data=past_report,
                    file_name=f"{selected_run}_report.txt",
                    mime="text/plain",
                )
            if os.path.exists(json_path):
                with open(json_path) as f:
                    past_json = json.load(f)
                with st.expander("View raw analysis.json"):
                    st.json(past_json)

        with past_tab_charts:
            if os.path.exists(charts_dir):
                chart_files = sorted([
                    f for f in os.listdir(charts_dir) if f.endswith(".png")
                ])
                if chart_files:
                    tab_eda, tab_clusters, tab_anomalies = st.tabs([
                        "📈 EDA", "🔵 Clusters", "🔴 Anomalies"
                    ])
                    eda_charts     = ["distributions", "skewness", "correlation_heatmap", "pair_plot"]
                    cluster_charts = ["silhouette_scores", "cluster_scatter", "cluster_sizes"]
                    anomaly_charts = ["anomaly_scatter"]

                    with tab_eda:
                        for f in chart_files:
                            name = f.replace(".png", "")
                            if name in eda_charts:
                                st.image(os.path.join(charts_dir, f), caption=name, use_container_width=True)

                    with tab_clusters:
                        for f in chart_files:
                            name = f.replace(".png", "")
                            if name in cluster_charts:
                                st.image(os.path.join(charts_dir, f), caption=name, use_container_width=True)

                    with tab_anomalies:
                        for f in chart_files:
                            name = f.replace(".png", "")
                            if name in anomaly_charts:
                                st.image(os.path.join(charts_dir, f), caption=name, use_container_width=True)
                else:
                    st.info("No saved charts found for this run.")
            else:
                st.info("No charts folder found for this run.")
    else:
        st.caption("No past runs yet.")
else:
    st.caption("No past runs yet.")