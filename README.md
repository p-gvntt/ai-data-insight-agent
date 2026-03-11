# 🤖 AI Senior Data Analyst Agent

An end-to-end automated data analysis pipeline powered by OpenAI GPT. Upload any CSV and receive a structured, senior-level analytical report covering EDA, clustering, anomaly detection, statistical testing, actionable insights, and a full visual analysis suite — all generated automatically.

---

## ✨ Features

- **Automated EDA**: Shape, missing values, descriptive stats, distributions, outliers, correlations, and duplicates
- **Cluster Detection**: KMeans with automatic optimal k selection via silhouette scoring, with quality assessment (weak / moderate / strong)
- **Anomaly Detection**: Isolation Forest with median imputation for full index alignment
- **Statistical Testing**: t-test, Mann-Whitney U, one-way ANOVA, and chi-squared — all with outlier-excluded group means
- **LLM Insights**: GPT-4o-mini generates a structured 6-section report grounded strictly in the data
- **Domain-Aware Insights**: Pipeline infers the business domain from column names and tailors insights accordingly — historical datasets get analytical recommendations, business datasets get operational ones
- **Correlation Hallucination Prevention**: Raw correlation matrices stripped before LLM sees them — only pre-filtered notable pairs with plain-English descriptions are passed
- **Cluster Quality Flagging**: Weak silhouette scores and singleton clusters flagged explicitly in the report
- **Priority-Based Column Selection**: Business-relevant columns (revenue, order_value, salary, etc.) selected over high-CV noise columns
- **Clean Group Means**: Anomaly rows excluded before computing group means to prevent outlier contamination
- **Conditional Report Generation**: Next steps and recommendations adapt to actual data properties — no hardcoded boilerplate
- **Retry Logic**: Exponential backoff on LLM calls — transient OpenAI outages don't crash the pipeline
- **Visual Analysis Suite**: 8 charts generated automatically and displayed in a tabbed layout alongside the report
- **Chart-Report Consistency**: Charts are suppressed when the report omits them — no heatmap if no correlations exist, no cluster scatter if quality is too weak
- **Streamlit Web App**: Upload CSV, explore report and charts in side-by-side tabs, download results, browse past runs

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (`OPENAI_API_KEY`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-data-insight-agent.git
cd ai-data-insight-agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env:
# OPENAI_API_KEY=your_openai_api_key
```

### Running the App

```bash
streamlit run app/streamlit_app.py
```

Upload a CSV file and click **Run Analysis** to generate your report.

---

## 📁 Project Structure

```
ai-data-insight-agent/
├── app/
│   └── streamlit_app.py          # Streamlit web interface
├── analysis/
│   ├── data_loader.py            # CSV loading and validation
│   ├── eda.py                    # Full EDA pipeline
│   ├── clustering.py             # KMeans with quality assessment
│   ├── anomalies.py              # Isolation Forest detection
│   ├── statistics.py             # Statistical tests
│   └── visualization.py          # 8-chart visual analysis suite
├── agents/
│   ├── planner_agent.py          # Pipeline orchestration plan
│   ├── eda_agent.py              # EDA agent wrapper
│   ├── pattern_agent.py          # Clustering + anomaly agent
│   ├── stats_agent.py            # Statistical tests agent
│   ├── insight_agent.py          # LLM insight generation
│   └── report_agent.py           # Report formatting
├── data/                         # Place datasets here
├── docs/
│   ├── index.html               # Demo documentation page
│   └── media/                   # Demo videos and assets
├── utils/
│   └── results_saver.py          # Saves outputs to results/
├── tests/
│   ├── test_data_loader.py
│   ├── test_eda.py
│   ├── test_clustering.py
│   ├── test_anomalies.py
│   └── test_statistics.py
├── results/                      # Generated reports and charts saved here
├── main.py                       # Pipeline orchestrator
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 How It Works

1. **Load**: CSV uploaded and validated via `data_loader.py`
2. **EDA**: Full exploratory analysis — shape, missing values, distributions, outliers, correlations, categoricals, duplicates
3. **Pattern Detection**: KMeans clustering with silhouette-based k selection + Isolation Forest anomaly detection
4. **Statistical Tests**: t-test and Mann-Whitney U (binary split), ANOVA (3+ groups), chi-squared (categorical association) — all computed on anomaly-excluded clean data
5. **LLM Insights**: GPT-4o-mini receives stripped, pre-processed EDA results and generates a 6-section analytical report
6. **Visualizations**: 8 charts generated from the same pipeline outputs — suppressed conditionally to match the report
7. **Report**: Formatted report with metadata header, insights body, and conditional next steps saved to `results/`

---

## 📊 Report Structure

Every generated report contains six sections:

| Section | Content |
|---|---|
| **1. Dataset Overview** | Rows, columns, missing values, duplicates |
| **2. Key Patterns** | Clusters with quality assessment, notable correlations |
| **3. Anomalies** | Count, indices, potential causes |
| **4. Statistical Findings** | Test results with group means, significance, outlier exclusion disclosure |
| **5. Business Insights** | Domain-aware implications grounded strictly in the data |
| **6. Recommendations** | Concrete next steps — no boilerplate |

---

## 📈 Visual Analysis Suite

8 charts are generated automatically alongside every report, displayed in a **📋 Report / 📊 Visual Analysis** tab layout:

| Chart | Tab | What it shows |
|---|---|---|
| **Distributions** | EDA | Histogram + KDE per numeric column. Mean/median lines. ⚠ flags highly skewed columns |
| **Skewness** | EDA | Horizontal bar chart coloured by severity — green / orange / red |
| **Correlation Heatmap** | EDA | Pearson heatmap, lower triangle only. *Suppressed if no notable pairs exist* |
| **Pair Plot** | EDA | Scatter matrix for all numeric column pairs, capped at 5 columns |
| **Silhouette Score by k** | Clusters | Line chart showing why the optimal k was chosen, with quality thresholds |
| **Cluster Projection (PCA)** | Clusters | 2D PCA scatter of clusters with anomaly rows as red X. *Suppressed if quality weak* |
| **Cluster Sizes** | Clusters | Bar chart with ⚠ highlighting for singleton clusters. *Suppressed if quality weak* |
| **Anomaly Scatter** | Anomalies | Two highest-variance columns with anomaly rows overlaid. *Suppressed if no anomalies* |

Charts are saved as PNGs to `results/<run>/charts/` and are viewable in the past results browser.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

The test suite covers:

- All individual statistical test functions (t-test, Mann-Whitney, ANOVA, chi-squared)
- `run_all_tests` integration — column selection, group context, anomaly exclusion, ANOVA gating
- Priority-based column selection vs CV fallback
- Anomaly exclusion correctness — mean contamination prevention
- Edge cases — empty DataFrames, no categorical columns, binary-only categoricals

---

## ⚙️ Key Design Decisions

**Why strip correlation matrices before the LLM?**
Passing full Pearson/Spearman matrices causes the LLM to hallucinate correlations from arbitrary pairs. Only `notable_pairs` (|r| > 0.5, with pre-computed plain-English direction descriptions) are passed — the LLM reproduces these verbatim without interpretation.

**Why exclude anomalies before computing group means?**
A single outlier (e.g. `avg_order_value = 9999`) can shift a group mean by orders of magnitude and produce a misleading report. Isolation Forest labels are computed first, then anomaly rows are excluded before any group means are calculated. The exclusion count is always disclosed in the report.

**Why suppress charts conditionally?**
Charts mirror the report exactly. If the report doesn't mention correlations, the heatmap isn't shown — showing it would imply there's something to look at. If cluster quality is too weak to present segments, the PCA scatter is suppressed with an explanatory message. The silhouette score chart is always shown because it explains *why* the decision was made.

**Why gate ANOVA on 3+ groups?**
Running ANOVA on a binary categorical (e.g. gender) produces the same finding as the t-test. ANOVA is only run when a categorical with 3+ unique values exists, ensuring each test adds new information.

**Why infer the business domain?**
LLMs default to generic business language regardless of dataset context. A historical passenger manifest doesn't need marketing recommendations. Domain inference from column names lets the pipeline tailor insights appropriately — blocking operational business language for historical datasets while allowing it for e-commerce/CRM data.

---

## ⚠️ Known Limitations

- Designed for structured tabular CSV data — unstructured or time-series data may produce suboptimal results
- LLM output is constrained by prompt instructions but not programmatically validated — unusual datasets may still produce imperfect reports
- Column selection uses name-pattern matching — columns with non-standard names may fall back to the CV heuristic
- Domain inference is heuristic-based — ambiguous datasets fall back to generic factual language