"""
Uses an LLM to generate senior-level data insights from analysis results.
"""
import json
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def _safe_serialize(obj) -> str:
    """Serialize analysis results to a clean JSON string for the prompt."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


def _strip_correlation_matrices(eda: dict) -> dict:
    """
    Remove raw pearson and spearman matrices before passing EDA to the LLM.
    Only notable_pairs (pre-filtered, with plain_english descriptions) should
    be visible to the LLM to prevent hallucinated correlations.
    """
    import copy
    eda = copy.deepcopy(eda)
    correlations = eda.get("correlations", {})
    correlations.pop("pearson", None)
    correlations.pop("spearman", None)
    return eda


def _build_data_context(eda: dict, patterns: dict) -> str:
    """
    Derive factual context flags from the EDA and patterns so the prompt
    gives the LLM conditional instructions grounded in actual data properties.
    """
    missing_pct = eda.get("missing", {}).get("overall_missing_pct", 0.0)
    has_missing = missing_pct > 0

    duplicate_rows = eda.get("duplicates", {}).get("duplicate_rows", 0)
    has_duplicates = duplicate_rows > 0

    notable_pairs = eda.get("correlations", {}).get("notable_pairs", [])
    n_notable = len(notable_pairs)

    # FIX: use pre-computed cluster quality fields from clustering.py
    cluster_info = patterns.get("clusters", {})
    quality_note = cluster_info.get("cluster_quality_note", None)
    singleton_clusters = cluster_info.get("singleton_clusters", [])

    context_lines = []

    if not has_missing:
        context_lines.append("- This dataset has NO missing values. Do NOT recommend addressing missing data.")
    else:
        context_lines.append(f"- This dataset has {missing_pct}% missing values across some columns.")

    if not has_duplicates:
        context_lines.append("- This dataset has NO duplicate rows. Do NOT recommend deduplication.")
    else:
        context_lines.append(f"- This dataset has {duplicate_rows} duplicate rows that should be addressed.")

    if n_notable == 0:
        context_lines.append(
            "- No strong correlations (|r| > 0.5) were found. Do NOT report any correlations."
        )
    else:
        context_lines.append(
            f"- There are {n_notable} notable correlation(s). "
            "ONLY discuss pairs listed in notable_pairs — do NOT reference the full pearson or spearman matrices."
        )

    # Use pre-computed quality note from clustering.py
    if quality_note:
        context_lines.append(f"- CLUSTER QUALITY: {quality_note}")

    if singleton_clusters:
        context_lines.append(
            f"- The following clusters contain 2 or fewer members and are NOT meaningful segments: "
            f"{', '.join(singleton_clusters)}. Flag these as outlier groups, not customer segments."
        )

    return "\n".join(context_lines)


def _call_llm_with_retry(prompt: str, max_retries: int = 3, backoff: float = 2.0) -> str:
    """
    FIX: Wrap LLM call with retry logic and error handling so a transient
    OpenAI outage doesn't crash the entire pipeline.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts. Last error: {last_error}"
    )


def _infer_domain(eda: dict) -> str:
    """
    Infer the likely business domain from column names and sample values
    so the LLM can anchor insights to the actual context of the data
    rather than inventing generic business advice.
    """
    columns = []
    for col, info in eda.get("columns", {}).items():
        columns.append(col.lower())

    col_str = ", ".join(columns)

    # Simple heuristic hints passed to the LLM as context
    hints = []
    if any(w in col_str for w in ["fare", "ticket", "cabin", "embarked", "pclass", "survived"]):
        hints.append("historical passenger transport (likely Titanic or similar voyage dataset)")
    elif any(w in col_str for w in ["order", "purchase", "churn", "salary", "region", "customer"]):
        hints.append("e-commerce or CRM customer dataset")
    elif any(w in col_str for w in ["patient", "diagnosis", "treatment", "hospital", "disease"]):
        hints.append("healthcare or medical dataset")
    elif any(w in col_str for w in ["student", "grade", "score", "school", "exam"]):
        hints.append("education dataset")
    elif any(w in col_str for w in ["price", "stock", "market", "revenue", "profit"]):
        hints.append("financial or sales dataset")

    if hints:
        domain = hints[0]
        is_historical = "historical" in domain or "titanic" in domain.lower() or "voyage" in domain.lower()
        if is_historical:
            return (
                f"This appears to be a {domain}. "
                "This is a HISTORICAL RECORD, not a live business system. "
                "There are NO customers, NO marketing campaigns, NO pricing strategies, "
                "NO service enhancements, NO customer segmentation. "
                "These words must NOT appear in your report. "
                "People in this dataset are passengers, subjects, or records — not customers. "
                "Business insights must focus ONLY on what patterns the historical data reveals "
                "and what further statistical or analytical investigation would be valuable. "
                "Recommendations must be purely analytical: further modelling, deeper statistical "
                "testing, or data quality improvements. Nothing operational."
            )
        else:
            return (
                f"Based on the column names, this appears to be a {domain}. "
                "Tailor all business insights and recommendations to this specific domain. "
                "Do NOT suggest actions that are irrelevant to this domain."
            )
    else:
        return (
            "The business domain is unclear from the column names alone. "
            "Keep business insights strictly factual and grounded in the data. "
            "Do NOT invent a business context or suggest domain-specific actions "
            "unless they are directly supported by the findings."
        )


def generate_insights(eda: dict, patterns: dict, stats: dict) -> str:
    """
    Generate structured insights from EDA, pattern detection,
    and statistical test results using an LLM.
    """
    data_context = _build_data_context(eda, patterns)
    domain_context = _infer_domain(eda)
    eda_for_llm = _strip_correlation_matrices(eda)

    prompt = f"""
You are a senior data analyst with 15+ years of experience.
Below is the output of an automated data analysis pipeline.
Use ONLY the data provided — do NOT invent numbers, percentages, or trends.

DOMAIN CONTEXT (inferred from the data — anchor all insights to this):
{domain_context}

DATA CONTEXT (read before writing — these are hard facts about this specific dataset):
{data_context}

═══════════════════════════════════════
EDA RESULTS:
{_safe_serialize(eda_for_llm)}

PATTERN DETECTION (Clusters & Anomalies):
{_safe_serialize(patterns)}

STATISTICAL TESTS:
{_safe_serialize(stats)}
═══════════════════════════════════════

Your task: Write a structured analytical report with the following sections:

1. DATASET OVERVIEW
   - Size, data quality, missing values, duplicates
   - Only mention missing values or duplicates as issues if they actually exist per the DATA CONTEXT above

2. KEY PATTERNS
   - Clusters found, what they may represent
   - Always include the silhouette score and its quality assessment from DATA CONTEXT.
     If DATA CONTEXT flags clusters as weak or having singleton groups, state this
     explicitly — do NOT present poor clusters as meaningful segments.
   - Notable correlations between variables
   - ONLY mention correlations listed in notable_pairs in the EDA RESULTS above.
     Do NOT reference any values from the pearson or spearman matrices directly.
     If the DATA CONTEXT says no strong correlations were found, state that explicitly
     and do not report any correlations at all.
   - For each pair in notable_pairs, reproduce the plain_english field exactly as written,
     as a plain sentence — do NOT wrap it in quotes or treat it as a quotation.
     Do NOT add any further interpretation, explanation, or context beyond the plain_english
     text itself. Do NOT explain what the correlation might mean or imply.

3. ANOMALIES
   - How many anomalies detected, which rows, potential causes

4. STATISTICAL FINDINGS
   - Interpret each test result in plain English, naming the exact columns or groups compared
   - State whether differences are statistically significant and what that means practically
   - For t-tests and Mann-Whitney tests, always use group_a_mean and group_b_mean to state
     which group is higher and by how much. Do NOT rely on the sign of the statistic to infer
     direction — always derive direction from the means directly.
   - If t-test and Mann-Whitney were run on the same column and groups, treat them as
     one finding. State the result once, then note that both parametric and non-parametric
     tests agree. Do NOT present them as two separate discoveries.
   - If outliers_excluded > 0, note that group means were computed after excluding
     that many anomaly rows to avoid contamination. Always mention this transparently.

5. BUSINESS INSIGHTS
   - What do these findings mean in the context of the DOMAIN CONTEXT above?
   - Stay strictly within the inferred domain. If the domain is historical or ambiguous,
     do NOT invent modern business recommendations (e.g. marketing campaigns, service
     enhancements, product features). Instead, state what the data shows and what
     further analysis would be warranted.
   - Only make recommendations that are actionable given the actual data available.
     Do NOT suggest collecting data that is already present, or fixing issues that do not exist.
   - Do NOT extrapolate or interpret correlations beyond what the plain_english field states.
     Do NOT infer causes, strategies, or explanations for a correlation unless explicitly
     stated in the data provided.
   - Do NOT speculate about causes, social behaviours, historical context, or any explanation
     for statistical findings unless that explanation is explicitly present in the data.

6. RECOMMENDATIONS
   - Concrete next steps grounded strictly in the findings above
   - Stay within the DOMAIN CONTEXT — do NOT suggest actions that belong to a
     different business domain than the one inferred
   - If the domain is historical or ambiguous, focus recommendations on further
     analysis (e.g. deeper segmentation, predictive modelling) rather than
     operational business actions
   - Do NOT include boilerplate recommendations that contradict the DATA CONTEXT
     (e.g. do not recommend fixing missing values if there are none)

Write clearly and concisely. Avoid jargon. Back every claim with the data above.
Do NOT invent or extrapolate beyond what the data shows.
"""
    return _call_llm_with_retry(prompt)