"""
Uses an LLM to generate senior-level data insights from analysis results.
"""

import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def _safe_serialize(obj) -> str:
    """Serialize analysis results to a clean JSON string for the prompt."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


def generate_insights(eda: dict, patterns: dict, stats: dict) -> str:
    """
    Generate structured insights from EDA, pattern detection,
    and statistical test results using an LLM.
    """
    prompt = f"""
You are a senior data analyst with 15+ years of experience.

Below is the output of an automated data analysis pipeline.
Use ONLY the data provided — do NOT invent numbers, percentages, or trends.

═══════════════════════════════════════
EDA RESULTS:
{_safe_serialize(eda)}

PATTERN DETECTION (Clusters & Anomalies):
{_safe_serialize(patterns)}

STATISTICAL TESTS:
{_safe_serialize(stats)}
═══════════════════════════════════════

Your task: Write a structured analytical report with the following sections:

1. DATASET OVERVIEW
   - Size, data quality, missing values, duplicates

2. KEY PATTERNS
   - Clusters found, what they may represent
   - Notable correlations between variables

3. ANOMALIES
   - How many anomalies detected, which rows, potential causes

4. STATISTICAL FINDINGS
   - Interpret each test result in plain English
   - State whether differences are statistically significant

5. BUSINESS INSIGHTS
   - What do these findings mean in a real-world context?

6. RECOMMENDATIONS
   - Concrete next steps for the data team or business stakeholders

Write clearly and concisely. Avoid jargon. Back every claim with the data above.
Do NOT invent or extrapolate beyond what the data shows.
"""
    response = llm([HumanMessage(content=prompt)])
    return response.content