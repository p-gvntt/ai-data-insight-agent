"""
Runs the full EDA pipeline on a df
"""

from analysis.eda import run_basic_eda
import pandas as pd


def eda_agent(df: pd.DataFrame) -> dict:
    return run_basic_eda(df)