"""
Runs all applicable statistical tests on the df
"""

import pandas as pd
from analysis.statistics import run_all_tests


def stats_agent(df: pd.DataFrame) -> dict:
    return run_all_tests(df)