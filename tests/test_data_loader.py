"""
Tests for analysis/data_loader.py
Run with: pytest tests/test_data_loader.py -v
"""

import io
import pytest
from analysis.data_loader import load_dataset 

def test_csv_loading():
    # Simulate a CSV file in memory
    csv_data = io.StringIO("col1,col2\n1,2\n3,4")
    csv_data.name = "sample.csv"  # required for your function to detect CSV
    df = load_dataset(csv_data)
    
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]

def test_excel_loading(tmp_path):
    import pandas as pd
    # Create a temporary Excel file
    excel_path = tmp_path / "sample.xlsx"
    pd.DataFrame({"col1":[1,2], "col2":[3,4]}).to_excel(excel_path, index=False)
    
    with open(excel_path, "rb") as f:
        df = load_dataset(f)
    
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]

def test_unsupported_file():
    # Test that a non-csv/xlsx file raises ValueError
    with pytest.raises(ValueError):
        fake_file = io.StringIO("Hello world")
        fake_file.name = "sample.txt"
        load_dataset(fake_file)