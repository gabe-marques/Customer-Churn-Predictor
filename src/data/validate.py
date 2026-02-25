from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import pandas as pd

@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    message: str

def validate_df(df: pd.DataFrame, target: str, required_cols: Optional[Iterable[str]]=None) -> ValidationReport:
    """
    Validate that the dataset is usable for training/evaluation.

    Parameters:
        df (pd.DataFrame): Input churn dataset (customer_churn.csv)
        target (str): The column that must exist before training/evaluation
        required_cols (Optional[Iterable[str]]): collection of columns that must exist before training/evaluation
    
    Returns:
        ValidationReport (object): Short message of the usability of the dataset
    """
    if df is None:
        return ValidationReport(False, 'DataFrame is None.')
    
    if df.empty:
        return ValidationReport(False, 'DataFrame is empty')
    
    if target not in df.columns:
        return ValidationReport(False, f'Target column "{target}" not found in columns: {list(df.columns)}')
    
    if required_cols is not None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return ValidationReport(False, f'Missing required columns: {missing}')
        
    # Basic target sanity check
    y = df[target].dropna().unique()
    if len(y) < 2:
        return ValidationReport(False, f"Target '{target}' has <2 unique non-null values: {y.tolist()}")

    return ValidationReport(True, "OK")