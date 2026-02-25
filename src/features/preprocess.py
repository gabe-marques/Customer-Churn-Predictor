from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for preprocessing. Comes from configs/base.yaml under `features:`
    """
    drop_cols: list[str]
    numeric_impute: str = "median"
    categorical_impute: str = "most_frequent"
    scaling: str = "standard"
    one_hot: bool = True

def infer_column_types(df: pd.DataFrame, target: str, drop_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Infer numeric vs categorical columns from a dataframe. 
    
    Parameters:
        df (pd.DataFrame): Input churn dataset (customer_churn.csv)
        target (str): Ignore
        drop_cols (list): Irrelevant columns
    
    Returns:
        (numeric_cols, categorical_cols) (tuple): Contains a list of the numerical columns and another list of the categorical columns
    """
    feature_cols = [c for c in df.columns if c != target and c not in drop_cols]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return numeric_cols, categorical_cols

def build_processor(df: pd.DataFrame, target: str, cfg: FeatureConfig) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a ColumnTransformer that:
        - imputes and scales numeric columns 
        - imputes and one-hot encodes categorical columns (optional)

    Returns:
        preprocessor, numeric_cols, categorical_cols
    """
    return 'TODO: build_processor'
