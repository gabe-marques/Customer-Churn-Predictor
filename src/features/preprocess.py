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
        target (str): Ignore target column
        drop_cols (list): Irrelevant columns
    
    Returns:
        (numeric_cols, categorical_cols) (tuple): Contains a list of the numerical columns and another list of the categorical columns
    """
    feature_cols = [c for c in df.columns if c != target and c not in drop_cols]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return numeric_cols, categorical_cols

def build_preprocessor(df: pd.DataFrame, target: str, cfg: FeatureConfig) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a ColumnTransformer that:
        - imputes and scales numeric columns 
        - imputes and one-hot encodes categorical columns 

    Returns:
        preprocessor, numeric_cols, categorical_cols
    """
    num_cols, cat_cols = infer_column_types(df, target, cfg.drop_cols)

    # Numeric pipeline: impute -> scale (optional)
    num_steps = [('imputer', SimpleImputer(strategy=cfg.numeric_impute))]
    if cfg.scaling == 'standard':
        num_steps.append(('scaler', StandardScaler()))
    numeric_pipe = Pipeline(steps=num_steps)

    # Categorical pipeline: impute -> onehot encoding (optional)
    cat_steps = [('imputer', SimpleImputer(strategy=cfg.categorical_impute))]
    if cfg.one_hot:
        cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))) # sparse_output=False gives a dense numpy array (easier to debug)
    categorical_pipe = Pipeline(steps=cat_steps)

    # Apply numeric and categorical pipelines to their respective columns
    # and combine the outputs into a single model-ready feature matrix
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, num_cols),
            ('cat', categorical_pipe, cat_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor, num_cols, cat_cols
