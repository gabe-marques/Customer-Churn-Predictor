from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.preprocess import FeatureConfig, build_preprocessor
from src.models.registry import build_model

@dataclass(frozen=True)
class TrainArtifacts:
    """
    Paths to artifacts produced by training.
    """
    model_path: Path
    schema_path: Path

def _encode_target(y: pd.Series) -> pd.Series:
    """
    Normalize target into 0/1 integers.
    Telco dataset often uses Yes/No. Many models accept strings
    but standardizing avoids suprises and makes metrics easier.
    """
    if y.dtype == object:
        mapped = y.map({'Yes': 1, 'No': 0})
        if mapped.isna().any():
            # If mapping failed, show unique values for debugging
            raise ValueError(f'Unexpected target values: {y.dropna().unique().tolist()}')
       
        return mapped.astype('int64')
   
    return y.astype('int64')

def train_model(train_df: pd.DataFrame, target: str, feature_cfg: FeatureConfig, model_cfg: Dict[str, Any], artifacts_dir: str | Path) -> TrainArtifacts:
    """
    Train a model pipeline (preprocess + model) and save artifacts.

    Parameters:
        train_df (pd.DataFrame): training data including the target column
        target (str): name of target column
        feature_cfg (object): data preprocessing settings
        model_cfg (dict): {'name': '...', 'params': {...}}
        artifacts_dir (str or Path object): root artifacts directory

    Returns:
        TrainArtifacts:
            artifacts/models/churn_model.joblib
            artifacts/metrics/feature_schema.json
    """
    artifacts_dir = Path(artifacts_dir)
    (artifacts_dir/'models').mkdir(parents=True, exist_ok=True)
    (artifacts_dir/'metrics').mkdir(parents=True, exist_ok=True)

    # Build preprocessor from training data (learns categories, medians, etc.)
    preprocessor, num_cols, cat_cols = build_preprocessor(train_df, target, feature_cfg)

    # Build model from config
    model_name = str(model_cfg['name'])
    model_params = dict(model_cfg.get('params', {}))
    est = build_model(model_name, model_params)

    # Combine into a single pipeline
    clf = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', est)
    ])

    # Split X/y and fit classifier
    X_train = train_df.drop(columns=[target])
    y_train = _encode_target(train_df[target])
    clf.fit(X_train, y_train)

    # Save model pipeline
    model_path = artifacts_dir/'models'/'churn_model.joblib'
    joblib.dump(clf, model_path)

    # Save feature schema metadata for traceability and debugging
    schema = {
        'target': target,
        'drop_cols': feature_cfg.drop_cols,
        'numeric_cols': num_cols,
        'categorical_cols': cat_cols,
        'model_name': model_name,
        'model_params': model_params
    }
    schema_path = artifacts_dir/'metrics'/'feature_schema.json'
    with schema_path.open('w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2)
    
    return TrainArtifacts(model_path=model_path, schema_path=schema_path)