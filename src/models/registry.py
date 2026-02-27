from __future__ import annotations

from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def build_model(name: str, params: Dict[str, Any]):
    """
    Factory function that builds a model from config.

    Parameters:
        name (str): name of model, loaded from cfg[model][name]
        params (dict): list of parameters for model, loaded from cfg[model][params]
    """
    name = name.lower()

    if name == 'logistic_regression':
        return LogisticRegression(**params)
    elif name == 'random_forest':
        return RandomForestClassifier(**params)
    elif name == 'xgboost':
        return XGBClassifier(**params)
    else:
        raise ValueError(f'Unkown model name: {name}')