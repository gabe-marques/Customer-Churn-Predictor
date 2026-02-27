from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

@dataclass(frozen=True)
class EvalArtifacts:
    metrics_path: Path
    roc_path: Path
    cm_path: Path

def _encode_target(y: pd.Series) -> pd.Series:
    """
    Keep target encoding consistent with training
    """
    if y.dtype == object:
        mapped = y.map({'Yes': 1, 'No': 0})
        if mapped.isna().any():
            # If mapping failed, show unique values for debugging
            raise ValueError(f'Unexpected target values: {y.dropna().unique().tolist()}')
       
        return mapped.astype('int64')
   
    return y.astype('int64')

def evaluate_model(model_path: str|Path, test_df: pd.DataFrame, target: str, artifacts_dir: str|Path, threshold: float=0.5) -> EvalArtifacts:
    """
    Evaluate a trained model pipeline on a test set and save metrics + plots
    """
    model_path = Path(model_path)
    artifacts_dir = Path(artifacts_dir)
    (artifacts_dir/'metrics').mkdir(parents=True, exist_ok=True)
    (artifacts_dir/'plots').mkdir(parents=True, exist_ok=True)

    clf = joblib.load(model_path)
    X_test = test_df.drop(columns=[target])
    y_true = _encode_target(test_df[target])

    # Probabilities for positive class
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test)[:,1]
    else:
        # Fallback for models without predict_proba
        scores = clf.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-scores)) # Converts score sigmoid ("probability-like")
    
    # Convert probabilities into predictions
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
    }

    metrics_path = artifacts_dir/'metrics'/'metrics.json'
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # ROC Curve plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_path = artifacts_dir/'plots'/'roc_curve.png'
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # Confusion Matrix plot
    cm = confusion_matrix(y_true, y_pred)
    cm_path = artifacts_dir/'plots'/'confusion_matrix.png'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return EvalArtifacts(metrics_path=metrics_path, roc_path=roc_path, cm_path=cm_path)