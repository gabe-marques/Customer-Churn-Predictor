from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass(frozen=True)
class SplitResult:
    """
    Container for train/test DataFrames and the row indices used to create them.
    """
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_idx: np.ndarray
    test_idx: np.ndarray

def split_and_save(df: pd.DataFrame, test_size: float, seed: int, artifacts_dir: str | Path) -> SplitResult:
    """
    Deterministically split df into train/test and persist the indices. Save the indices to re-run same split later.

    Parameters:
        df (pd.DataFrame): Input churn dataset (customer_churn.csv)
        test_size (float): Percentage of dataset that is allocated to test_df
        seed (int): number used to initialize pseudorandom number
        artifacts_dir (str or Path object): path to artifacts folder
    Returns:
        SplitResult (object)
    """
    artifacts_dir = Path(artifacts_dir)

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)

    # Save split indices as an artifact
    out_path = artifacts_dir/'metrics'/'split_indices.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(
            {
                "seed": seed,
                "test_size": test_size,
                "n_rows": int(len(df)),
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
            },
            f,
            indent=2,
        )
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return SplitResult(train_df=train_df, test_df=test_df, train_idx=train_idx, test_idx=test_idx)