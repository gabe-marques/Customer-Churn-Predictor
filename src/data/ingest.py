from __future__ import annotations

from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    This exists to keep file I/O separate from ML logic.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Raw data not found at: {path.resolve()}')
    
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f'Loaded DataFrame is emptyL {path.resolve()}')
    
    return df