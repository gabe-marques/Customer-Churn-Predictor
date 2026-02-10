from __future__ import annotations
import os
import random
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SeedConfig:
    seed: int

def set_global_seed(seed: int) -> None:
    """
    Makes runs more reproducible.
    
    Parameters:
        seed (int): number used to initialize pseudorandom number
        generator of Python's random module and Numpy RNG.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)