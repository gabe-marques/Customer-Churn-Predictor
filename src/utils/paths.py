from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ArtifactPaths:
    """
    Small helper to standardize where artifacts live.
    This will keep paths consistent across the whole pipeline.
    Pass in a root (e.g., 'artifacts) and it exposes subfolders.
    """

    root: Path

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    def ensure(self) -> "ArtifactPaths":
        """
        Ensure all artifact directories exist.
        Call this at program start so later code can assume folders exist.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        return self   