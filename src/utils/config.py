from __future__ import annotations

from pathlib import Path
import yaml

class ConfigError(RuntimeError):
    """Raised when config is missing required keys or has wrong structure."""
    pass

