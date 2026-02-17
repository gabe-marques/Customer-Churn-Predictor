from __future__ import annotations

from pathlib import Path
import yaml
from typing import Any, Dict

class ConfigError(RuntimeError):
    """Raised when config is missing required keys or has wrong structure."""
    pass

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base configs.

    Example:
      base: {"model": {"name": "lr", "params": {"max_iter": 1000}}}
      override: {"model": {"name": "xgb"}}
      => {"model": {"name": "xgb", "params": {"max_iter": 1000}}}
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config

    Parameters:
        path (str or Path object): directory of .yaml configs
    
    Returns:
        cfg (Dict): Dictionary containing configs
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {} # If file is empty safe load returns none so convert that to an empty dictionary

    if not isinstance(cfg, dict):
        raise ConfigError("Config root must be a YAML mapping (dict).")

    inherit = cfg.pop("inherit", None) # Handle inheritance
    if inherit:
        base_path = (path.parent / inherit).resolve()
        base_cfg = load_config(base_path)  # recursion allows multi-level inherit
        cfg = _deep_merge(base_cfg, cfg) 

    return cfg

def require(cfg: Dict[str, Any], dotted_key: str):
    """
    Get a required config value using dotted notation, ex: 'run.seed'
    Prevents silent failures and gives clear error messages when config is wrong.
    """
    cur = cfg
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f'Missing required config key; {dotted_key}')
        cur = cur[part]
        
    return cur
    
