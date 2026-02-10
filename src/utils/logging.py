from __future__ import annotations
import logging 
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "run.log"    

def setup_logging(cfg: LoggingConfig, artifacts_dir: Path) -> logging.Logger:
    """
    Configure a console logger + optional file logger to artifacts_dir/run.log
    Supports observability: when something breaks, the logs show what happened and
    with what config.
    """
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    logger = logging.getLogger("churn")
    logger.setLevel(level)
    logger.propagate = False  # prevents double logging

    # Prevent duplicate handlers in notebooks
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if cfg.log_to_file:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fh_path = artifacts_dir / cfg.log_file
        fh = logging.FileHandler(fh_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger