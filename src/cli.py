from __future__ import annotations

import argparse
import json
from pathlib import Path 
from typing import Any, Dict

from src.config import load_config, require
from src.utils.logging import LoggingConfig, setup_logging
from src.utils.paths import ArtifactPaths
from src.utils.seed import set_global_seed

def _write_run_config(cfg: Dict[str, Any], artifacts_root: Path):
    """
    Save the exact config used for a run.
    """
    out_path = artifacts_root / "metrics" / "run_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    
def build_parser() -> argparse.ArgumentParser:
    """
    Define CLI commands: train, eval, predict
    Require config and input arguments
    """
    p = argparse.ArgumentParser(
        prog="Customer-Churn-Predictor",
        description="Config-driven churn ML pipeline.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a model and write artifacts.")
    p_train.add_argument("--config", required=True, type=str, help="Path to YAML config.")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate an existing model.")
    p_eval.add_argument("--config", required=True, type=str, help="Path to YAML config.")

    # predict
    p_pred = sub.add_parser("predict", help="Run inference on a CSV input.")
    p_pred.add_argument("--config", required=True, type=str, help="Path to YAML config.")
    p_pred.add_argument("--input", required=True, type=str, help="Path to input CSV.")

    return p

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    # Required run settings
    artifacts_dir = Path(require(cfg, "run.artifacts_dir"))
    seed = int(require(cfg, "run.seed"))
    log_level = str(require(cfg, "run.log_level"))

    # Ensure output folders exist
    paths = ArtifactPaths(artifacts_dir).ensure()

    # Configure logging early so all later steps can log
    logger = setup_logging(
        LoggingConfig(level=log_level, log_to_file=True, log_file="run.log"),
        artifacts_dir=paths.root,
    )

    # Reproducibility
    set_global_seed(seed)

    # Save config used for the run (traceability)
    _write_run_config(cfg, paths.root)

    logger.info("Command: %s", args.command)
    logger.info("Config: %s", Path(args.config).resolve())
    logger.info("Artifacts dir: %s", paths.root.resolve())
    logger.info("Seed: %d", seed)

    # Placeholders to wire in later
    if args.command == "train":
        logger.info("TODO: call training pipeline here (src/models/train.py)")
        logger.info("Expected outputs: artifacts/models/*.joblib + artifacts/metrics/metrics.json + plots/")
        return 0

    if args.command == "eval":
        logger.info("TODO: call evaluation pipeline here (src/models/evaluate.py)")
        return 0

    if args.command == "predict":
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        logger.info("TODO: load model + preprocess + predict on %s", input_path.resolve())
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())