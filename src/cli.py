from __future__ import annotations

import argparse
import json
from pathlib import Path 
from typing import Any, Dict
import joblib 
import pandas as pd

from src.config import load_config, require
from src.utils.logging import LoggingConfig, setup_logging
from src.utils.paths import ArtifactPaths
from src.utils.seed import set_global_seed
from src.data.ingest import load_csv
from src.data.validate import validate_df
from src.data.split import split_and_save
from src.features.preprocess import FeatureConfig
from src.models.train import train_model
from src.models.evaluate import evaluate_model

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
def _feature_cfg_from_cfg(cfg: Dict[str,Any]) -> FeatureConfig:
    f = cfg['features']
    return FeatureConfig(
        drop_cols=f["drop_cols"],
        numeric_impute=f.get("numeric_impute", "median"),
        categorical_impute=f.get("categorical_impute", "most_frequent"),
        scaling=f.get("scaling", "standard"),
        one_hot=bool(f.get("one_hot", True)),
    )

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

    # Load and validate data
    raw_path = require(cfg, 'data.raw_path')
    target = require(cfg, 'data.target')
    test_size = float(require(cfg, 'data.test_size'))

    df = load_csv(raw_path)
    report = validate_df(df, target)
    if not report.ok:
        raise RuntimeError(f'Data validation failed: {report.message}')
    logger.info(f'Data validation {report.message}')
    logger.info(f'Loaded data shape: {df.shape}')

    # Deterministic split (saved to artifacts/metrics/split_indices.json)
    split = split_and_save(df, test_size=test_size, seed=seed, artifacts_dir=paths.root)
    logger.info(f'Train shape: {split.train_df.shape} | Test shape: {split.test_df.shape}')

    # Command execution
    feature_cfg = _feature_cfg_from_cfg(cfg)
    model_path = paths.root/'models'/'churn_model.joblib'

    if args.command == "train":
        train_artifact = train_model(
            train_df=split.train_df,
            target=target,
            feature_cfg=feature_cfg,
            model_cfg=cfg["model"],
            artifacts_dir=paths.root
        )
        logger.info(f'Saved model: {train_artifact.model_path.resolve()}')
        logger.info(f'Saved feature schema: {train_artifact.schema_path.resolve()}')

        # Evaluate immediately after training 
        eval_artifact = evaluate_model(
            model_path=train_artifact.model_path,
            test_df=split.test_df,
            target=target,
            artifacts_dir=paths.root,
            threshold=0.5
        )
        logger.info(f'Saved metrics: {eval_artifact.metrics_path.resolve()}')
        logger.info(f'Saved ROC plot: {eval_artifact.roc_path.resolve()}')
        logger.info(f'Saved confusion matrix plot: {eval_artifact.cm_path.resolve()}')
        return 0
    
    if args.command == "eval":
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found: {model_path.resolve()} (run train first)')
        
        eval_artifact = evaluate_model(
            model_path=model_path,
            test_df=split.test_df,
            target=target,
            artifacts_dir=paths.root,
            threshold=0.5
        )
        logger.info(f'Saved metrics: {eval_artifact.metrics_path.resolve()}')
        logger.info(f'Saved ROC plot: {eval_artifact.roc_path.resolve()}')
        logger.info(f'Saved confusion matrix plot: {eval_artifact.cm_path.resolve()}')
        return 0
    
    if args.command == "predict":
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found: {model_path.resolve()} (run train first)')
        
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        
        clf = joblib.load(model_path)
        X_new = pd.read_csv(input_path)
        if target in X_new.columns:
            X_new = X_new.drop(columns=[target])
        
        # Generate probabilities and predictions
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(X_new)[:,1]
        else:
            prob = clf.predict(X_new)

        out_path = paths.root/'metrics'/'predictions.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'churn_probability': prob}).to_csv(out_path, index=False)
        logger.info(f'Saved predictions: {out_path.resolve()}')
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())