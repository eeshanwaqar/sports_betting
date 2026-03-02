"""
Train Model - Complete training pipeline with MLflow tracking.

Usage:
    python scripts/train.py                    # Full pipeline with MLflow
    python scripts/train.py --no-mlflow        # Train without MLflow tracking
    python scripts/train.py --config path.yaml # Custom config file

This script orchestrates:
    1. Load model_ready.csv (output of feature engineering)
    2. Time-based train/test split
    3. Train multiple models (LR, RF, GB, XGB)
    4. Evaluate with cross-validation
    5. Log everything to MLflow (params, metrics, artifacts)
    6. Register best model in MLflow Model Registry
    7. Save model artifacts to disk
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("scripts.train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="EPL Match Prediction - Model Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: auto-discover)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override path to model_ready.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()

    print("=" * 60)
    print("EPL BETTING PREDICTOR - MODEL TRAINING")
    print("=" * 60)

    # Load config
    config = load_config(args.config)
    logger.info(f"Config loaded: test_size={config.model.test_size}")

    # Determine data path
    data_path = args.data_path or f"{config.data.features}/model_ready.csv"
    if not Path(data_path).exists():
        logger.error(f"Features file not found: {data_path}")
        logger.error("Run the feature engineering pipeline first (notebook 03)")
        sys.exit(1)

    # Disable MLflow if requested
    if args.no_mlflow:
        config.mlflow.enabled = False
        logger.info("MLflow tracking disabled via --no-mlflow flag")

    # Choose pipeline based on MLflow availability
    if config.mlflow.enabled:
        logger.info("Running WITH MLflow tracking")
        from src.training.mlflow_trainer import MlflowTrainer

        mlflow_trainer = MlflowTrainer(config)
        trainer = mlflow_trainer.run(data_path)

        print("\n" + "-" * 60)
        print("MLflow Tracking")
        print("-" * 60)
        print(f"  Tracking URI : {config.mlflow.tracking_uri}")
        print(f"  Experiment   : {config.mlflow.experiment_name}")
        print(f"  Registry     : {config.mlflow.registry_name}")
        print("\n  To view the MLflow UI:")
        print(f"    mlflow ui --backend-store-uri {config.mlflow.tracking_uri}")
    else:
        logger.info("Running WITHOUT MLflow tracking")
        from src.training.trainer import run_pipeline

        trainer = run_pipeline(config)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Best model   : {trainer.best_name}")
    print(f"  Test accuracy: {trainer.best['test_acc']:.4f} "
          f"({trainer.best['test_acc']:.1%})")
    print(f"  Test F1 macro: {trainer.best['test_f1']:.4f}")

    print("\n  All models:")
    for name, result in sorted(
        trainer.results.items(),
        key=lambda x: x[1]["test_acc"],
        reverse=True,
    ):
        marker = " *" if name == trainer.best_name else ""
        print(
            f"    {name:25s} acc={result['test_acc']:.4f}  "
            f"f1={result['test_f1']:.4f}{marker}"
        )

    print(f"\n  Artifacts saved to: {config.model.models_dir}/")
    print("  You can now use scripts/predict.py to make predictions!")


if __name__ == "__main__":
    main()
