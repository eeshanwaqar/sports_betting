"""
Hyperparameter Tuning - Find optimal model parameters with Optuna.

Usage:
    python scripts/tune.py                     # 50 trials per model (default)
    python scripts/tune.py --trials 100        # 100 trials per model
    python scripts/tune.py --no-mlflow         # Tune without MLflow logging
    python scripts/tune.py --retrain           # Tune + retrain with best params

After tuning, use --retrain to automatically retrain and save the best model.
"""

import argparse
import json
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("scripts.tune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Trials per model (default: 50)")
    parser.add_argument("--folds", type=int, default=5, help="CV folds (default: 5)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--retrain", action="store_true", help="Retrain with best params after tuning")
    parser.add_argument("--data-path", type=str, default=None, help="Override features path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("EPL PREDICTOR - HYPERPARAMETER TUNING")
    print("=" * 60)

    config = load_config()
    if args.no_mlflow:
        config.mlflow.enabled = False

    data_path = args.data_path or f"{config.data.features}/model_ready.csv"
    if not Path(data_path).exists():
        logger.error(f"Features not found: {data_path}")
        sys.exit(1)

    from src.training.hyperparameter_tuning import HyperparameterTuner

    tuner = HyperparameterTuner(
        config=config,
        n_trials=args.trials,
        cv_folds=args.folds,
    )

    print(f"\nTrials per model: {args.trials}")
    print(f"CV folds:         {args.folds}")
    print(f"MLflow logging:   {'enabled' if config.mlflow.enabled else 'disabled'}")
    print()

    best_params = tuner.run(data_path)

    # Print results
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)

    for model_name, params in best_params.items():
        cv_score = tuner.best_scores[model_name]
        print(f"\n  {model_name} (CV={cv_score:.4f}):")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")

    overall_best = max(tuner.best_scores, key=tuner.best_scores.get)
    print(f"\n  Overall best: {overall_best} (CV={tuner.best_scores[overall_best]:.4f})")

    # Save best params to JSON
    output_path = Path("models/tuning_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "n_trials": args.trials,
        "cv_folds": args.folds,
        "best_params": best_params,
        "best_scores": tuner.best_scores,
        "overall_best_model": overall_best,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    # Retrain if requested
    if args.retrain:
        print("\n" + "-" * 60)
        print("RETRAINING WITH TUNED PARAMETERS")
        print("-" * 60)
        _retrain_with_tuned_params(tuner, config, data_path)


def _retrain_with_tuned_params(tuner, config, data_path: str) -> None:
    """Retrain all models using the best hyperparameters found by Optuna."""
    import pandas as pd
    from src.data.splitter import time_based_split
    from src.training.trainer import ModelTrainer
    from src.utils.constants import META_COLUMNS, TARGET_COLUMNS

    df = pd.read_csv(data_path, parse_dates=["date"])
    feature_cols = [c for c in df.columns if c not in META_COLUMNS + TARGET_COLUMNS]
    train_df, test_df = time_based_split(df, config.model.test_size)

    trainer = ModelTrainer(config.model)
    # Replace default models with tuned ones
    trainer.models = tuner.get_tuned_models()

    trainer.train(train_df, test_df, feature_cols)
    trainer.save(config.model.models_dir)
    trainer.save_predictions(test_df, f"{config.data.features}/test_predictions.csv")

    print(f"\n  Best model   : {trainer.best_name}")
    print(f"  Test accuracy: {trainer.best['test_acc']:.4f} ({trainer.best['test_acc']:.1%})")
    print(f"  Test F1 macro: {trainer.best['test_f1']:.4f}")

    print("\n  All tuned models:")
    for name, result in sorted(
        trainer.results.items(),
        key=lambda x: x[1]["test_acc"],
        reverse=True,
    ):
        marker = " *" if name == trainer.best_name else ""
        print(f"    {name:25s} acc={result['test_acc']:.4f}  f1={result['test_f1']:.4f}{marker}")

    # Log to MLflow if enabled
    if config.mlflow.enabled:
        try:
            from src.training.mlflow_trainer import MlflowTrainer
            mlflow_trainer = MlflowTrainer(config)
            mlflow_trainer.run(data_path)
            print("\n  MLflow experiment logged with tuned parameters.")
        except Exception as exc:
            logger.warning(f"MLflow logging failed: {exc}")


if __name__ == "__main__":
    main()
