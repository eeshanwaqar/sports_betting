"""
MLflow Trainer - Training with MLflow experiment tracking.

Wraps ModelTrainer to log every aspect of the training pipeline to MLflow:
- Parameters: feature config, model hyperparameters, data splits
- Metrics: accuracy, F1, log_loss per model; cross-validation scores
- Artifacts: confusion matrix plots, feature importance, model files
- Model registry: registers the best model for deployment

Usage:
    from src.training.mlflow_trainer import MlflowTrainer
    trainer = MlflowTrainer(config)
    trainer.run()  # Full pipeline with tracking
"""

import json
import tempfile
from typing import Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data.splitter import time_based_split
from src.training.cross_validation import cv_all_models
from src.training.evaluator import (
    bookmaker_accuracy,
    compute_log_loss,
    get_feature_importance,
)
from src.training.trainer import ModelTrainer
from src.utils.config import AppConfig, MlflowConfig, load_config
from src.utils.constants import META_COLUMNS, TARGET_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MlflowTrainer:
    """
    MLflow-integrated training pipeline.

    Orchestrates the full flow:
        load data -> split -> train models -> evaluate -> log to MLflow -> register best

    Each model gets its own nested run inside a parent experiment run,
    making it easy to compare across models and experiments.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.mlflow_cfg: MlflowConfig = self.config.mlflow
        self._setup_mlflow()
        self.trainer: Optional[ModelTrainer] = None

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking URI and experiment."""
        if not self.mlflow_cfg.enabled:
            logger.info("MLflow tracking disabled — running without tracking")
            return

        mlflow.set_tracking_uri(self.mlflow_cfg.tracking_uri)
        mlflow.set_experiment(self.mlflow_cfg.experiment_name)
        logger.info(
            f"MLflow configured: uri={self.mlflow_cfg.tracking_uri}, "
            f"experiment={self.mlflow_cfg.experiment_name}"
        )

    def run(self, data_path: Optional[str] = None) -> ModelTrainer:
        """
        Execute the full MLflow-tracked training pipeline.

        Args:
            data_path: Path to model_ready.csv. Defaults to config value.

        Returns:
            Trained ModelTrainer instance.
        """
        if data_path is None:
            data_path = f"{self.config.data.features}/model_ready.csv"

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=["date"])

        feature_cols = [
            c for c in df.columns if c not in META_COLUMNS + TARGET_COLUMNS
        ]
        logger.info(f"Found {len(feature_cols)} features, {len(df)} matches")

        # Time-based split
        train_df, test_df = time_based_split(df, self.config.model.test_size)

        if not self.mlflow_cfg.enabled:
            return self._train_without_tracking(train_df, test_df, feature_cols)

        return self._train_with_tracking(train_df, test_df, feature_cols, df)

    def _train_without_tracking(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> ModelTrainer:
        """Train without MLflow (fallback when disabled)."""
        self.trainer = ModelTrainer(self.config.model)
        self.trainer.train(train_df, test_df, feature_cols)
        self.trainer.save(self.config.model.models_dir)
        self.trainer.save_predictions(
            test_df, f"{self.config.data.features}/test_predictions.csv"
        )
        return self.trainer

    def _train_with_tracking(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        full_df: pd.DataFrame,
    ) -> ModelTrainer:
        """Train all models with full MLflow tracking."""
        with mlflow.start_run(run_name="training-pipeline") as parent_run:
            logger.info(f"MLflow parent run: {parent_run.info.run_id}")

            # --- Tag with git commit SHA for lineage ---
            import os
            git_sha = os.environ.get("GIT_COMMIT_SHA", "")
            if git_sha:
                mlflow.set_tag("git.commit", git_sha)
                mlflow.set_tag("mlflow.source.git.commit", git_sha)
                logger.info(f"Tagged run with git commit: {git_sha[:8]}")

            # --- Log global parameters ---
            self._log_global_params(train_df, test_df, feature_cols)

            # --- Train and log each model ---
            self.trainer = ModelTrainer(self.config.model)
            results = self.trainer.train(train_df, test_df, feature_cols)

            # --- Cross-validation ---
            cv_results = self._run_cross_validation(train_df, feature_cols)

            # --- Log each model as a nested run ---
            for model_name, result in results.items():
                self._log_model_run(
                    model_name, result, test_df, feature_cols, cv_results
                )

            # --- Log best model summary to parent ---
            best = self.trainer.best
            mlflow.log_metrics({
                "best_test_accuracy": best["test_acc"],
                "best_test_f1_macro": best["test_f1"],
            })
            mlflow.log_param("best_model", self.trainer.best_name)

            # Log bookmaker baseline if available
            bookie_acc = bookmaker_accuracy(test_df)
            if bookie_acc is not None:
                mlflow.log_metric("bookmaker_accuracy", bookie_acc)
                mlflow.log_metric(
                    "lift_over_bookmaker", best["test_acc"] - bookie_acc
                )
                logger.info(
                    f"Bookmaker accuracy: {bookie_acc:.4f}, "
                    f"Model lift: {best['test_acc'] - bookie_acc:+.4f}"
                )

            # --- Save artifacts locally and to MLflow ---
            self.trainer.save(self.config.model.models_dir)
            self.trainer.save_predictions(
                test_df, f"{self.config.data.features}/test_predictions.csv"
            )
            mlflow.log_artifacts(self.config.model.models_dir, "model_artifacts")

            # --- Register best model ---
            self._register_best_model(best, feature_cols)

            logger.info(
                f"MLflow run complete: {parent_run.info.run_id} | "
                f"Best: {self.trainer.best_name} ({best['test_acc']:.4f})"
            )

        return self.trainer

    def _log_global_params(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> None:
        """Log dataset and feature configuration parameters."""
        mlflow.log_params({
            # Data split info
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(feature_cols),
            "test_size": self.config.model.test_size,
            "train_date_range": (
                f"{train_df['date'].min().date()} to {train_df['date'].max().date()}"
            ),
            "test_date_range": (
                f"{test_df['date'].min().date()} to {test_df['date'].max().date()}"
            ),
            # Feature config
            "form_windows": str(self.config.features.form_windows),
            "exp_decay": self.config.features.exp_decay,
            "elo_k_factor": self.config.features.elo.k_factor,
            "elo_home_advantage": self.config.features.elo.home_advantage,
            "venue_window": self.config.features.venue_window,
            "h2h_window": self.config.features.h2h_window,
            "shooting_window": self.config.features.shooting_window,
        })

        # Log feature list as artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("\n".join(feature_cols))
            mlflow.log_artifact(f.name, "features")

    def _run_cross_validation(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, np.ndarray]:
        """Run cross-validation on all models and return results."""
        from sklearn.preprocessing import StandardScaler

        X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
        y_train = self.trainer.label_encoder.transform(train_df["target"])
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        cv_results = cv_all_models(
            self.trainer.models,
            X_train.values,
            X_train_scaled,
            y_train,
            n_splits=5,
        )
        return cv_results

    def _log_model_run(
        self,
        model_name: str,
        result: dict,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        cv_results: Dict[str, np.ndarray],
    ) -> None:
        """Log a single model's results as a nested MLflow run."""
        with mlflow.start_run(run_name=model_name, nested=True):
            model = result["model"]

            # Log hyperparameters
            params = model.get_params()
            # MLflow has a 500-char limit on param values; filter safely
            safe_params = {
                k: str(v)[:500] for k, v in params.items()
                if v is not None and not callable(v)
            }
            mlflow.log_params(safe_params)

            # Log metrics
            metrics = {
                "train_accuracy": result["train_acc"],
                "test_accuracy": result["test_acc"],
                "test_f1_macro": result["test_f1"],
                "overfit_gap": result["train_acc"] - result["test_acc"],
            }

            # Log loss
            y_test = self.trainer.label_encoder.transform(test_df["target"])
            try:
                ll = compute_log_loss(y_test, result["probabilities"])
                metrics["log_loss"] = ll
            except Exception:
                pass

            # CV scores
            if model_name in cv_results:
                cv_scores = cv_results[model_name]
                metrics["cv_mean_accuracy"] = float(cv_scores.mean())
                metrics["cv_std_accuracy"] = float(cv_scores.std())
                for i, score in enumerate(cv_scores):
                    metrics[f"cv_fold_{i+1}"] = float(score)

            mlflow.log_metrics(metrics)

            # Feature importance
            importance_df = get_feature_importance(model, feature_cols)
            if importance_df is not None:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False
                ) as f:
                    importance_df.to_csv(f.name, index=False)
                    mlflow.log_artifact(f.name, "feature_importance")

            # Confusion matrix as artifact
            cm = confusion_matrix(
                y_test,
                result["predictions"],
                labels=range(len(self.trainer.label_encoder.classes_)),
            )
            cm_data = {
                "labels": list(self.trainer.label_encoder.classes_),
                "matrix": cm.tolist(),
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(cm_data, f, indent=2)
                mlflow.log_artifact(f.name, "confusion_matrix")

            # Log the sklearn model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=np.zeros((1, len(feature_cols))),
            )

            logger.info(
                f"  Logged {model_name}: "
                f"acc={result['test_acc']:.4f}, "
                f"f1={result['test_f1']:.4f}"
            )

    def _register_best_model(
        self, best_result: dict, feature_cols: List[str]
    ) -> None:
        """Register the best model in MLflow Model Registry."""
        registry_name = self.mlflow_cfg.registry_name
        model = best_result["model"]

        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="best_model",
            registered_model_name=registry_name,
            input_example=np.zeros((1, len(feature_cols))),
        )
        logger.info(
            f"Registered model '{registry_name}' -> {model_info.model_uri}"
        )


def run_pipeline(config: Optional[AppConfig] = None) -> ModelTrainer:
    """
    Convenience entry point: run full MLflow-tracked training.

    Same interface as trainer.run_pipeline(), but with MLflow.
    """
    mlflow_trainer = MlflowTrainer(config)
    return mlflow_trainer.run()
