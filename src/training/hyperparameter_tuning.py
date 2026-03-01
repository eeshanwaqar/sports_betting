"""
Hyperparameter Tuning - Optuna + MLflow integration.

Defines search spaces for each model, runs Bayesian optimization with
TimeSeriesSplit cross-validation, and logs every trial to MLflow.

Usage:
    from src.training.hyperparameter_tuning import HyperparameterTuner
    tuner = HyperparameterTuner(config)
    best_params = tuner.run(data_path)
"""

from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import AppConfig
from src.utils.constants import DEFAULT_RANDOM_STATE, META_COLUMNS, TARGET_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress Optuna's verbose default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_logistic_regression(trial: optuna.Trial) -> dict:
    """Search space for Logistic Regression."""
    return {
        "C": trial.suggest_float("lr_C", 0.01, 10.0, log=True),
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": DEFAULT_RANDOM_STATE,
    }


def _suggest_random_forest(trial: optuna.Trial) -> dict:
    """Search space for Random Forest."""
    return {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("rf_max_depth", 4, 12),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 5, 50),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 10, 80),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2"]),
        "random_state": DEFAULT_RANDOM_STATE,
    }


def _suggest_gradient_boosting(trial: optuna.Trial) -> dict:
    """Search space for Gradient Boosting."""
    return {
        "n_estimators": trial.suggest_int("gb_n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("gb_max_depth", 2, 6),
        "learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("gb_subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("gb_min_samples_leaf", 10, 50),
        "random_state": DEFAULT_RANDOM_STATE,
    }


def _suggest_xgboost(trial: optuna.Trial) -> dict:
    """Search space for XGBoost."""
    return {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("xgb_max_depth", 2, 8),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 20),
        "eval_metric": "mlogloss",
        "random_state": DEFAULT_RANDOM_STATE,
    }


MODEL_FACTORIES = {
    "Logistic Regression": (LogisticRegression, _suggest_logistic_regression),
    "Random Forest": (RandomForestClassifier, _suggest_random_forest),
    "Gradient Boosting": (GradientBoostingClassifier, _suggest_gradient_boosting),
}

# Add XGBoost if available
try:
    from xgboost import XGBClassifier
    MODEL_FACTORIES["XGBoost"] = (XGBClassifier, _suggest_xgboost)
except ImportError:
    pass


class HyperparameterTuner:
    """
    Bayesian hyperparameter optimization with Optuna.

    Each model type gets its own Optuna study. Every trial is evaluated
    using TimeSeriesSplit CV and optionally logged to MLflow.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        n_trials: int = 50,
        cv_folds: int = 5,
    ):
        if config is None:
            from src.utils.config import load_config
            config = load_config()

        self.config = config
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params: Dict[str, dict] = {}
        self.best_scores: Dict[str, float] = {}

    def _load_data(self, data_path: Optional[str] = None):
        """Load and prepare the feature matrix."""
        path = data_path or f"{self.config.data.features}/model_ready.csv"
        df = pd.read_csv(path, parse_dates=["date"])

        feature_cols = [
            c for c in df.columns if c not in META_COLUMNS + TARGET_COLUMNS
        ]

        from src.data.splitter import time_based_split
        train_df, _ = time_based_split(df, self.config.model.test_size)

        train_medians = train_df[feature_cols].median()
        X = train_df[feature_cols].fillna(train_medians).values
        le = LabelEncoder()
        y = le.fit_transform(train_df["target"])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X, X_scaled, y, feature_cols

    def _make_objective(
        self,
        model_name: str,
        X: np.ndarray,
        X_scaled: np.ndarray,
        y: np.ndarray,
    ):
        """Create an Optuna objective function for a given model type."""
        model_class, suggest_fn = MODEL_FACTORIES[model_name]
        use_scaled = "Logistic" in model_name

        def objective(trial: optuna.Trial) -> float:
            params = suggest_fn(trial)
            model = model_class(**params)
            data = X_scaled if use_scaled else X

            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = cross_val_score(model, data, y, cv=tscv, scoring="accuracy")
            return scores.mean()

        return objective

    def tune_model(
        self,
        model_name: str,
        X: np.ndarray,
        X_scaled: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Run Optuna study for a single model type.

        Returns the best hyperparameters found.
        """
        logger.info(f"Tuning {model_name} ({self.n_trials} trials, {self.cv_folds}-fold CV)...")

        study = optuna.create_study(
            direction="maximize",
            study_name=f"tune-{model_name.lower().replace(' ', '-')}",
        )

        objective = self._make_objective(model_name, X, X_scaled, y)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_trial
        logger.info(f"  Best CV accuracy: {best.value:.4f} (trial {best.number})")

        self.best_params[model_name] = best.params
        self.best_scores[model_name] = best.value

        return best.params

    def run(self, data_path: Optional[str] = None) -> Dict[str, dict]:
        """
        Tune all model types and return best params.

        Optionally logs to MLflow if enabled.
        """
        X, X_scaled, y, feature_cols = self._load_data(data_path)

        logger.info(
            f"Starting hyperparameter tuning: "
            f"{len(MODEL_FACTORIES)} models, {self.n_trials} trials each"
        )

        mlflow_enabled = self.config.mlflow.enabled
        mlflow = None
        parent_run = None

        if mlflow_enabled:
            try:
                import mlflow as _mlflow
                mlflow = _mlflow
                mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
                mlflow.set_experiment(self.config.mlflow.experiment_name)
                parent_run = mlflow.start_run(run_name="hyperparameter-tuning")
                mlflow.log_param("n_trials_per_model", self.n_trials)
                mlflow.log_param("cv_folds", self.cv_folds)
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_param("n_train_samples", len(y))
            except Exception as exc:
                logger.warning(f"MLflow setup failed, continuing without tracking: {exc}")
                mlflow_enabled = False

        for model_name in MODEL_FACTORIES:
            best = self.tune_model(model_name, X, X_scaled, y)

            if mlflow_enabled and mlflow is not None:
                try:
                    with mlflow.start_run(
                        run_name=f"tune-{model_name}", nested=True,
                    ):
                        mlflow.log_params({
                            f"best_{k}": v for k, v in best.items()
                        })
                        mlflow.log_metric("best_cv_accuracy", self.best_scores[model_name])
                        mlflow.log_metric("n_trials", self.n_trials)
                except Exception as exc:
                    logger.warning(f"Failed to log {model_name} to MLflow: {exc}")

        # Log overall best
        overall_best_name = max(self.best_scores, key=self.best_scores.get)
        overall_best_acc = self.best_scores[overall_best_name]

        if mlflow_enabled and mlflow is not None:
            try:
                mlflow.log_param("overall_best_model", overall_best_name)
                mlflow.log_metric("overall_best_cv_accuracy", overall_best_acc)
            except Exception:
                pass

        if parent_run is not None:
            mlflow.end_run()

        logger.info(f"\nTuning complete. Overall best: {overall_best_name} (CV={overall_best_acc:.4f})")
        for name in self.best_scores:
            marker = " *" if name == overall_best_name else ""
            logger.info(f"  {name:25s} CV={self.best_scores[name]:.4f}{marker}")

        return self.best_params

    def get_tuned_models(self) -> Dict[str, Any]:
        """
        Build model instances using the best hyperparameters found.

        Returns a dict of model_name -> configured model instance.
        """
        if not self.best_params:
            raise ValueError("No tuning results. Call run() first.")

        tuned = {}
        for model_name, params in self.best_params.items():
            model_class, _ = MODEL_FACTORIES[model_name]
            # Strip parameter name prefixes (e.g. lr_C -> C)
            clean_params = {}
            for key, value in params.items():
                parts = key.split("_", 1)
                clean_key = parts[1] if len(parts) > 1 else key
                clean_params[clean_key] = value
            # Add common fixed params
            clean_params["random_state"] = DEFAULT_RANDOM_STATE
            if model_name == "Logistic Regression":
                clean_params["solver"] = "lbfgs"
                clean_params["max_iter"] = 2000
            elif model_name == "XGBoost":
                clean_params["eval_metric"] = "mlogloss"

            tuned[model_name] = model_class(**clean_params)
        return tuned
