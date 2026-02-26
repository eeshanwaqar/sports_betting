"""
Trainer - Core model training logic.

Maps to: notebooks/04_modeling.ipynb (model definition, training loop, saving)

Trains multiple models, compares them, and saves the best.
"""

import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Any
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score

from src.utils.config import AppConfig, ModelConfig
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir
from src.utils.constants import META_COLUMNS, TARGET_COLUMNS, DEFAULT_RANDOM_STATE

logger = get_logger(__name__)


def _build_models(random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, Any]:
    """
    Define models with regularization tuned for ~59 features.

    Mirrors notebook 04 model definitions.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.5,
            solver="lbfgs",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            min_samples_split=40,
            max_features="sqrt",
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=20,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        ),
    }

    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=10,
            eval_metric="mlogloss",
            random_state=random_state,
        )
    except ImportError:
        logger.info("XGBoost not installed, skipping")

    return models


class ModelTrainer:
    """
    Trains, compares, and saves ML models for match prediction.

    Usage:
        trainer = ModelTrainer(config)
        trainer.train(train_df, test_df, feature_cols)
        trainer.save("models/")
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = _build_models(self.config.random_state)
        self.results: Dict[str, dict] = {}
        self.calibrators: Dict[str, Any] = {}
        self.best_name: Optional[str] = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols: List[str] = []
        self._train_medians: Optional[pd.Series] = None
        self._cal_samples: int = 0

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, dict]:
        """
        Train all models, calibrate probabilities, and evaluate on test set.

        Carves a calibration set from the end of training data (respecting
        time ordering) and fits isotonic calibrators for each model.

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            feature_cols: List of feature column names.

        Returns:
            Dict of model name → results dict.
        """
        self.feature_cols = feature_cols

        # Prepare data
        self._train_medians = train_df[feature_cols].median()
        X_full_train = train_df[feature_cols].fillna(self._train_medians)
        X_test = test_df[feature_cols].fillna(self._train_medians)

        y_full_train = self.label_encoder.fit_transform(train_df["target"])
        y_test = self.label_encoder.transform(test_df["target"])

        # Carve calibration set from end of training data (last 12.5% of train = ~10% of total)
        cal_fraction = 0.125
        cal_split = int(len(X_full_train) * (1 - cal_fraction))
        X_train = X_full_train.iloc[:cal_split]
        X_cal = X_full_train.iloc[cal_split:]
        y_train = y_full_train[:cal_split]
        y_cal = y_full_train[cal_split:]
        self._cal_samples = len(y_cal)

        logger.info(
            f"Data split: {len(y_train)} train, {len(y_cal)} calibration, {len(y_test)} test"
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_cal_scaled = self.scaler.transform(X_cal)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Training {len(self.models)} models on {len(feature_cols)} features...")

        for name, model in self.models.items():
            logger.info(f"  Training {name}...")

            # Linear models use scaled data, tree models use raw
            if "Logistic" in name:
                model.fit(X_train_scaled, y_train)
                cal_data, test_data = X_cal_scaled, X_test_scaled
            else:
                model.fit(X_train, y_train)
                cal_data, test_data = X_cal, X_test

            # Fit isotonic calibrator on held-out calibration set
            # FrozenEstimator prevents refitting — calibrator only learns the probability mapping
            calibrator = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
            calibrator.fit(cal_data, y_cal)
            self.calibrators[name] = calibrator

            # Evaluate using calibrated probabilities
            train_pred = model.predict(X_train_scaled if "Logistic" in name else X_train)
            test_pred = calibrator.predict(test_data)
            test_proba = calibrator.predict_proba(test_data)

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average="macro")

            self.results[name] = {
                "model": model,
                "calibrator": calibrator,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "predictions": test_pred,
                "probabilities": test_proba,
            }

            overfit = train_acc - test_acc
            logger.info(
                f"    Train: {train_acc:.4f}, Test: {test_acc:.4f}, "
                f"F1: {test_f1:.4f}, Overfit: {overfit:.4f} (calibrated)"
            )

        # Select best by test accuracy
        self.best_name = max(self.results, key=lambda k: self.results[k]["test_acc"])
        logger.info(f"Best model: {self.best_name} (test acc: {self.results[self.best_name]['test_acc']:.4f})")

        return self.results

    @property
    def best(self) -> dict:
        """Get the best model results."""
        if self.best_name is None:
            raise ValueError("No models trained yet. Call train() first.")
        return self.results[self.best_name]

    def save(self, output_dir: str = "models") -> None:
        """
        Save best model and all artifacts.

        Saves: best_model.joblib, best_calibrator.joblib, scaler.joblib,
               label_encoder.joblib, features.txt, model_info.json
        """
        if self.best_name is None:
            raise ValueError("No models trained yet. Call train() first.")

        ensure_dir(output_dir)
        out = Path(output_dir)

        joblib.dump(self.best["model"], out / "best_model.joblib")
        logger.info(f"Saved: {out / 'best_model.joblib'}")

        # Save calibrator for the best model
        if self.best_name in self.calibrators:
            joblib.dump(self.calibrators[self.best_name], out / "best_calibrator.joblib")
            logger.info(f"Saved: {out / 'best_calibrator.joblib'}")

        joblib.dump(self.scaler, out / "scaler.joblib")
        logger.info(f"Saved: {out / 'scaler.joblib'}")

        joblib.dump(self.label_encoder, out / "label_encoder.joblib")
        logger.info(f"Saved: {out / 'label_encoder.joblib'}")

        # Save training medians for NaN imputation at inference time.
        # Features like betting odds aren't available for future matches,
        # so we fill them with training medians instead of 0.0 to avoid
        # sending extreme out-of-distribution values to the model.
        if self._train_medians is not None:
            joblib.dump(self._train_medians.to_dict(), out / "train_medians.joblib")
            logger.info(f"Saved: {out / 'train_medians.joblib'}")

        with open(out / "features.txt", "w") as f:
            f.write("\n".join(self.feature_cols))
        logger.info(f"Saved: {out / 'features.txt'}")

        model_info = {
            "model_type": self.best_name,
            "test_accuracy": round(self.best["test_acc"], 4),
            "test_f1_macro": round(self.best["test_f1"], 4),
            "n_features": len(self.feature_cols),
            "calibration_method": "isotonic",
            "calibration_samples": self._cal_samples,
            "all_models": {
                name: {
                    "test_acc": round(r["test_acc"], 4),
                    "test_f1": round(r["test_f1"], 4),
                }
                for name, r in self.results.items()
            },
        }
        with open(out / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Saved: {out / 'model_info.json'}")

    def save_predictions(
        self, test_df: pd.DataFrame, output_path: str = "data/features/test_predictions.csv"
    ) -> None:
        """Save test set with predictions and probabilities."""
        if self.best_name is None:
            raise ValueError("No models trained yet.")

        pred_df = test_df.copy()
        pred_df["predicted"] = self.label_encoder.inverse_transform(self.best["predictions"])
        for i, cls in enumerate(self.label_encoder.classes_):
            pred_df[f"prob_{cls}"] = self.best["probabilities"][:, i]

        ensure_dir(str(Path(output_path).parent))
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions: {output_path} ({len(pred_df)} rows)")


def run_pipeline(config: Optional[AppConfig] = None) -> ModelTrainer:
    """
    Execute the full training pipeline.

    Loads model_ready.csv → splits → trains → saves.
    """
    from src.data.splitter import time_based_split

    if config is None:
        from src.utils.config import load_config
        config = load_config()

    df = pd.read_csv(f"{config.data.features}/model_ready.csv", parse_dates=["date"])

    feature_cols = [
        c for c in df.columns if c not in META_COLUMNS + TARGET_COLUMNS
    ]

    train_df, test_df = time_based_split(df, config.model.test_size)

    trainer = ModelTrainer(config.model)
    trainer.train(train_df, test_df, feature_cols)
    trainer.save(config.model.models_dir)
    trainer.save_predictions(test_df, f"{config.data.features}/test_predictions.csv")

    return trainer
