"""
Model Registry - MLflow model registry operations.

Provides utilities to:
- List registered model versions
- Load a model from the registry by name/alias
- Transition model stages (via aliases: champion, challenger)
- Compare model versions

Usage:
    from src.training.model_registry import ModelRegistry
    registry = ModelRegistry(config)
    model = registry.load_champion()
    registry.promote_to_champion(version=3)
"""

from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.config import AppConfig, MlflowConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Interface for MLflow Model Registry operations.

    MLflow 3.x uses model aliases (e.g., 'champion', 'challenger')
    instead of the deprecated stages (Staging, Production, Archived).
    """

    CHAMPION_ALIAS = "champion"
    CHALLENGER_ALIAS = "challenger"

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.mlflow_cfg: MlflowConfig = self.config.mlflow
        self.registry_name: str = self.mlflow_cfg.registry_name

        mlflow.set_tracking_uri(self.mlflow_cfg.tracking_uri)
        self.client = MlflowClient()
        logger.info(f"Registry initialized for model '{self.registry_name}'")

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all registered versions of the model.

        Returns:
            List of dicts with version info (version, run_id, aliases, etc.).
        """
        try:
            versions = self.client.search_model_versions(
                f"name='{self.registry_name}'"
            )
        except Exception as exc:
            logger.warning(f"Could not list versions: {exc}")
            return []

        result = []
        for v in versions:
            result.append({
                "version": int(v.version),
                "run_id": v.run_id,
                "aliases": list(v.aliases) if v.aliases else [],
                "creation_timestamp": v.creation_timestamp,
                "description": v.description or "",
            })

        result.sort(key=lambda x: x["version"], reverse=True)
        logger.info(f"Found {len(result)} model version(s)")
        return result

    def load_champion(self) -> Any:
        """
        Load the current champion (production) model.

        Returns:
            Loaded sklearn model object.

        Raises:
            ValueError: If no champion alias is set.
        """
        model_uri = f"models:/{self.registry_name}@{self.CHAMPION_ALIAS}"
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded champion model from {model_uri}")
            return model
        except Exception as exc:
            raise ValueError(
                f"No champion model found for '{self.registry_name}'. "
                f"Set the '{self.CHAMPION_ALIAS}' alias first. Error: {exc}"
            ) from exc

    def load_version(self, version: int) -> Any:
        """
        Load a specific model version.

        Args:
            version: Integer version number.

        Returns:
            Loaded sklearn model object.
        """
        model_uri = f"models:/{self.registry_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model version {version} from {model_uri}")
        return model

    def promote_to_champion(
        self,
        version: int,
        description: Optional[str] = None,
    ) -> None:
        """
        Promote a model version to champion (production).

        The previous champion (if any) gets the 'challenger' alias removed,
        and the new version gets the 'champion' alias.

        Args:
            version: Version number to promote.
            description: Optional description for the version.
        """
        # Set the champion alias on the new version
        self.client.set_registered_model_alias(
            name=self.registry_name,
            alias=self.CHAMPION_ALIAS,
            version=str(version),
        )
        logger.info(
            f"Version {version} promoted to '{self.CHAMPION_ALIAS}'"
        )

        if description:
            self.client.update_model_version(
                name=self.registry_name,
                version=str(version),
                description=description,
            )

    def set_challenger(self, version: int) -> None:
        """
        Mark a model version as challenger (staging/candidate).

        Args:
            version: Version number to set as challenger.
        """
        self.client.set_registered_model_alias(
            name=self.registry_name,
            alias=self.CHALLENGER_ALIAS,
            version=str(version),
        )
        logger.info(
            f"Version {version} set as '{self.CHALLENGER_ALIAS}'"
        )

    def get_version_metrics(self, version: int) -> Dict[str, float]:
        """
        Retrieve metrics for a specific model version from its run.

        Args:
            version: Version number.

        Returns:
            Dict of metric name → value.
        """
        versions = self.client.search_model_versions(
            f"name='{self.registry_name}'"
        )
        target = None
        for v in versions:
            if int(v.version) == version:
                target = v
                break

        if target is None:
            raise ValueError(f"Version {version} not found")

        run = self.client.get_run(target.run_id)
        return dict(run.data.metrics)

    def compare_versions(
        self, version_a: int, version_b: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics between two model versions.

        Args:
            version_a: First version.
            version_b: Second version.

        Returns:
            Dict with keys 'version_a', 'version_b', each containing metrics.
        """
        metrics_a = self.get_version_metrics(version_a)
        metrics_b = self.get_version_metrics(version_b)

        logger.info(
            f"Comparing v{version_a} vs v{version_b}: "
            f"acc {metrics_a.get('test_accuracy', 'N/A')} vs "
            f"{metrics_b.get('test_accuracy', 'N/A')}"
        )

        return {
            "version_a": {"version": version_a, **metrics_a},
            "version_b": {"version": version_b, **metrics_b},
        }

    def delete_version(self, version: int) -> None:
        """
        Delete a model version from the registry.

        Args:
            version: Version number to delete.
        """
        self.client.delete_model_version(
            name=self.registry_name, version=str(version)
        )
        logger.info(f"Deleted version {version} of '{self.registry_name}'")
