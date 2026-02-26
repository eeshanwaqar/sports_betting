"""
Config - Configuration loader and access.

Loads config.yaml and provides typed access to settings.
Supports environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    raw: str = "data/raw"
    processed: str = "data/processed"
    features: str = "data/features"
    predictions: str = "data/predictions"
    source_path: str = "archive/Datasets"


@dataclass
class EloConfig:
    k_factor: int = 20
    home_advantage: int = 100
    initial_rating: int = 1500


@dataclass
class FeatureConfig:
    form_windows: List[int] = field(default_factory=lambda: [3, 5, 10])
    exp_decay: float = 0.7
    venue_window: int = 5
    h2h_window: int = 10
    shooting_window: int = 10
    elo: EloConfig = field(default_factory=EloConfig)


@dataclass
class ModelConfig:
    algorithm: str = "gradient_boosting"
    test_size: float = 0.2
    random_state: int = 42
    models_dir: str = "models"


@dataclass
class MlflowConfig:
    tracking_uri: str = "mlruns"
    experiment_name: str = "epl-match-prediction"
    registry_name: str = "epl-predictor"
    enabled: bool = True


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def _find_config_path() -> Optional[Path]:
    """Search for config.yaml up the directory tree."""
    candidates = [
        Path("config.yaml"),
        Path(__file__).resolve().parent.parent.parent / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file and return typed AppConfig.

    Falls back to defaults if no config file is found.
    Environment variables prefixed with EPL_ override config values.
    """
    raw = {}
    path = Path(config_path) if config_path else _find_config_path()

    if path and path.exists():
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

    data_raw = raw.get("data", {})
    data_cfg = DataConfig(
        raw=data_raw.get("raw", "data/raw"),
        processed=data_raw.get("processed", "data/processed"),
        features=data_raw.get("features", "data/features"),
        predictions=data_raw.get("predictions", "data/predictions"),
        source_path=raw.get("source", {}).get("path", "archive/Datasets"),
    )

    feat_raw = raw.get("features", {})
    elo_raw = feat_raw.get("elo", {})
    elo_cfg = EloConfig(
        k_factor=elo_raw.get("k_factor", 20),
        home_advantage=elo_raw.get("home_advantage", 100),
        initial_rating=elo_raw.get("initial_rating", 1500),
    )
    feat_cfg = FeatureConfig(
        form_windows=feat_raw.get("form_windows", [3, 5, 10]),
        exp_decay=feat_raw.get("exp_decay", 0.7),
        venue_window=feat_raw.get("venue_window", 5),
        h2h_window=feat_raw.get("h2h_window", 10),
        shooting_window=feat_raw.get("shooting_window", 10),
        elo=elo_cfg,
    )

    model_raw = raw.get("model", {})
    model_cfg = ModelConfig(
        algorithm=model_raw.get("algorithm", "gradient_boosting"),
        test_size=model_raw.get("test_size", 0.2),
        random_state=model_raw.get("random_state", 42),
        models_dir=model_raw.get("models_dir", "models"),
    )

    mlflow_raw = raw.get("mlflow", {})
    mlflow_cfg = MlflowConfig(
        tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI",
            mlflow_raw.get("tracking_uri", "mlruns"),
        ),
        experiment_name=mlflow_raw.get("experiment_name", "epl-match-prediction"),
        registry_name=mlflow_raw.get("registry_name", "epl-predictor"),
        enabled=mlflow_raw.get("enabled", True),
    )

    api_raw = raw.get("api", {})
    api_cfg = ApiConfig(
        host=os.environ.get("EPL_API_HOST", api_raw.get("host", "0.0.0.0")),
        port=int(os.environ.get("EPL_API_PORT", api_raw.get("port", 8000))),
        debug=api_raw.get("debug", False),
    )

    return AppConfig(
        data=data_cfg,
        features=feat_cfg,
        model=model_cfg,
        mlflow=mlflow_cfg,
        api=api_cfg,
    )
