"""Configuration loading for CalibMiss."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class UniverseConfig(BaseModel):
    """Universe configuration."""

    tickers: list[str]
    start_date: str
    end_date: str
    source: str = "yfinance"


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ImputerConfig(BaseModel):
    """Imputer configuration."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class CalibratorConfig(BaseModel):
    """Calibrator configuration."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class MissingnessConfig(BaseModel):
    """Missingness configuration."""

    mechanisms: list[str]
    rates: list[float]
    seeds: list[int]


class SplitsConfig(BaseModel):
    """Walk-forward splits configuration."""

    train_years: int
    calibrate_months: int
    test_months: int
    roll_forward_months: int
    purge_days: int


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    n_bins_ece: int
    metrics: list[str]


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    models: list[ModelConfig]
    imputers: list[ImputerConfig]
    calibrators: list[CalibratorConfig]
    missingness: MissingnessConfig
    splits: SplitsConfig
    evaluation: EvaluationConfig


class MARRuleConfig(BaseModel):
    """MAR rule configuration."""

    target_feature: str
    depends_on: str
    rule_type: str
    params: dict[str, Any]


class MissingnessRulesConfig(BaseModel):
    """Missingness rules configuration."""

    mar_rules: list[MARRuleConfig]


class Config(BaseModel):
    """Root configuration."""

    universe: UniverseConfig
    experiment: ExperimentConfig


_config: Config | None = None


def get_config(config_dir: str | Path = "config") -> Config:
    """Load and return the full configuration.

    Args:
        config_dir: Directory containing config YAML files.

    Returns:
        Config object with all settings.
    """
    global _config
    if _config is not None:
        return _config

    config_path = Path(config_dir)

    # Load universe config
    with open(config_path / "universe.yaml") as f:
        universe_data = yaml.safe_load(f)
    universe = UniverseConfig(**universe_data["universe"])

    # Load experiment config
    with open(config_path / "experiment.yaml") as f:
        experiment_data = yaml.safe_load(f)
    experiment = ExperimentConfig(**experiment_data["experiment"])

    _config = Config(universe=universe, experiment=experiment)
    return _config


def get_mar_rules(config_dir: str | Path = "config") -> MissingnessRulesConfig:
    """Load MAR rules configuration.

    Args:
        config_dir: Directory containing config YAML files.

    Returns:
        MissingnessRulesConfig object.
    """
    config_path = Path(config_dir)
    with open(config_path / "missingness.yaml") as f:
        data = yaml.safe_load(f)
    return MissingnessRulesConfig(**data)


def reset_config() -> None:
    """Reset cached config (mainly for testing)."""
    global _config
    _config = None
