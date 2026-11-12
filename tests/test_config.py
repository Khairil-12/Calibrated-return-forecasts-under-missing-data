"""Tests for config module."""

from pathlib import Path

from calibmiss.config import get_config, get_mar_rules, reset_config


def test_load_universe_config():
    """Test loading universe configuration."""
    reset_config()
    cfg = get_config()

    assert len(cfg.universe.tickers) == 30
    assert cfg.universe.start_date == "2010-01-01"
    assert cfg.universe.end_date == "2024-12-31"
    assert cfg.universe.source == "yfinance"

    # Check some specific tickers
    assert "AAPL" in cfg.universe.tickers
    assert "MSFT" in cfg.universe.tickers
    assert "GOOGL" in cfg.universe.tickers


def test_load_experiment_config():
    """Test loading experiment configuration."""
    reset_config()
    cfg = get_config()

    assert len(cfg.experiment.models) == 2
    assert len(cfg.experiment.imputers) == 3
    assert len(cfg.experiment.calibrators) == 3

    # Check model names
    model_names = [m.name for m in cfg.experiment.models]
    assert "logistic_regression" in model_names
    assert "lightgbm" in model_names

    # Check imputer names
    imputer_names = [i.name for i in cfg.experiment.imputers]
    assert "forward_fill" in imputer_names
    assert "median" in imputer_names
    assert "iterative" in imputer_names

    # Check calibrator names
    calibrator_names = [c.name for c in cfg.experiment.calibrators]
    assert "none" in calibrator_names
    assert "platt" in calibrator_names
    assert "isotonic" in calibrator_names

    # Check missingness config
    assert cfg.experiment.missingness.mechanisms == ["MCAR", "MAR"]
    assert cfg.experiment.missingness.rates == [0.0, 0.1, 0.2, 0.3]
    assert cfg.experiment.missingness.seeds == [42, 123, 456, 789, 999]

    # Check splits config
    assert cfg.experiment.splits.train_years == 3
    assert cfg.experiment.splits.calibrate_months == 6
    assert cfg.experiment.splits.test_months == 6
    assert cfg.experiment.splits.roll_forward_months == 6
    assert cfg.experiment.splits.purge_days == 5

    # Check evaluation config
    assert cfg.experiment.evaluation.n_bins_ece == 10
    assert "brier" in cfg.experiment.evaluation.metrics
    assert "ece" in cfg.experiment.evaluation.metrics
    assert "roc_auc" in cfg.experiment.evaluation.metrics


def test_load_mar_rules():
    """Test loading MAR rules configuration."""
    rules = get_mar_rules()

    assert len(rules.mar_rules) == 1
    rule = rules.mar_rules[0]
    assert rule.target_feature == "volatility_20d"
    assert rule.depends_on == "volume_ratio"
    assert rule.rule_type == "quantile_threshold"
    assert rule.params["base_prob"] == 0.05
    assert rule.params["max_prob"] == 0.5
    assert rule.params["quantile_threshold"] == 0.7


def test_config_caching():
    """Config loads fresh each call; the argumentless form matches defaults."""
    reset_config()
    cfg1 = get_config()
    cfg2 = get_config()

    assert cfg1.experiment.splits.purge_days == cfg2.experiment.splits.purge_days


def test_get_config_respects_experiment_file_argument():
    """Second call with a different experiment_file must honor it, not return the cached first config."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    reset_config()
    direction = get_config(config_dir, experiment_file="experiment.yaml")
    reset_config()
    volatility = get_config(config_dir, experiment_file="experiment_volatility.yaml")

    assert direction.experiment.target == "direction"
    assert volatility.experiment.target == "volatility"
    assert volatility.experiment.splits.purge_days == 60


def test_get_config_honors_experiment_file_without_reset_between_calls():
    """No reset between calls: the second call must still return the volatility config."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    reset_config()
    _ = get_config(config_dir, experiment_file="experiment.yaml")
    volatility = get_config(config_dir, experiment_file="experiment_volatility.yaml")

    assert volatility.experiment.target == "volatility"


def test_reset_config():
    """reset_config remains callable for test fixtures."""
    reset_config()
    cfg1 = get_config()
    reset_config()
    cfg2 = get_config()

    assert cfg1.experiment.splits.purge_days == cfg2.experiment.splits.purge_days
