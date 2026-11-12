"""Model wrappers with a shared positive-class probability interface."""

from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from calibmiss.config import ModelConfig


class LogisticRegressionModel:
    """Standardized logistic regression classifier."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | dict[int, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            **kwargs,
        )
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self.model.predict_proba(self.scaler.transform(X))[:, 1])


class LightGBMModel:
    """LightGBM binary classifier."""

    def __init__(self, **kwargs: Any) -> None:
        params = {"n_jobs": 1, **kwargs}
        self.model = LGBMClassifier(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return cast(NDArray[np.float64], self.model.predict_proba(X)[:, 1])


def get_model(config: ModelConfig) -> LogisticRegressionModel | LightGBMModel:
    """Build a model wrapper from experiment configuration."""
    if config.name == "logistic_regression":
        return LogisticRegressionModel(**config.params)
    if config.name == "lightgbm":
        return LightGBMModel(**config.params)
    raise ValueError(f"Unknown model: {config.name}")
