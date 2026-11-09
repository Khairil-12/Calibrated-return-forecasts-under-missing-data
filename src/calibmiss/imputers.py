"""Imputation strategies for CalibMiss."""

from typing import Any

import numpy as np
import pandas as pd

# Required for IterativeImputer (experimental)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


class ForwardFillImputer:
    """Forward fill imputer with per-ticker grouping and median fallback for leading NaNs."""

    def __init__(self) -> None:
        self.medians_: dict[str, float | dict[str, float]] = {}

    def _has_ticker_level(self, X: pd.DataFrame) -> bool:
        """Check if DataFrame has a 'ticker' level in its index."""
        return "ticker" in X.index.names

    def fit(self, X: pd.DataFrame) -> "ForwardFillImputer":
        """Compute medians for leading NaNs per ticker per column.

        Args:
            X: Training data with MultiIndex (ticker, date) or simple index

        Returns:
            Self
        """
        self.medians_ = {}
        for col in X.columns:
            if self._has_ticker_level(X):
                # Get first non-NaN per ticker group, or overall median
                first_valid = X.groupby(level="ticker")[col].apply(
                    lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan
                )
                overall_median = float(X[col].median())
                self.medians_[col] = first_valid.fillna(overall_median).to_dict()
            else:
                # Simple case: just use overall median (flat structure for compatibility)
                overall_median = float(X[col].median())
                self.medians_[col] = overall_median
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Forward fill within each ticker, then fill remaining leading NaNs with medians.

        Args:
            X: Data to impute

        Returns:
            Imputed DataFrame
        """
        X_imp = X.copy()
        for col in X.columns:
            med = self.medians_[col]
            if self._has_ticker_level(X_imp):
                # Forward fill within each ticker
                X_imp[col] = X_imp.groupby(level="ticker")[col].ffill()
                # Fill remaining leading NaNs with stored medians
                if isinstance(med, dict):
                    for ticker, m in med.items():
                        mask = (X_imp.index.get_level_values("ticker") == ticker) & X_imp[col].isna()
                        X_imp.loc[mask, col] = m
            else:
                # Simple forward fill
                X_imp[col] = X_imp[col].ffill()
                # Fill remaining leading NaNs with stored median
                if isinstance(med, int | float):
                    X_imp[col] = X_imp[col].fillna(med)
        return X_imp


class MedianImputer:
    """Median imputer - fits column medians on training data only."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame) -> "MedianImputer":
        """Compute column medians from training data.

        Args:
            X: Training data

        Returns:
            Self
        """
        self.medians_ = X.median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with stored medians.

        Args:
            X: Data to impute

        Returns:
            Imputed DataFrame
        """
        X_imp = X.copy()
        for col, med in self.medians_.items():
            X_imp[col] = X_imp[col].fillna(med)
        return X_imp


class IterativeImputerWrapper:
    """Wrapper for sklearn's IterativeImputer (MICE)."""

    def __init__(self, max_iter: int = 10, random_state: int = 42) -> None:
        self.imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        self.columns_: list[str] = []
        self.index_: pd.Index = pd.Index([])

    def fit(self, X: pd.DataFrame) -> "IterativeImputerWrapper":
        """Fit IterativeImputer on training data.

        Args:
            X: Training data

        Returns:
            Self
        """
        self.columns_ = X.columns.tolist()
        self.index_ = X.index
        self.imputer.fit(X.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted imputer.

        Args:
            X: Data to impute

        Returns:
            Imputed DataFrame
        """
        X_imp = pd.DataFrame(
            self.imputer.transform(X.values),
            columns=self.columns_,
            index=X.index
        )
        return X_imp


def get_imputer(config: Any) -> ForwardFillImputer | MedianImputer | IterativeImputerWrapper:
    """Factory function to get imputer instance from config.

    Args:
        config: ImputerConfig object with name and params

    Returns:
        Imputer instance
    """
    if config.name == "forward_fill":
        return ForwardFillImputer()
    elif config.name == "median":
        return MedianImputer()
    elif config.name == "iterative":
        return IterativeImputerWrapper(**config.params)
    raise ValueError(f"Unknown imputer: {config.name}")

