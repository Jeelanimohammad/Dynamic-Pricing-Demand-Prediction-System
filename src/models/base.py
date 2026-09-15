"""
Base class for demand forecasting models.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class BaseDemandModel(ABC):
    """
    Abstract interface for demand forecasting models.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDemandModel":
        """Fits the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts expected demand volume."""
        pass

    @abstractmethod
    def get_feature_importance(self, feature_names: list) -> Dict[str, float]:
        """Returns feature importance scores."""
        pass
