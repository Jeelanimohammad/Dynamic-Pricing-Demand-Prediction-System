"""
Demand forecasting models including Random Forest, Gradient Boosting, Ridge, and Ensemble.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from src.models.base import BaseDemandModel
from src.config import RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger("demand_predictor")


class DemandPredictorEnsemble(BaseDemandModel):
    """
    State-of-the-art Ensemble Model for demand forecasting combining Gradient Boosting,
    Random Forest, and Regularized ElasticNet regression.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.models: Dict[str, Any] = {
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=5,
                subsample=0.85,
                random_state=seed
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=4,
                n_jobs=-1,
                random_state=seed
            ),
            "elastic_net": ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=seed
            )
        }
        # Learned or heuristic ensemble weights
        self.weights = {
            "gradient_boosting": 0.50,
            "random_forest": 0.35,
            "elastic_net": 0.15
        }
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DemandPredictorEnsemble":
        """
        Fits all constituent sub-models on training features X and target demand y.
        """
        logger.info(f"Training DemandPredictorEnsemble on {X.shape[0]} samples with {X.shape[1]} features...")
        for name, model in self.models.items():
            logger.info(f"Fitting sub-model: {name}...")
            model.fit(X, y)
        logger.info("DemandPredictorEnsemble training completed successfully.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates ensemble weighted demand predictions. Enforces non-negativity.
        """
        ensemble_pred = np.zeros(X.shape[0], dtype=float)
        for name, model in self.models.items():
            pred = model.predict(X)
            pred = np.maximum(pred, 0.0)
            ensemble_pred += self.weights[name] * pred

        return np.maximum(ensemble_pred, 0.0)

    def predict_individual(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns predictions from each individual sub-model for comparison and variance analysis.
        """
        results = {}
        for name, model in self.models.items():
            pred = np.maximum(model.predict(X), 0.0)
            results[name] = pred
        results["ensemble"] = self.predict(X)
        return results

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Computes aggregated feature importance scores across tree models.
        """
        self.feature_names = feature_names
        gb_importances = self.models["gradient_boosting"].feature_importances_
        rf_importances = self.models["random_forest"].feature_importances_

        # Blend tree importances
        blended = 0.6 * gb_importances + 0.4 * rf_importances
        importance_dict = {
            feature_names[i]: round(float(blended[i]), 4)
            for i in range(min(len(feature_names), len(blended)))
        }

        # Sort descending
        sorted_importances = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return sorted_importances
