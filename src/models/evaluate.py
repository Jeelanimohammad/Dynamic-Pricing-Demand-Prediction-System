"""
Evaluation and diagnostic benchmarking for demand prediction models.
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.utils.metrics import calculate_regression_metrics
from src.config import METRICS_FILE
from src.utils.logger import get_logger

logger = get_logger("model_evaluator")


class ModelEvaluator:
    """
    Evaluates individual models and ensemble performance on test data.
    """

    @staticmethod
    def evaluate_models(
        ensemble_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        save_json: bool = True
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation on test set across all sub-models and ensemble.
        """
        predictions = ensemble_model.predict_individual(X_test)
        evaluation_results: Dict[str, Any] = {}

        for model_name, y_pred in predictions.items():
            metrics = calculate_regression_metrics(y_test, y_pred)
            evaluation_results[model_name] = metrics
            logger.info(
                f"Model [{model_name.upper()}]: RMSE={metrics['rmse']}, "
                f"MAE={metrics['mae']}, MAPE={metrics['mape_percent']}%, R2={metrics['r2_score']}"
            )

        # Feature Importance
        feature_importances = ensemble_model.get_feature_importance(feature_names)
        top_10_features = dict(list(feature_importances.items())[:10])

        report = {
            "model_metrics": evaluation_results,
            "top_features": top_10_features,
            "all_features": feature_importances,
            "test_sample_count": len(y_test)
        }

        if save_json:
            METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(METRICS_FILE, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Saved evaluation metrics to {METRICS_FILE}")

        return report
