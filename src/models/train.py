"""
End-to-end model training, cross-validation, and model serialization pipeline.
"""
import joblib
import pandas as pd
from typing import Dict, Any, Optional
from src.config import (
    MODEL_CHECKPOINT_FILE,
    ELASTICITY_MODEL_FILE,
    RAW_DATA_FILE,
    RANDOM_SEED
)
from src.data.generator import MarketDataGenerator
from src.data.preprocessor import DataPreprocessor
from src.models.demand_predictor import DemandPredictorEnsemble
from src.models.elasticity_model import PriceElasticityModel
from src.models.evaluate import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger("model_trainer")


class ModelTrainer:
    """
    Orchestrates data preparation, training of forecasting and elasticity models,
    and serializes production-ready artifacts.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.preprocessor = DataPreprocessor()
        self.demand_model = DemandPredictorEnsemble(seed=seed)
        self.elasticity_model = PriceElasticityModel()
        self.evaluator = ModelEvaluator()

    def train_pipeline(
        self,
        force_regenerate: bool = False,
        days: int = 365
    ) -> Dict[str, Any]:
        """
        Executes complete training pipeline.
        """
        logger.info("--- Starting End-to-End Training Pipeline ---")

        # 1. Data Generation (if missing or forced)
        if force_regenerate or not RAW_DATA_FILE.exists():
            generator = MarketDataGenerator(seed=self.seed)
            df_raw = generator.generate(days=days, save_csv=True)
        else:
            df_raw = pd.read_csv(RAW_DATA_FILE)

        # 2. Preprocessing & Feature Engineering
        data_bundle = self.preprocessor.process_and_split(df=df_raw)
        X_train = data_bundle["X_train"]
        y_train = data_bundle["y_train"]
        X_test = data_bundle["X_test"]
        y_test = data_bundle["y_test"]
        feature_names = data_bundle["feature_names"]
        df_featured = data_bundle["df_featured"]
        transformer = data_bundle["transformer"]

        # 3. Fit Demand Predictor Ensemble
        self.demand_model.fit(X_train, y_train)

        # 4. Fit Price Elasticity Model
        self.elasticity_model.fit(df_featured)

        # 5. Evaluate Performance
        eval_report = self.evaluator.evaluate_models(
            ensemble_model=self.demand_model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            save_json=True
        )

        # 6. Save Models & Transformers
        MODEL_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        model_artifact = {
            "model": self.demand_model,
            "transformer": transformer,
            "feature_names": feature_names,
            "training_metrics": eval_report
        }
        joblib.dump(model_artifact, MODEL_CHECKPOINT_FILE)
        logger.info(f"Saved Demand Predictor artifact to {MODEL_CHECKPOINT_FILE}")

        joblib.dump(self.elasticity_model, ELASTICITY_MODEL_FILE)
        logger.info(f"Saved Price Elasticity model to {ELASTICITY_MODEL_FILE}")

        logger.info("--- Training Pipeline Completed Successfully ---")
        return {
            "evaluation_report": eval_report,
            "model_path": str(MODEL_CHECKPOINT_FILE),
            "elasticity_path": str(ELASTICITY_MODEL_FILE)
        }


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_pipeline(force_regenerate=True)
