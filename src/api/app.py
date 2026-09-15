"""
FastAPI REST API Service for Dynamic Pricing & Demand Prediction.
"""
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src.config import (
    MODEL_CHECKPOINT_FILE,
    ELASTICITY_MODEL_FILE,
    METRICS_FILE,
    RAW_DATA_FILE
)
from src.api.schemas import (
    ProductFeatureInput,
    DemandPredictionResponse,
    OptimizationRequest,
    OptimizationResponse,
    WhatIfRequest,
    ABTestRequest,
    RetrainRequest
)
from src.data.generator import CATALOG_DEFINITIONS, MarketDataGenerator
from src.models.train import ModelTrainer
from src.models.elasticity_model import PriceElasticityModel
from src.pricing.optimizer import PriceOptimizer
from src.pricing.strategies import PricingStrategyEngine
from src.pricing.simulator import PricingSimulator
from src.utils.metrics import calculate_financial_metrics
from src.utils.logger import get_logger

logger = get_logger("fastapi_app")

# Global instances
app_state: Dict[str, Any] = {
    "model_artifact": None,
    "elasticity_model": None,
    "optimizer": None,
    "simulator": None
}


def load_or_train_models():
    """Ensures model artifacts exist and loads them into memory."""
    if not MODEL_CHECKPOINT_FILE.exists() or not ELASTICITY_MODEL_FILE.exists():
        logger.info("Artifacts not found. Triggering automated model training...")
        trainer = ModelTrainer()
        trainer.train_pipeline(force_regenerate=True)

    logger.info("Loading serialized models into memory...")
    app_state["model_artifact"] = joblib.load(MODEL_CHECKPOINT_FILE)
    app_state["elasticity_model"] = joblib.load(ELASTICITY_MODEL_FILE)

    demand_model = app_state["model_artifact"]["model"]
    transformer = app_state["model_artifact"]["transformer"]
    elasticity_model = app_state["elasticity_model"]

    app_state["optimizer"] = PriceOptimizer(
        demand_model=demand_model,
        transformer=transformer,
        elasticity_model=elasticity_model
    )
    app_state["simulator"] = PricingSimulator(optimizer=app_state["optimizer"])
    logger.info("All models and optimizers initialized successfully.")


def ensure_models_loaded():
    """Safety check to ensure models are initialized before handling requests."""
    if app_state["optimizer"] is None:
        load_or_train_models()


# Call eagerly on import
try:
    load_or_train_models()
except Exception as e:
    logger.warning(f"Initial eager load deferred: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_or_train_models()
    yield
    # Shutdown
    logger.info("Shutting down API server...")


app = FastAPI(
    title="Dynamic Pricing & Demand Prediction API",
    description="High-performance REST API for real-time demand forecasting, price elasticity estimation, and profit/revenue optimization.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    return {
        "system": "Dynamic Pricing & Demand Prediction System",
        "status": "active",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/api/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": app_state["model_artifact"] is not None,
        "elasticity_model_loaded": app_state["elasticity_model"] is not None
    }


@app.get("/api/products", tags=["Catalog"])
def get_product_catalog():
    """Returns predefined product catalog templates."""
    return {"products": CATALOG_DEFINITIONS}


@app.post("/api/predict-demand", response_model=DemandPredictionResponse, tags=["Forecasting"])
def predict_demand(payload: ProductFeatureInput):
    """
    Predicts demand volume and expected financial return for given product conditions and price.
    """
    try:
        ensure_models_loaded()
        data_dict = payload.model_dump()
        price = float(data_dict["price"])
        cost = float(data_dict["cost_price"])

        optimizer: PriceOptimizer = app_state["optimizer"]
        predicted_demand = optimizer.predict_demand_at_price(data_dict, price)

        financials = calculate_financial_metrics(price, cost, predicted_demand)

        # Get sub-model predictions
        from src.data.feature_engineering import FeatureEngineer
        df_feat = FeatureEngineer.create_features(pd.DataFrame([data_dict]))
        X_trans = app_state["model_artifact"]["transformer"].transform(df_feat)
        submodels = app_state["model_artifact"]["model"].predict_individual(X_trans)
        submodel_summary = {k: round(float(v[0]), 2) for k, v in submodels.items()}

        return {
            "product_id": data_dict["product_id"],
            "price": price,
            "predicted_demand_units": round(predicted_demand, 2),
            "expected_revenue": financials["expected_revenue"],
            "expected_profit": financials["expected_profit"],
            "unit_margin": financials["unit_margin"],
            "profit_margin_pct": financials["profit_margin_pct"],
            "submodel_predictions": submodel_summary
        }
    except Exception as e:
        logger.error(f"Error in predict_demand: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/optimize-price", response_model=OptimizationResponse, tags=["Optimization"])
def optimize_price(payload: OptimizationRequest):
    """
    Computes optimal price maximizing profit, revenue, or balanced objective with constraints.
    """
    try:
        ensure_models_loaded()
        optimizer: PriceOptimizer = app_state["optimizer"]
        result = optimizer.optimize_price(
            product_features=payload.features.model_dump(),
            objective=payload.objective,
            min_margin_pct=payload.min_margin_pct,
            max_price_change=payload.max_price_change_pct,
            search_points=payload.search_points
        )
        return result
    except Exception as e:
        logger.error(f"Error in optimize_price: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/elasticity/{product_id}", tags=["Elasticity"])
def get_elasticity_profile(product_id: str):
    """
    Fetches econometric own-price and cross-price elasticity for a product.
    """
    ensure_models_loaded()
    elasticity_model: PriceElasticityModel = app_state["elasticity_model"]
    return elasticity_model.get_product_elasticity(product_id)


@app.post("/api/simulate/what-if", tags=["Simulation"])
def simulate_what_if(payload: WhatIfRequest):
    """
    Evaluates demand, revenue, and profit under simulated market shifts.
    """
    try:
        ensure_models_loaded()
        simulator: PricingSimulator = app_state["simulator"]
        return simulator.run_what_if_scenario(
            base_features=payload.features.model_dump(),
            price_delta_pct=payload.price_delta_pct,
            competitor_delta_pct=payload.competitor_delta_pct,
            ad_spend_multiplier=payload.ad_spend_multiplier,
            holiday_effect=payload.holiday_effect
        )
    except Exception as e:
        logger.error(f"Error in simulate_what_if: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/simulate/ab-test", tags=["Simulation"])
def simulate_ab_test(payload: ABTestRequest):
    """
    Simulates a randomized A/B trial comparing static baseline price vs dynamic optimized price.
    """
    try:
        ensure_models_loaded()
        simulator: PricingSimulator = app_state["simulator"]
        return simulator.simulate_ab_test(
            product_features=payload.features.model_dump(),
            control_price=payload.control_price,
            treatment_price=payload.treatment_price,
            sample_size=payload.sample_size
        )
    except Exception as e:
        logger.error(f"Error in simulate_ab_test: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/retrain", tags=["Model Management"])
def retrain_models(payload: RetrainRequest):
    """
    Retrains the demand forecasting and price elasticity models.
    """
    try:
        trainer = ModelTrainer()
        result = trainer.train_pipeline(
            force_regenerate=payload.force_regenerate,
            days=payload.simulation_days
        )
        load_or_train_models()
        return {
            "status": "success",
            "message": "Models retrained and reloaded successfully.",
            "metrics": result["evaluation_report"]
        }
    except Exception as e:
        logger.error(f"Error in retrain_models: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
