"""
Tests for demand forecasting models and price elasticity estimation.
"""
import pytest
import numpy as np
import pandas as pd
from src.models.demand_predictor import DemandPredictorEnsemble
from src.models.elasticity_model import PriceElasticityModel
from src.data.generator import MarketDataGenerator


def test_elasticity_model_fit():
    generator = MarketDataGenerator(seed=42)
    df = generator.generate(days=30, save_csv=False)

    elast_model = PriceElasticityModel()
    elast_model.fit(df)

    # Elasticity for products should be negative
    profile = elast_model.get_product_elasticity("ELEC_001")
    assert "own_price_elasticity" in profile
    assert profile["own_price_elasticity"] < 0
    assert "cross_price_elasticity" in profile
    assert isinstance(profile["cross_price_elasticity"], (int, float))


def test_arc_elasticity():
    # Price rises from 100 to 120 (+20%), demand drops from 50 to 40 (-20%)
    ped = PriceElasticityModel.calculate_arc_elasticity(100, 50, 120, 40)
    assert pytest.approx(ped, 0.1) == -1.22


def test_demand_predictor_ensemble():
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = np.random.uniform(10, 100, size=50)

    ensemble = DemandPredictorEnsemble(seed=42)
    ensemble.fit(X, y)

    preds = ensemble.predict(X)
    assert len(preds) == 50
    assert (preds >= 0).all()

    sub_preds = ensemble.predict_individual(X)
    assert "gradient_boosting" in sub_preds
    assert "random_forest" in sub_preds
    assert "elastic_net" in sub_preds
