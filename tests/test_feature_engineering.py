"""
Tests for feature engineering transformations.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.feature_engineering import FeatureEngineer


def test_feature_engineering_transforms():
    sample_data = pd.DataFrame([{
        "price": 100.0,
        "cost_price": 40.0,
        "competitor_price": 95.0,
        "discount_percent": 10.0,
        "stock_level": 250,
        "historical_demand_7d": 50.0,
        "month": 6,
        "day_of_week": 3,
        "ad_spend_usd": 200.0,
        "rating": 4.5,
        "review_count": 500,
        "demand_units": 45
    }])

    featured = FeatureEngineer.create_features(sample_data)

    assert "price_ratio_competitor" in featured.columns
    assert "price_cost_margin" in featured.columns
    assert "month_sin" in featured.columns
    assert "month_cos" in featured.columns
    assert "log_price" in featured.columns

    # Verify margin calculation: (100 - 40) / 100 = 0.60
    assert pytest.approx(featured["price_cost_margin"].iloc[0], 0.01) == 0.60
    # Verify price ratio: 100 / 95 = 1.0526
    assert pytest.approx(featured["price_ratio_competitor"].iloc[0], 0.01) == 1.0526
