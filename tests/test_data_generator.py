"""
Tests for market data generation pipeline.
"""
import pytest
import pandas as pd
from src.data.generator import MarketDataGenerator, CATALOG_DEFINITIONS


def test_catalog_definitions():
    assert len(CATALOG_DEFINITIONS) >= 10
    for prod in CATALOG_DEFINITIONS:
        assert "product_id" in prod
        assert "base_price" in prod
        assert "base_cost" in prod
        assert prod["base_price"] > prod["base_cost"]
        assert prod["base_elasticity"] < 0  # PED must be negative


def test_market_data_generator_output():
    generator = MarketDataGenerator(seed=42)
    df = generator.generate(days=14, save_csv=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 14 * len(CATALOG_DEFINITIONS)
    assert "date" in df.columns
    assert "demand_units" in df.columns
    assert "revenue" in df.columns
    assert "profit" in df.columns

    # Verify no NaN values
    assert df.isnull().sum().sum() == 0
    # Demand cannot be negative
    assert (df["demand_units"] >= 0).all()
    # Prices must be strictly positive
    assert (df["price"] > 0).all()
