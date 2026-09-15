"""
Tests for FastAPI endpoints using TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_products():
    response = client.get("/api/products")
    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert len(data["products"]) > 0


def test_api_predict_demand():
    payload = {
        "product_id": "ELEC_001",
        "category": "Electronics",
        "price": 199.99,
        "cost_price": 85.00,
        "competitor_price": 205.00,
        "stock_level": 250,
        "discount_percent": 0.0,
        "rating": 4.5,
        "review_count": 350,
        "ad_spend_usd": 150.0,
        "holiday_effect": 0.0,
        "customer_segment": "Standard",
        "season": "Fall",
        "month": 10,
        "day_of_week": 2,
        "is_weekend": 0,
        "historical_demand_7d": 45.0,
        "historical_demand_30d": 42.0
    }
    response = client.post("/api/predict-demand", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_demand_units" in data
    assert data["predicted_demand_units"] >= 0
    assert "expected_revenue" in data
    assert "expected_profit" in data


def test_api_optimize_price():
    payload = {
        "features": {
            "product_id": "ELEC_001",
            "category": "Electronics",
            "price": 199.99,
            "cost_price": 85.00,
            "competitor_price": 205.00,
            "stock_level": 250,
            "discount_percent": 0.0,
            "rating": 4.5,
            "review_count": 350,
            "ad_spend_usd": 150.0,
            "holiday_effect": 0.0,
            "customer_segment": "Standard",
            "season": "Fall",
            "month": 10,
            "day_of_week": 2,
            "is_weekend": 0,
            "historical_demand_7d": 45.0,
            "historical_demand_30d": 42.0
        },
        "objective": "profit",
        "min_margin_pct": 0.15,
        "max_price_change_pct": 0.30
    }
    response = client.post("/api/optimize-price", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_price" in data
    assert data["optimal_price"] > 0
    assert "price_curve" in data
