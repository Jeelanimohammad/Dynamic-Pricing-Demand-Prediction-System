"""
Tests for price optimizer and pricing strategies.
"""
import pytest
from src.pricing.optimizer import PriceOptimizer
from src.pricing.strategies import PricingStrategyEngine
from src.pricing.simulator import PricingSimulator


def test_pricing_strategy_surge():
    surge = PricingStrategyEngine.apply_surge_pricing(
        base_price=100.0,
        stock_level=20,
        demand_forecast=50.0,
        is_peak_period=True
    )
    assert surge["recommended_price"] > 100.0
    assert surge["surge_multiplier"] > 1.0


def test_pricing_strategy_clearance():
    clearance = PricingStrategyEngine.apply_clearance_markdown(
        current_price=100.0,
        cost_price=40.0,
        stock_level=600,
        days_in_inventory=100,
        min_margin_pct=0.10
    )
    assert clearance["recommended_price"] < 100.0
    assert clearance["recommended_price"] >= 44.0  # Floor price: 40 * 1.10


def test_optimizer_bounds_enforcement():
    optimizer = PriceOptimizer()
    features = {
        "product_id": "ELEC_001",
        "price": 100.0,
        "cost_price": 50.0,
        "competitor_price": 105.0,
        "stock_level": 200,
        "historical_demand_7d": 40.0
    }

    opt_res = optimizer.optimize_price(
        product_features=features,
        objective="profit",
        min_margin_pct=0.20,
        max_price_change=0.25
    )

    # Optimal price must strictly respect bounds
    assert opt_res["optimal_price"] >= opt_res["min_allowed_price"]
    assert opt_res["optimal_price"] <= opt_res["max_allowed_price"]
    assert opt_res["optimal_metrics"]["expected_profit"] >= 0


def test_simulator_ab_testing():
    simulator = PricingSimulator()
    features = {
        "price": 100.0,
        "cost_price": 40.0,
        "historical_demand_7d": 50.0
    }
    ab_res = simulator.simulate_ab_test(
        product_features=features,
        control_price=100.0,
        treatment_price=110.0,
        sample_size=500
    )

    assert "control_group_A" in ab_res
    assert "treatment_group_B" in ab_res
    assert "comparison" in ab_res
    assert "p_value" in ab_res["comparison"]
