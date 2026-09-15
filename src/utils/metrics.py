"""
Financial, pricing and forecasting evaluation metrics.
"""
import numpy as np
from typing import Dict, Any

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard forecasting performance metrics:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - WAPE (Weighted Absolute Percentage Error)
    - R2 Score
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0.0)  # Demand cannot be negative

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    # MAPE with epsilon safeguard against zero demand
    eps = 1e-6
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)

    # WAPE (Weighted Absolute Percentage Error) = sum(|y - y_hat|) / sum(y)
    sum_true = np.sum(np.abs(y_true))
    wape = float((np.sum(np.abs(y_true - y_pred)) / max(sum_true, eps)) * 100)

    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / max(ss_tot, eps)))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_percent": round(mape, 2),
        "wape_percent": round(wape, 2),
        "r2_score": round(r2, 4)
    }

def calculate_financial_metrics(
    price: float,
    cost_price: float,
    predicted_demand: float
) -> Dict[str, float]:
    """
    Computes expected financial outcomes for a given price point:
    - Revenue = Price * Demand
    - Total Cost = Cost * Demand
    - Profit = Revenue - Total Cost
    - Profit Margin % = (Profit / Revenue) * 100
    - Unit Margin = Price - Cost
    """
    price = float(price)
    cost = float(cost_price)
    demand = max(float(predicted_demand), 0.0)

    revenue = price * demand
    total_cost = cost * demand
    profit = revenue - total_cost
    unit_margin = price - cost
    margin_pct = (unit_margin / max(price, 1e-6)) * 100.0

    return {
        "price": round(price, 2),
        "cost_price": round(cost, 2),
        "unit_margin": round(unit_margin, 2),
        "predicted_demand": round(demand, 2),
        "expected_revenue": round(revenue, 2),
        "expected_profit": round(profit, 2),
        "profit_margin_pct": round(margin_pct, 2)
    }
