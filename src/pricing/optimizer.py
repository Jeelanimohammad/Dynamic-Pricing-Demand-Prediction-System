"""
Constrained Price Optimization Engine for Revenue and Profit Maximization.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import minimize_scalar
from src.config import (
    DEFAULT_MIN_MARGIN_PCT,
    DEFAULT_MAX_PRICE_CHANGE,
    DEFAULT_SEARCH_GRANULARITY
)
from src.models.elasticity_model import PriceElasticityModel
from src.utils.metrics import calculate_financial_metrics
from src.utils.logger import get_logger

logger = get_logger("price_optimizer")


class PriceOptimizer:
    """
    Finds optimal price point maximizing profit or revenue under business guardrails.
    """

    def __init__(
        self,
        demand_model=None,
        transformer=None,
        elasticity_model: Optional[PriceElasticityModel] = None
    ):
        self.demand_model = demand_model
        self.transformer = transformer
        self.elasticity_model = elasticity_model or PriceElasticityModel()

    def optimize_price(
        self,
        product_features: Dict[str, Any],
        objective: str = "profit",  # 'profit', 'revenue', or 'balanced'
        min_margin_pct: float = DEFAULT_MIN_MARGIN_PCT,
        max_price_change: float = DEFAULT_MAX_PRICE_CHANGE,
        search_points: int = DEFAULT_SEARCH_GRANULARITY
    ) -> Dict[str, Any]:
        """
        Finds the global optimum price within feasible bounds.
        """
        current_price = float(product_features["price"])
        cost_price = float(product_features["cost_price"])
        competitor_price = float(product_features.get("competitor_price", current_price))
        product_id = product_features.get("product_id", "UNKNOWN")

        # 1. Establish Feasible Price Boundaries
        # Price cannot fall below cost + min_margin
        min_cost_floor = cost_price * (1.0 + min_margin_pct)
        # Price change limit
        min_allowed_price = max(min_cost_floor, current_price * (1.0 - max_price_change))
        max_allowed_price = current_price * (1.0 + max_price_change)

        # Ensure valid interval
        if min_allowed_price >= max_allowed_price:
            min_allowed_price = min_cost_floor
            max_allowed_price = max(min_allowed_price * 1.05, current_price * 1.20)

        # 2. Generate Search Grid
        candidate_prices = np.linspace(min_allowed_price, max_allowed_price, search_points)

        # 3. Simulate demand and financial outcomes along grid
        curve_data = self.simulate_price_curve(
            product_features=product_features,
            price_grid=candidate_prices
        )

        # 4. Find Optimum based on objective
        if objective == "profit":
            best_idx = np.argmax(curve_data["expected_profits"])
        elif objective == "revenue":
            best_idx = np.argmax(curve_data["expected_revenues"])
        else:  # 'balanced': 70% profit, 30% revenue normalized
            norm_profit = (curve_data["expected_profits"] - np.min(curve_data["expected_profits"])) / (np.ptp(curve_data["expected_profits"]) + 1e-6)
            norm_rev = (curve_data["expected_revenues"] - np.min(curve_data["expected_revenues"])) / (np.ptp(curve_data["expected_revenues"]) + 1e-6)
            score = 0.7 * norm_profit + 0.3 * norm_rev
            best_idx = np.argmax(score)

        optimal_price = round(float(candidate_prices[best_idx]), 2)
        optimal_demand = round(float(curve_data["predicted_demands"][best_idx]), 2)
        optimal_revenue = round(float(curve_data["expected_revenues"][best_idx]), 2)
        optimal_profit = round(float(curve_data["expected_profits"][best_idx]), 2)
        optimal_margin_pct = round(((optimal_price - cost_price) / max(optimal_price, 1e-6)) * 100, 2)

        # Current baseline financial metrics for uplift comparison
        base_demand = self.predict_demand_at_price(product_features, current_price)
        base_financials = calculate_financial_metrics(current_price, cost_price, base_demand)

        revenue_uplift_pct = round(((optimal_revenue - base_financials["expected_revenue"]) / max(base_financials["expected_revenue"], 1e-6)) * 100, 2)
        profit_uplift_pct = round(((optimal_profit - base_financials["expected_profit"]) / max(base_financials["expected_profit"], 1e-6)) * 100, 2)
        price_change_pct = round(((optimal_price - current_price) / current_price) * 100, 2)

        # Fetch product elasticity info
        elasticity_info = self.elasticity_model.get_product_elasticity(product_id)

        return {
            "product_id": product_id,
            "objective": objective,
            "current_price": current_price,
            "optimal_price": optimal_price,
            "price_change_pct": price_change_pct,
            "cost_price": cost_price,
            "competitor_price": competitor_price,
            "min_allowed_price": round(min_allowed_price, 2),
            "max_allowed_price": round(max_allowed_price, 2),
            "baseline_metrics": base_financials,
            "optimal_metrics": {
                "predicted_demand": optimal_demand,
                "expected_revenue": optimal_revenue,
                "expected_profit": optimal_profit,
                "profit_margin_pct": optimal_margin_pct,
                "revenue_uplift_pct": revenue_uplift_pct,
                "profit_uplift_pct": profit_uplift_pct
            },
            "elasticity_info": elasticity_info,
            "price_curve": {
                "prices": [round(p, 2) for p in candidate_prices],
                "demands": [round(d, 2) for d in curve_data["predicted_demands"]],
                "revenues": [round(r, 2) for r in curve_data["expected_revenues"]],
                "profits": [round(pr, 2) for pr in curve_data["expected_profits"]]
            }
        }

    def predict_demand_at_price(
        self,
        base_features: Dict[str, Any],
        price: float
    ) -> float:
        """
        Predicts demand for arbitrary candidate price point using the ML pipeline
        or econometric elasticity fallback.
        """
        features_copy = dict(base_features)
        features_copy["price"] = price

        if self.demand_model is not None and self.transformer is not None:
            try:
                from src.data.feature_engineering import FeatureEngineer
                df_single = pd.DataFrame([features_copy])
                df_feat = FeatureEngineer.create_features(df_single)
                X_trans = self.transformer.transform(df_feat)
                pred = self.demand_model.predict(X_trans)[0]
                return max(0.0, float(pred))
            except Exception as e:
                logger.debug(f"ML predictor fallback to elasticity formula: {e}")

        # Fallback econometric elasticity calculation
        product_id = base_features.get("product_id", "")
        elast = self.elasticity_model.get_product_elasticity(product_id)["own_price_elasticity"]
        base_p = float(base_features.get("base_price", base_features.get("price", 100.0)))
        base_q = float(base_features.get("historical_demand_7d", 50.0))
        predicted = PriceElasticityModel.estimate_demand_shift(base_p, price, base_q, elast)
        return max(0.0, float(predicted))

    def simulate_price_curve(
        self,
        product_features: Dict[str, Any],
        price_grid: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generates demand, revenue, and profit curves across candidate price range.
        """
        cost = float(product_features["cost_price"])
        demands = []
        revenues = []
        profits = []

        for p in price_grid:
            d = self.predict_demand_at_price(product_features, p)
            rev = p * d
            prof = (p - cost) * d
            demands.append(d)
            revenues.append(rev)
            profits.append(prof)

        return {
            "predicted_demands": np.array(demands),
            "expected_revenues": np.array(revenues),
            "expected_profits": np.array(profits)
        }
